import argparse
import torch.distributed as dist
import torch.optim as optim
import test
from models import *
from utils.datasets import *
from utils.utils import *
mixed_precision = True
try:  # Mixed precision
    from apex import amp
except:
    mixed_precision = False

# os.sep根据你所处的平台，自动采用相应的分隔符号
wdir = 'weights' + os.sep
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'
# 超参数
hyp = {'giou': 3.50,            # giou loss gain
       'cls': 37.4,             # cls loss gain
       'cls_pw': 1.0,           # cls BCELoss positive_weight
       'obj': 34.5,             # obj loss gain
       'obj_pw': 1.0,           # obj BCELoss positive_weight
       'iou_t': 0.215,          # iou training threshold
       'lr': 0.001,             # initial learning rate (SGD=5E-3, Adam=5E-4)
       'momentum': 0.9,         # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,         # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,         # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,          # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.20,           # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,     # image rotation (+/- deg)
       'translate': 0.05 * 0,   # image translation (+/- fraction)
       'scale': 0.05 * 0,       # image scale (+/- gain)
       'shear': 0.641 * 0,      # image shear (+/- deg)
       }

def train(hyp):
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    img_size = opt.img_size  # img sizes
    imgsz_test = opt.img_size

    # Image Sizes
    gs = 32  # 32倍降采样
    assert math.fmod(img_size, gs) == 0, '--img-size %g must be a %g-multiple' % (img_size, gs)  # fmod(x,y)取x/y的余数
    # 多尺度训练
    if opt.multi_scale:
        imgsz_min = round(img_size / 32 / 1.5) + 1
        imgsz_max = round(img_size / 32 * 1.5) - 1
        img_size = imgsz_max * 32  # initialize with max size
        print('Using multi-scale %g - %g' % (imgsz_min * 32, img_size))

    # Configure run
    init_seeds()
    # 解析data文件
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    # number of classes
    nc = int(data_dict['classes'])
    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr'])
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            # numel()函数：返回数组中元素的个数
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. "  \
                % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights'
        load_darknet_weights(model, weights)

    if opt.freeze_layers:
        # isinstance()函数判断一个对象是否是一个已知的类型
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)]                                                                                                                      
        freeze_layer_indices = [x for x in range(len(model.module_list)) if                                                                                                         
                                (x not in output_layer_indices) and                                                                                                               
                                (x - 1 not in output_layer_indices)]
        for idx in freeze_layer_indices:                                                                                                                                             
            for parameter in model.module_list[idx].parameters():                                                                                                                    
                parameter.requires_grad_(False)  # 输出层不更新梯度

    # Mixed precision training
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Learning rate setup
    def adjust_learning_rate(optimizer, epoch, iteration, burn_in):
        """调整学习率进行warm up和学习率衰减
        """
        learning_rate = hyp['lr']
        if iteration < burn_in:
            # warm up
            learning_rate = 1e-6 + learning_rate * pow(iteration / burn_in, 2)
        else:
            if epoch > opt.epochs * 0.7:
                learning_rate = learning_rate * 0.1
            if epoch > opt.epochs * 0.9:
                learning_rate = learning_rate * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        return learning_rate

    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  )
    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 ),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # 类别数
    model.hyp = hyp  # 超参数
    # Model EMA
    ema = torch_utils.ModelEMA(model)  # EMA 指数移动平均,对模型的参数做平均,提高测试指标并增加模型鲁棒

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()

    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    print('-----------------------------------------------------------------------------------')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(4).to(device)  # mean losses
        print(('%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0 归一化
            targets = targets.to(device)

            # 调整学习率，进行warm up和学习率衰减
            lr = adjust_learning_rate(optimizer, epoch, ni, n_burn)
            if i == 0:
                print('\nlearning rate:', lr)

            # Multi-Scale
            if opt.multi_scale:
                if ni / accumulate % 10 == 0:  # adjust img_size (67% - 150%) every 10 batch
                    img_size = random.randrange(imgsz_min, imgsz_max+ 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # Math.ceil()  "向上取整", 即小数部分直接舍去, 并向正数部分进1
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)  # 上采样，线性插值
            # Forward
            pred = model(imgs)
            # Loss
            loss, loss_items = compute_loss(pred, targets, model)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)
            # Plot
            if ni < 1:
                f = 'train_batch%g.jpg' % i  # filename
                plot_images(images=imgs, targets=targets, paths=paths, fname=f)
            # end batch ------------------------------------------------------------------------------------------------

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      imgsz=imgsz_test,
                                      model=ema.ema,
                                      dataloader=testloader,
                                      multi_label=ni > n_burn)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt.nosave) or final_epoch
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}
            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    plot_results(cls=nc)  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='data/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path(weights/last.pt)')
    parser.add_argument('--data', type=str, default='data/obj.data', help='*.data path')
    parser.add_argument('--multi-scale', type=bool, default=True, help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
    opt = parser.parse_args()

    opt.weights = last if opt.resume and not opt.weights else opt.weights
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    # 加载gpu
    device = torch_utils.select_device(opt.device, apex=mixed_precision)
    if device.type == 'cpu':
        mixed_precision = False
    train(hyp)  # train normally
