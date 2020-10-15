# Pytorch YOLOv3
A PyTorch implementation of YOLOv3 base on ultralytics-yolov3, with support for training, inference and evaluation.

# Installation
$ git clone https://github.com/ghang-ouc/Pytorch-YOLOv3.git  
$ cd PyTorch-YOLOv3/  
$ sudo pip3 install -r requirements.txt  

# Install Nvidia/apex  
https://zhuanlan.zhihu.com/p/146515828  
git clone https://github.com/NVIDIA/apex  
cd apex  
python3 setup.py install

# Train
python3 train.py --data data/obj.data  --cfg cfg/yolov3.cfg
