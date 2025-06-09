#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=bhaskarsteve
#SBATCH --mail-type=END
#SBATCH --exclude=gnode[001-012,015-017,019-023,025-035]
#SBATCH --output=op1.txt

python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 200 --name "kitti_yolov5l_200"
cp -r runs/train/kitti_yolov5l_200/weights/best.pt ../pretrained/kitti/
