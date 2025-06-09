#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=bhaskarsteve
#SBATCH --mail-type=END
#SBATCH --exclude=gnode[001-012,015-017,019-023,025-035]
#SBATCH --output=op2.txt

python train.py --data acdc.yaml --weights yolov5l.pt --img 640 --epochs 200 --name "acdc_yolov5l_200"
cp -r runs/train/acdc_yolov5l_200/weights/best.pt ../pretrained/acdc/
