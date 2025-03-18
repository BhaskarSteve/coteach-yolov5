# python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 200
# python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 200 


# mv /home/cdq2kor/Desktop/code/datasets/kitti/labels /home/cdq2kor/Desktop/code/datasets/kitti/origlabels
# mv /home/cdq2kor/Desktop/code/datasets/kitti/autolabels /home/cdq2kor/Desktop/code/datasets/kitti/labels

# mv /home/cdq2kor/Desktop/code/datasets/bdd/labels /home/cdq2kor/Desktop/code/datasets/bdd/origlabels
# mv /home/cdq2kor/Desktop/code/datasets/bdd/autolabels /home/cdq2kor/Desktop/code/datasets/bdd/labels

# python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 200 
# python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 200 

python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.2
python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.2

python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.1
python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.1

python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.4
python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.4

mv /home/cdq2kor/Desktop/code/datasets/kitti/labels /home/cdq2kor/Desktop/code/datasets/kitti/autolabels
mv /home/cdq2kor/Desktop/code/datasets/kitti/origlabels /home/cdq2kor/Desktop/code/datasets/kitti/labels

mv /home/cdq2kor/Desktop/code/datasets/bdd/labels /home/cdq2kor/Desktop/code/datasets/bdd/autolabels
mv /home/cdq2kor/Desktop/code/datasets/bdd/origlabels /home/cdq2kor/Desktop/code/datasets/bdd/labels