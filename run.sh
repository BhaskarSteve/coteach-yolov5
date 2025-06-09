# python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 200
# python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 200 

# python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 200 
# python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 200 

python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.2
python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.2

python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.1
python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.1

python train.py --data kitti.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.4
python train.py --data bdd.yaml --weights yolov5l.pt --img 640 --epochs 300 --coteaching --forget-rate 0.4