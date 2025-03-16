#!/usr/bin/env python3
"""
YOLOv5 Model Evaluation Script
This script runs inference on a test set and saves predictions in two formats:
1. Evaluation format: Low confidence threshold to calculate mAP
2. Production format: Higher confidence for practical use
"""

import argparse
import os
import glob
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add YOLOv5 modules to path
sys.path.insert(0, os.path.abspath('./'))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Model Evaluation Script")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLOv5 model")
    parser.add_argument("--img-dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for inference")
    parser.add_argument("--output-dir", type=str, default="preds", help="Directory to save results")
    parser.add_argument("--eval-conf", type=float, default=0.001, help="Confidence threshold for evaluation")
    parser.add_argument("--prod-conf", type=float, default=0.25, help="Confidence threshold for production")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    return parser.parse_args()


def create_output_dirs(base_dir):
    """Create output directories for evaluation and production results"""
    eval_dir = os.path.join(base_dir, "evaluation")
    prod_dir = os.path.join(base_dir, "production")
    img_dir = os.path.join(base_dir, "visualization")
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(prod_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    return eval_dir, prod_dir, img_dir


def get_image_list(img_dir):
    """Get list of all images in the directory"""
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    image_list = []
    for ext in extensions:
        image_list.extend(glob.glob(os.path.join(img_dir, f'*.{ext}')))
        image_list.extend(glob.glob(os.path.join(img_dir, f'*.{ext.upper()}')))
    return sorted(image_list)


def save_detections(detections, output_path, img_name, class_names, img_shape):
    """Save detections in YOLO format: [class_id x_center y_center width height confidence]
    All coordinates are normalized to [0, 1]
    """
    # Create detection file path
    base_name = os.path.splitext(os.path.basename(img_name))[0]
    detection_file = os.path.join(output_path, f"{base_name}.txt")
    
    # Get image dimensions for normalization
    img_height, img_width = img_shape[:2]
    
    with open(detection_file, 'w') as f:
        for *xyxy, conf, cls in detections:
            # Convert from xyxy (top-left, bottom-right) to xywh (center, width, height)
            x1, y1, x2, y2 = [float(coord) for coord in xyxy]
            
            # Calculate normalized center coordinates, width and height
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Write in YOLO format: class_id x_center y_center width height confidence
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")


def visualize_detections(img, detections, output_path, class_names):
    """Visualize detections on the image and save it"""
    # Draw bounding boxes
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = xyxy
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        color = (0, 255, 0)  # Green color for bounding box
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save the image
    cv2.imwrite(output_path, img)


def load_coteaching_model(weights, device):
    """Load a co-teaching model checkpoint and extract the primary model"""
    print(f"Loading co-teaching model from {weights}...")
    ckpt = torch.load(weights, map_location=device)
    
    # Try various model keys that might exist in our co-teaching checkpoint
    if 'model1' in ckpt:
        print("Found model1 in checkpoint")
        model = ckpt['model1']
    elif 'ema1' in ckpt:
        print("Found ema1 in checkpoint")
        if hasattr(ckpt['ema1'], 'ema'):
            model = ckpt['ema1'].ema
        else:
            model = ckpt['ema1']
    elif 'model' in ckpt:
        print("Found standard model in checkpoint")
        model = ckpt['model']
    elif 'ema' in ckpt and ckpt['ema'] is not None:
        print("Found standard ema in checkpoint")
        model = ckpt['ema']
    else:
        raise RuntimeError(f"Could not find model in checkpoint. Keys: {list(ckpt.keys())}")
    
    # Get class names if available
    class_names = ckpt.get('names', {i: f'class{i}' for i in range(1000)})
    
    return model.float(), class_names


def preprocess_image(img_path, img_size=640, device='cpu'):
    """Preprocess image for inference"""
    # Load and resize image
    img0 = cv2.imread(img_path)  # BGR
    if img0 is None:
        raise ValueError(f"Could not read image {img_path}")
        
    # Padded resize
    img = letterbox(img0, img_size, stride=32, auto=True)[0]
    
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
        
    return img, img0


def run_inference(model, img, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000):
    """Run inference and apply NMS"""
    # Inference
    pred = model(img, augment=False)[0]
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, max_det=max_det)
    
    return pred


def process_predictions(pred, img, img0):
    """Process predictions and rescale boxes to original image size"""
    det = pred[0].clone()  # detections per image
    
    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
    
    return det


def main():
    args = parse_args()
    
    # Create output directories
    eval_dir, prod_dir, img_dir = create_output_dirs(args.output_dir)
    
    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    
    # Load model - try co-teaching approach first, if fails use regular approach
    try:
        model, class_names = load_coteaching_model(args.model, device)
    except Exception as e:
        print(f"Co-teaching model loading failed: {e}")
        print("Falling back to standard model loading...")
        model = attempt_load(args.model, device)
        class_names = model.names if hasattr(model, 'names') else {i: f'class{i}' for i in range(1000)}
    
    # Get list of test images
    img_list = get_image_list(args.img_dir)
    print(f"Found {len(img_list)} images for evaluation")
    
    # Process each image
    for img_path in tqdm(img_list, desc="Processing images"):
        # Get image name
        img_name = os.path.basename(img_path)
        
        try:
            # Preprocess image
            img, img0 = preprocess_image(img_path, args.img_size, device)
            
            # Run inference with evaluation confidence
            eval_pred = run_inference(model, img, conf_thres=args.eval_conf, iou_thres=args.iou)
            eval_det = process_predictions(eval_pred, img, img0)
            
            # Save evaluation detections
            save_detections(eval_det, eval_dir, img_name, class_names, img0.shape)
            
            # Run inference with production confidence
            prod_pred = run_inference(model, img, conf_thres=args.prod_conf, iou_thres=args.iou)
            prod_det = process_predictions(prod_pred, img, img0)
            
            # Save production detections
            save_detections(prod_det, prod_dir, img_name, class_names, img0.shape)
            
            # Visualize results
            vis_img = img0.copy()
            vis_path = os.path.join(img_dir, img_name)
            visualize_detections(vis_img, prod_det, vis_path, class_names)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Evaluation complete! Results saved to {args.output_dir}")
    print(f"- Evaluation results (low conf={args.eval_conf}): {eval_dir}")
    print(f"- Production results (high conf={args.prod_conf}): {prod_dir}")
    print(f"- Visualizations: {img_dir}")


if __name__ == "__main__":
    main()
