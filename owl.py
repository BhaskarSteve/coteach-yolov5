import os
import argparse
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection

def convert_to_yolo(box, image_width, image_height):
    """
    Convert bounding box from Pascal VOC (xmin, ymin, xmax, ymax)
    to YOLO format (x_center, y_center, width, height) normalized by image size.
    """
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    box_width = xmax - xmin
    box_height = ymax - ymin
    # Normalize by image dimensions
    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height
    return x_center, y_center, box_width, box_height

def main(input_dir, output_dir, threshold, prompts):
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Load the processor and model.
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # Prepare the text queries. The prompts argument is a list of strings.
    texts = [prompts]

    # Process each image in the input directory.
    # for filename in os.listdir(input_dir):
    for filename in tqdm(os.listdir(input_dir), desc="Processing Files"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error opening {filename}: {e}")
                continue

            image_width, image_height = image.size

            # Prepare input for the processor.
            inputs = processor(text=texts, images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            # Rescale box predictions to the original image size.
            target_sizes = torch.Tensor([image.size[::-1]])  # [height, width]
            results = processor.post_process_object_detection(outputs=outputs,
                                                              target_sizes=target_sizes,
                                                              threshold=threshold)
            # There is one result per image; retrieve the boxes, scores, and labels.
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

            # Write the detections to a text file in YOLO format.
            output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
            with open(output_file, "w") as f:
                for box, score, label in zip(boxes, scores, labels):
                    # Convert box to YOLO format.
                    x_center, y_center, box_width, box_height = convert_to_yolo(box.tolist(),
                                                                                 image_width,
                                                                                 image_height)
                    class_id = label.item()  # Corresponds to the index in the prompts list.
                    # Write in the format: class_id confidence x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f} {score.item():.3f}\n")
            # print(f"Processed {filename} -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label objects in images and save YOLO format detections.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input image directory.")
    parser.add_argument("--output_dir", type=str, default="labels",
                        help="Path to the output directory for YOLO format labels.")
    parser.add_argument("--threshold", type=float, default=0.001,
                        help="Detection threshold for post-processing.")
    parser.add_argument("--prompts", type=str, default="a photo of a cat,a photo of a dog",
                        help="Comma-separated list of prompts to use for object detection.")
    args = parser.parse_args()
    
    # Split the comma-separated prompts and strip any extra spaces.
    prompts = [prompt.strip() for prompt in args.prompts.split(",")]
    
    main(args.input_dir, args.output_dir, args.threshold, prompts)