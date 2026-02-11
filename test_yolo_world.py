import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize model
print("Initializing YOLO-World model...")
model = YOLOWorld(model_id="yolo_world/l")

# Configuration
image_path = "IMG_8144.png"
classes = ["plate", "license plate"]
confidence_threshold = 0.01

print(f"Running inference on {image_path} with classes: {classes}")

# Run inference
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image {image_path}")
    exit(1)

results = model.infer(image, text=classes, confidence=confidence_threshold)
# Wrapper logic might return a list or valid object, checking...
if isinstance(results, list):
    result = results[0]
else:
    result = results

# Visualize
print("Processing results...")
detections = sv.Detections.from_inference(result)
print(f"Found {len(detections)} detections.")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{classes[class_id]} {confidence:0.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

output_path = "yolo_world_result.jpg"
cv2.imwrite(output_path, annotated_image)
print(f"Result saved to {output_path}")

# Print detection details
for i in range(len(detections)):
    print(f" - {labels[i]}")
