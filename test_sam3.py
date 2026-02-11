import os
import sys
import requests
import cv2
import numpy as np
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("Error: ROBOFLOW_API_KEY not found in .env file.")
    sys.exit(1)

# Configuration
image_source = "IMG_8144.png" # URL or local file path
prompts = [
    { "type": "text", "text": "plate" },
    { "type": "text", "text": "person" }
]

print(f"Testing SAM 3 via Roboflow API using image: {image_source}")

try:
    # Prepare image payload
    image_payload = {}
    if os.path.exists(image_source):
        print("Detected local file. Encoding to base64...")
        with open(image_source, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            image_payload = {
                "type": "base64",
                "value": encoded_string
            }
    else:
        print("Assuming URL...")
        image_payload = {
            "type": "url",
            "value": image_source
        }

    # 1. Send Request
    print("Sending request to SAM 3 API...")
    response = requests.post(
        f"https://serverless.roboflow.com/sam3/concept_segment?api_key={api_key}",
        headers={
            "Content-Type": "application/json"
        },
        json={
            "format": "polygon", # Get polygons for drawing
            "image": image_payload,
            "prompts": prompts
        }
    )

    if response.status_code != 200:
        print(f"Error: API Request failed with status code {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()
    print("Inference successful!")
    
    # 2. Process and Visualize Results
    # Load image for visualization
    if image_payload["type"] == "base64":
         # Decode base64 to numpy array for OpenCV
        img_data = base64.b64decode(image_payload["value"])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        # Download image from URL
        img_req = requests.get(image_source)
        img_arr = np.asarray(bytearray(img_req.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

    prompt_results = result.get("prompt_results", [])
    for prompt_res in prompt_results:
        prompt_text = prompt_res.get("echo", {}).get("text", "unknown")
        predictions = prompt_res.get("predictions", [])
        print(f"Prompt: '{prompt_text}' found {len(predictions)} matches.")
        
        for pred in predictions:
            confidence = pred.get("confidence")
            masks = pred.get("masks", [])
            print(f"  - Confidence: {confidence}")
            
            # Draw polygons
            for polygon in masks:
                polygon_arr = np.array(polygon).astype(np.int32)
                cv2.polylines(img, [polygon_arr], True, (0, 255, 0), 2)
                # Put label
                x, y = polygon_arr[0]
                cv2.putText(img, f"{prompt_text} ({confidence:.2f})", (int(x), int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save output image
    output_filename = "sam3_result.jpg"
    cv2.imwrite(output_filename, img)
    print(f"Result image saved to {output_filename}")

except Exception as e:
    print(f"An error occurred: {e}")
