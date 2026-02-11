import requests
import shutil

url = "https://media.roboflow.com/notebooks/examples/dog.jpeg"
filename = "sample_image.jpg"

print(f"Downloading sample image from {url}...")
response = requests.get(url, stream=True)

if response.status_code == 200:
    with open(filename, 'wb') as f:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, f)
    print(f"Image successfully downloaded: {filename}")
else:
    print(f"Failed to download image. Status code: {response.status_code}")
