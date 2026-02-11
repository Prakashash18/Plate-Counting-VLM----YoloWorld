# Deploying to RunPod

This guide explains how to deploy your **Streamlit Plate Counter** to RunPod to leverage high-performance GPUs.

> **⚠️ IMPORTANT NOTE ON WEBCAMS**: 
> RunPod is a remote cloud server. It **cannot access your local laptop webcam** directly. 
> When running on RunPod, you should use the **"Upload Video"** feature. If you try to use "Webcam", it will fail (or try to open a USB device on the server rack, which doesn't exist).

## 1. Prepare Docker Image
You need to build and push your Docker image to a registry (like Docker Hub) so RunPod can pull it.

### Option A: Build Locally (If you have Docker installed)
```bash
# Login to Docker Hub
docker login

# Build the image (replace 'yourusername' with your Docker Hub username)
docker build -t yourusername/plate-counter:latest .

# Push to Docker Hub
docker push yourusername/plate-counter:latest
```

## 2. Deploy on RunPod

1.  **Login** to [RunPod.io](https://www.runpod.io/).
2.  Go to **Pods** > **Deploy**.
3.  Select a GPU (e.g., **RTX 3070** or **A4000** are good budget options for inference).
4.  **Template Configuration**:
    *   Click **Select Image** or **Edit Template**.
    *   **Image Name**: Enter `yourusername/plate-counter:latest` (the one you just pushed).
    *   **Container Disk**: 10 GB (should be plenty).
    *   **Volume Disk**: 10 GB.
5.  **Expose Port**: 
    *   Expand **Port Mapping**.
    *   Add `8501` (Container Port) -> `8501` (Public Port, or leave auto).
    *   *Tip: RunPod requires HTTP/TCP ports exposed.*
6.  **Deploy**.

## 3. Access the App
1.  Once the Pod says **Running**, click the **Connect** button.
2.  Find the **HTTP Service** mapped to port `8501`.
3.  Click the link. Your Streamlit app is now running on a cloud GPU!

## 4. Run Inference
*   Select **"Upload Video"** in the sidebar.
*   Upload a video file.
*   The inference speed should be significantly faster than your local CPU.
