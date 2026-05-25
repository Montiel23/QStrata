FROM python:3.11-slim

# System dependencies required by OpenCV and pydicom
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python packages — single layer, CPU-only, EDA scope
RUN pip install --no-cache-dir \
        jupyterlab \
        notebook \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn \
        opencv-python-headless \
        pydicom \
        pillow \
        tqdm \
        medmnist

# CPU-only PyTorch — installed from the dedicated CPU wheel index (no CUDA)
RUN pip install --no-cache-dir \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--IdentityProvider.token=", \
     "--ServerApp.password="]
