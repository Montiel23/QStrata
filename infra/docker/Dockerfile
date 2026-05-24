FROM python:3.11-slim

# System dependencies required by OpenCV and pydicom
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python packages — single layer, CPU-only, EDA scope
# torch / QML / GPU dependencies are intentionally excluded
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
        tqdm

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/workspace"]
