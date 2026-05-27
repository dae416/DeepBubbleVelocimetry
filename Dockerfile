FROM --platform=linux/amd64 tensorflow/tensorflow:1.14.0-py3

RUN pip install --quiet --upgrade pip && \
    pip install --quiet \
    "scikit-image==0.17.2" \
    tqdm \
    "opencv-python-headless==4.5.5.64" \
    Pillow \
    "matplotlib==3.3.4" \
    "pandas==1.1.5" \
    jupyter

WORKDIR /workspace
