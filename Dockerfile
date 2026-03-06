FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0

WORKDIR /workspace

RUN git clone https://github.com/Xiaojiu-z/Stable-Hair.git

WORKDIR /workspace/Stable-Hair

RUN pip install torch torchvision
RUN pip install diffusers transformers accelerate
RUN pip install opencv-python pillow numpy
RUN pip install runpod

COPY handler.py .

CMD ["python", "handler.py"]
