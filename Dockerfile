FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

RUN pip install runpod opencv-python pillow torch torchvision diffusers transformers accelerate

COPY handler.py .

CMD ["python", "handler.py"]
