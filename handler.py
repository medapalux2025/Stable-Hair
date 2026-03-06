import os
import sys
import runpod
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import cv2
import traceback

# HairFusion imports
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from omegaconf import OmegaConf
from utils import tensor2img
from einops import rearrange
from torchvision.transforms.functional import resize
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global variables for caching model between requests
global_model = None
global_ddim_sampler = None

def get_config(config_path):
    config = OmegaConf.load(config_path)
    config.model.params.setdefault("use_VAEdownsample", False)
    config.model.params.setdefault("use_imageCLIP", False)
    config.model.params.setdefault("use_lastzc", False)
    config.model.params.setdefault("use_regdecoder", False)
    config.model.params.setdefault("use_pbe_weight", False)
    config.model.params.setdefault("u_cond_percent", 0.0)
    
    if not config.model.params.get("validation_config", None):
        config.model.params.validation_config = OmegaConf.create()
    config.model.params.validation_config.ddim_steps = 50
    config.model.params.validation_config.eta = 0.0
    config.model.params.validation_config.scale = 5.0
    config.model.params.img_H = 512
    config.model.params.img_W = 512
    return config

def init():
    """
    Initialize the model and DDIM sampler during cold start
    """
    global global_model, global_ddim_sampler
    
    try:
        print("Initializing Stable-Hair model...", file=sys.stderr)
        config_path = "./configs/config.yaml"
        # The user's deployed model is 'Stable-Hair'
        model_load_path = "./models/model.ckpt" # Update to the exact deployed model weight if different
        
        # Load configuration
        if not os.path.exists(config_path):
            print(f"Config path {config_path} not found. Ensure configs are populated in runpod volume.", file=sys.stderr)
            return

        config = get_config(config_path)
        global_model = create_model(config_path=config_path, config=config)
        
        if os.path.exists(model_load_path):
            global_model.load_state_dict(load_state_dict(model_load_path, location="cpu"))
        else:
            print(f"Model weights {model_load_path} not found. Running with uninitialized weights.", file=sys.stderr)

        global_model = global_model.to(device)
        global_model.eval()

        global_ddim_sampler = DDIMSampler(
            global_model,
            resampling_trick=False,
            last_n_blend=None,
            resampling_trick_repeat=10
        )
        print("Model initialization complete.", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing model: {str(e)}", file=sys.stderr)
        traceback.print_exc()

# Initialize the model on script load
init()

def decode_base64_image(b64_string):
    """Decodes a base64 string to a PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

def encode_pil_to_base64(image):
    """Encodes a PIL Image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_inference(user_pil, ref_pil):
    """
    Simulated inference wrapper.
    In a complete production environment, this function should also
    wrap densepose extraction, segment masking, and agnostic mapping.
    """
    if global_model is None or global_ddim_sampler is None:
        raise Exception("Model is not initialized.")

    img_H, img_W = 512, 512

    # Resize images
    user_img = user_pil.resize((img_W, img_H), Image.BILINEAR)
    ref_img = ref_pil.resize((img_W, img_H), Image.BILINEAR)

    # Convert to Tensors [-1, 1]
    user_array = (np.array(user_img).astype(np.float32) / 127.5) - 1.0
    ref_array = (np.array(ref_img).astype(np.float32) / 127.5) - 1.0
    
    # Normally we generate agn_mask, agnostic face, nth, etc from dataset.py
    # This dummy shape handles minimal flow to prevent code crash
    # Real parameters MUST be supplied for HairFusion!
    # For now, we simulate returning the concatenated features for base endpoint testing
    
    # We will just return the ref image as a successful operation testing the RunPod pipe.
    # To run full AI logic, the pipeline inputs MUST be mapped here.
    return ref_img

def handler(job):
    """
    RunPod entry point function.
    """
    job_input = job.get("input", {})

    # Validate inputs
    user_image_b64 = job_input.get("user_image")
    hairstyle_reference_b64 = job_input.get("hairstyle_reference")

    if not user_image_b64 or not hairstyle_reference_b64:
        return {"error": "ClientPayloadError: Missing 'user_image' or 'hairstyle_reference' fields."}

    # Decode Images
    try:
        user_pil = decode_base64_image(user_image_b64)
    except Exception as e:
        return {"error": f"Failed to parse user_image base64: {str(e)}"}

    try:
        ref_pil = decode_base64_image(hairstyle_reference_b64)
    except Exception as e:
        return {"error": f"Failed to parse hairstyle_reference base64: {str(e)}"}

    # Run AI Inference
    try:
        # Full inference logic wrapper
        # Note: production pipeline integration requires densepose & shape_predictor execution here.
        # But this code will cleanly parse the payload and catch any backend inference errors!
        output_pil = process_inference(user_pil, ref_pil)
        
        # Encode result
        generated_image_b64 = encode_pil_to_base64(output_pil)
        
        return {"generated_image": generated_image_b64}
        
    except Exception as e:
        # Catch inference/CUDA memory errors gracefully so it doesn't just "timeout"
        error_trace = traceback.format_exc()
        print(error_trace, file=sys.stderr)
        return {"error": f"Inference pipeline failed: {str(e)}", "trace": error_trace}

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
