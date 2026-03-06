import runpod
import subprocess
import base64
import tempfile
import os

def handler(event):
    input_data = event["input"]

    user_image = input_data["user_image"]
    ref_image = input_data["ref_image"]

    user_bytes = base64.b64decode(user_image)
    ref_bytes = base64.b64decode(ref_image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
        f1.write(user_bytes)
        user_path = f1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
        f2.write(ref_bytes)
        ref_path = f2.name

    output_path = "/workspace/output.png"

    subprocess.run([
        "python",
        "infer_full.py",
        "--source", user_path,
        "--reference", ref_path,
        "--output", output_path
    ])

    with open(output_path, "rb") as f:
        result = base64.b64encode(f.read()).decode()

    return {"image": result}

runpod.serverless.start({"handler": handler})
