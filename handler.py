import runpod
import base64
import tempfile
import os
import subprocess

def save_base64_image(base64_string, path):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)

    with open(path, "wb") as f:
        f.write(image_bytes)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(event):

    input_data = event["input"]

    # test endpoint
    if "ping" in input_data:
        return {"status": "server working"}

    user_image = input_data.get("image_user")
    ref_image = input_data.get("image_ref")

    if user_image is None or ref_image is None:
        return {"error": "image_user and image_ref required"}

    with tempfile.TemporaryDirectory() as tmp:

        user_path = os.path.join(tmp, "user.png")
        ref_path = os.path.join(tmp, "hair.png")
        output_path = os.path.join(tmp, "result.png")

        save_base64_image(user_image, user_path)
        save_base64_image(ref_image, ref_path)

        try:

            command = [
                "python",
                "scripts/inference.py",
                "--input_face", user_path,
                "--input_hair", ref_path,
                "--output", output_path
            ]

            subprocess.run(command, check=True)

        except Exception as e:
            return {"error": str(e)}

        if not os.path.exists(output_path):
            return {"error": "model did not generate output"}

        result_base64 = encode_image(output_path)

        return {
            "image": result_base64
        }


runpod.serverless.start({"handler": handler})
