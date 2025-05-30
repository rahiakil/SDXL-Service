import base64
import io
from PIL import Image
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from huggingface_hub import hf_hub_download

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize CUDA device if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# SSL Configuration
DOMAIN = 'non-fungible-t-shirts.com'
ssl_context = None
cert_path = os.path.join(os.path.dirname(__file__), 'ssl', 'cert.pem')
key_path = os.path.join(os.path.dirname(__file__), 'ssl', 'key.pem')

if os.path.exists(cert_path) and os.path.exists(key_path):
    ssl_context = (cert_path, key_path)
else:
    print("Warning: SSL certificates not found. Running in HTTP mode.")

# Initialize the models
def initialize_pipeline():
    # Load the ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        use_auth_token=os.getenv("HUGGING_FACE_TOKEN")
    )

    # Load the SDXL pipeline with ControlNet
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_auth_token=os.getenv("HUGGING_FACE_TOKEN")
    )
    
    # Enable memory efficient attention
    pipe.enable_model_cpu_offload()
    return pipe

# Initialize the pipeline
pipe = initialize_pipeline()

def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Error decoding base64 image: {str(e)}")

def encode_pil_to_base64(image):
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data or 'layout' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Get parameters from request
        prompt = data['prompt']
        layout_image = decode_base64_image(data['layout'])
        negative_prompt = data.get('negative_prompt', '')
        num_inference_steps = data.get('num_inference_steps', 30)
        guidance_scale = data.get('guidance_scale', 7.5)

        # Prepare layout image
        layout_image = layout_image.convert('RGB')
        layout_image = layout_image.resize((1024, 1024))

        # Generate image
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=layout_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Convert output image to base64
        output_base64 = encode_pil_to_base64(output)

        return jsonify({
            'status': 'success',
            'image': output_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5111,
        ssl_context=ssl_context
    ) 