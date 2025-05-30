# SDXL-Service with ControlNet

A Flask-based backend service that generates shirt designs using Stable Diffusion XL (SDXL) with ControlNet for layout-aware image generation.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your HuggingFace token:
```
HUGGING_FACE_TOKEN=your_token_here
```

4. Run the service:
```bash
python app.py
```

## API Endpoints

### POST /generate
Generates a shirt design based on the provided prompt and layout.

Request body:
```json
{
    "prompt": "a beautiful floral pattern",
    "negative_prompt": "blurry, bad quality",
    "layout": "base64_encoded_layout_image",
    "num_inference_steps": 30,
    "guidance_scale": 7.5
}
```

Response:
```json
{
    "image": "base64_encoded_generated_image",
    "status": "success"
}
```

## Notes
- The service uses SDXL 1.0 as the base model
- ControlNet is used for layout conditioning
- CORS is enabled for cross-origin requests
- Default port is 5000
