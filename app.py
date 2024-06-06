

import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import tempfile
import os
import base64
from huggingface_hub import snapshot_download

# Load the SSD-1B model
model_path_ssd1b = snapshot_download(repo_id="segmind/SSD-1B")
pipe_ssd1b = StableDiffusionXLPipeline.from_pretrained(
    model_path_ssd1b,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe_ssd1b.to("cuda")

def generate_image(prompt, neg_prompt, file_format):
    try:
        # Generate the image
        image = pipe_ssd1b(
            prompt=prompt,
            negative_prompt=neg_prompt
        ).images[0]

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as tmp_file:
            image_path = tmp_file.name
            image.save(image_path, format=file_format.upper())

        # Read the image as bytes
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        # Encode the bytes in base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Delete the temporary file
        os.remove(image_path)

        # Create a download link
        download_link = f'<a href="data:image/{file_format};base64,{base64_image}" download="generated_image.{file_format}" class="download-button">Download Image</a>'

        return image, download_link
    except Exception as e:
        return None, str(e)

prompt = gr.Text(
    label="Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your prompt",
    container=False,
)
neg_prompt = gr.Text(
    label="Negative Prompt",
    show_label=False,
    max_lines=1,
    placeholder="Enter your negative prompt",
    container=False,
)
file_format = gr.Radio(
    choices=["png", "jpg", "jpeg", "bmp", "tiff", "gif"],
    value="png",
    label="Choose file format"
)

iface = gr.Interface(
    fn=generate_image,
    inputs=[prompt, neg_prompt, file_format],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.HTML(label="Download Link")
    ],
    title="SSD-1B Model Image Generator",
    allow_flagging=False
)

iface.launch(share=True)
