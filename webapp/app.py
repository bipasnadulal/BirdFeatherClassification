import gradio as gr
import tensorflow as tf
import numpy as np
from io import BytesIO
from model_loader import load_model, predict_feather_color

# Load model once
model = load_model()

def predict_image(image):
    # Convert image to BytesIO
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Predict
    predicted_color = predict_feather_color(img_byte_arr, model)

    return image, f"Predicted Feather Color: {predicted_color}"

# Gradio UI
demo = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil"), 
    outputs=[gr.Image(), gr.Text()],
    title="Bird Feather Color Classification",
    description="Upload an image of a bird's feather to predict its color."
)

if __name__ == "__main__":
    demo.launch(share=True)
