import tensorflow as tf
import numpy as np
from io import BytesIO
import os


def load_model():
    model_path = r'D:\BirdFeatherClassification\webapp\models\BirdFeatherClassificationTrainedModel1.h5'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    return model




def predict_feather_color(file, model):
    # Read the file content in memory
    img = tf.keras.preprocessing.image.load_img(BytesIO(file.read()), target_size=(128, 128))  # Adjust size as needed
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    # model's predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    class_names = ["black feathers", "blue feathers", "brown feathers", "gray feathers", "green feathers", "iridescent feathers", "multicolor feathers", "orange feathers", "purple feathers", "red feathers", "white feathers", "yellow feathers"]

    # Return predicted feather color
    return class_names[predicted_class]

