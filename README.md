 # 🪶 Bird Feather Color Classification System
A Deep Learning-based image classification project to detect bird feather colors using a custom Convolutional Neural Network (CNN). Built with TensorFlow, Keras, and Gradio, this system classifies feathers into 12 color categories with high accuracy.
---
### Project Overview

The goal of this project is to identify the color of bird feathers from images using deep learning. The system is trained on a dataset of ~2400 feather images across the following 12 classes:
- Black
- Blue
- Brown
- Gray
- Green
- Orange
- Purple
- Red
- White
- Yellow
- Iridescent
- Multicolor

Users can upload an image via a simple Gradio interface and receive a prediction in real-time.
---
### Features

- Real-time prediction of feather colors
- Custom CNN architecture trained from scratch
- User-friendly Gradio interface
- Image preprocessing: resizing, normalization, augmentation
- Model accuracy: **99.5%**
- Confusion matrix and classification report for performance analysis

---
### Tech Stack
- **Programming Language**: Python
- **Frameworks/Libraries**: TensorFlow, Keras, Albumentations, Gradio
- **Dataset Management**: NumPy, OpenCV
- **Training & Evaluation Tools**: Confusion Matrix, Classification Report
---
 ### Model Architecture
- Input size: 128 × 128 × 3 (RGB images)
- 4 Convolutional blocks with ReLU, BatchNorm, and MaxPooling
- Flatten layer to convert feature maps
- Dense layers: 512 → Dropout → 256 → Dropout
- Output layer: Softmax (12 classes)
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

---
### User Interface Preview
![Feather Prediction UI](https://i.imgur.com/lkVg6mO.png)
