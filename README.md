AI/ML Beginners Project
# Handwritten Digit Recognition

## Project Overview
This project implements a **Handwritten Digit Recognition** system using **Machine Learning** techniques. The main objective of this project is to classify handwritten digits (0–9) from images, similar to the well-known **MNIST** dataset. It demonstrates how neural networks can learn to recognize numerical digits from images and accurately predict their class.

The project is designed to provide an end-to-end workflow including:
- Model training
- Evaluation
- Real-time prediction using a Gradio-based web interface

## Technologies Used
- **Python**
- **TensorFlow / Keras** for building and training the neural network model
- **NumPy** and **OpenCV** for image preprocessing
- **Matplotlib** for visualization
- **Gradio** for creating an interactive web interface

## Dataset
We did not download or use an external dataset manually.  
Instead, we used the **MNIST dataset** that comes preloaded with the `tensorflow.keras.datasets` module.

The dataset is automatically fetched from Keras during runtime, containing:
60,000 training images
10,000 testing images
Each image is a 28x28 grayscale handwritten digit.
This eliminates the need for manual dataset downloads and makes the project portable and easy to execute on any environment.

## Model Description

The model is a simple Feedforward Neural Network (FNN) trained on the MNIST dataset.

Key points:

Input layer of 784 neurons (flattened 28x28 image)

Two hidden layers using ReLU activation

Output layer of 10 neurons with softmax activation

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy metric used for evaluation

## Image Preprocessing

Before feeding the image into the model, preprocessing steps are applied to ensure consistency with MNIST:

Convert to grayscale using OpenCV

Resize to 28x28 pixels

Invert colors to match the MNIST format (white digit on black background)

Normalize pixel values to range [0, 1]

Flatten the image into a 1D vector of 784 values

These preprocessing steps ensure that the model performs accurately on both uploaded and drawn images.

## Gradio App Integration

We used Gradio to create an interactive user interface for real-time predictions.

Why Gradio?

Gradio allows anyone to:

Interact with a trained machine learning model through a simple web interface

Draw or upload images directly in the browser

Get real-time predictions without writing any extra code

It helps visualize the model performance intuitively and is easy to integrate with notebooks or web apps.

How Gradio is used in this project

After training the model, a Gradio interface was created using:
```
import gradio as gr

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="numpy", image_mode="RGB"),
    outputs=[gr.Label(num_top_classes=10), gr.Textbox(label="Prediction")],
    title="Handwritten Digit Recognition",
    description="Upload or draw a digit (0–9) and see the model’s prediction."
)

interface.launch(share=True)
```
The app allows the user to:

Draw digits directly on the canvas

Upload any digit image

View the model’s predicted digit and confidence scores instantly

The share=True parameter generates a temporary public URL, making the app accessible even if localhost is restricted (e.g., inside Cursor or VS Code).

## How to Run the Project

Clone or download the project files.

Open the main.ipynb file in Jupyter Notebook, JupyterLab, or Cursor.

Run all cells sequentially.

The Gradio app will automatically open with a public link (if share=True is enabled).

Draw or upload a digit image and view the prediction in real-time.

## Results

The model achieves high accuracy on test data and performs well on drawn digits.
The Gradio interface provides an easy and interactive way to visualize how the model interprets handwritten digits.

## Future Improvements

Improve preprocessing for more accurate recognition on varied handwriting styles.

Integrate convolutional neural networks (CNNs) for higher accuracy.

Deploy the Gradio app permanently using platforms like Hugging Face Spaces or Render.

## Conclusion

This project successfully demonstrates how to build and deploy a handwritten digit recognition model with an interactive front-end using Gradio.
It shows the complete lifecycle of a machine learning project — from data preparation to model deployment — without requiring any manual dataset download.

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)


