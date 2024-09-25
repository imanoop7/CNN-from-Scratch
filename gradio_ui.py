import gradio as gr
import numpy as np
from neural_network import NeuralNetwork
from PIL import Image

def load_model():
    print("Loading model weights and biases.")
    nn = NeuralNetwork()
    nn.conv.filters = np.load('conv_filters.npy')
    print("Loaded conv_filters.npy")
    nn.conv.biases = np.load('conv_biases.npy')
    print("Loaded conv_biases.npy")
    nn.fc.weights = np.load('fc_weights.npy')
    print("Loaded fc_weights.npy")
    nn.fc.bias = np.load('fc_bias.npy')
    print("Loaded fc_bias.npy")
    print("Model loaded successfully.")
    return nn

# Load the trained model
print("Initiating model loading.")
nn = load_model()

def predict(image):
    print("Received image for prediction.")
    image = np.array(image)  # Convert to numpy array
    print(f"Original image shape: {image.shape}")
    
    # Resize the image to 28x28 if it's not already
    if image.shape != (28, 28):
        print("Resizing image to 28x28.")
        image = Image.fromarray(image).resize((28, 28))
        image = np.array(image)
        print(f"Resized image shape: {image.shape}")
    
    # Reshape for CNN input (add batch and channel dimensions)
    image = image.reshape(1, 1, 28, 28) / 255.0  # Normalize to [0, 1]
    print(f"Image reshaped for CNN input: {image.shape}")
    
    prediction = nn.forward(image)
    predicted_class = int(np.argmax(prediction))
    print(f"Predicted class: {predicted_class}")
    return predicted_class  # Changed to return only the integer class

# Create Gradio interface
print("Setting up Gradio interface.")
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(image_mode='L'),
    outputs=gr.Label(num_top_classes=3)
)

if __name__ == "__main__":
    print("Launching Gradio interface.")
    interface.launch()
    print("Gradio interface launched.")