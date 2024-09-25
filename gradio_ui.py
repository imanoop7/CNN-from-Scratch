import gradio as gr
import numpy as np
from neural_network import NeuralNetwork
from PIL import Image

def load_model():
    nn = NeuralNetwork()
    nn.conv.filters = np.load('conv_filters.npy')
    nn.conv.biases = np.load('conv_biases.npy')
    nn.fc.weights = np.load('fc_weights.npy')
    nn.fc.bias = np.load('fc_bias.npy')
    return nn

# Load the trained model
nn = load_model()

# Load the trained model
nn = load_model()

def predict(image):
    image = np.array(image)  # Convert to numpy array
    
    # Resize the image to 28x28 if it's not already
    if image.shape != (28, 28):
        image = Image.fromarray(image).resize((28, 28))
        image = np.array(image)
    
    # Reshape for CNN input (add batch and channel dimensions)
    image = image.reshape(1, 1, 28, 28) / 255.0  # Normalize to [0, 1]
    prediction = nn.forward(image)
    return int(np.argmax(prediction))  # Ensure the output is an integer


# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(image_mode='L'),
    outputs=gr.Label(num_top_classes=3)
)

if __name__ == "__main__":
    interface.launch()
