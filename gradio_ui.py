import gradio as gr
import numpy as np
from neural_network import NeuralNetwork
from PIL import Image

def load_model():
    input_size = 28 * 28  # Flattened image size
    hidden_size = 128
    output_size = 10  # Number of classes in MNIST

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.weights_input_hidden = np.load('weights_input_hidden.npy')
    nn.weights_hidden_output = np.load('weights_hidden_output.npy')
    nn.bias_hidden = np.load('bias_hidden.npy')
    nn.bias_output = np.load('bias_output.npy')
    
    return nn

# Load the trained model
nn = load_model()

def predict(image):
    image = np.array(image)  # Convert to numpy array
    
    # Check if the image is already in the correct shape
    if image.shape != (28, 28):
        # Resize the image to 28x28 if it's not already
        image = Image.fromarray(image).resize((28, 28))
        image = np.array(image)
    
    image = image.flatten().reshape(1, -1)  # Flatten the image
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
