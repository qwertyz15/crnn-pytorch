import cv2
import numpy as np
import onnxruntime as ort

def load_model(model_path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(model_path, providers=providers)

def preprocess_image(image_path, img_height, img_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0  # Normalize
    image = image.reshape(1, 1, img_height, img_width)  # Add batch and channel dimensions
    return image

def run_inference(model, image):
    inputs = {model.get_inputs()[0].name: image}
    outputs = model.run(None, inputs)
    return outputs

# Paths to your ONNX models and the image
model_path_dynamic = '/home/dev/Documents/new_crnn/crnn-pytorch/src/real_.onnx'
model_path_static = '/home/dev/Documents/new_crnn/crnn-pytorch/src/real.onnx'
image_path = '/home/dev/Documents/new_crnn/crnn-pytorch/24가8694.jpg'

# Load models
model_dynamic = load_model(model_path_dynamic)
model_static = load_model(model_path_static)

# Image dimensions
img_height = 32  # replace with your model's input height
img_width = 100  # replace with your model's input width

# Preprocess image
image = preprocess_image(image_path, img_height, img_width)

# Inference
output_dynamic = run_inference(model_dynamic, image)
print(output_dynamic.shape)
output_static = run_inference(model_static, image)
print(output_static.shape)

# Print or process the outputs
print("Output with dynamic axes model:", output_dynamic)
print("Output with static axes model:", output_static)

