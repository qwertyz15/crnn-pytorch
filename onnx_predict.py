import cv2
import numpy as np
import torch
import onnxruntime as ort
from dataset import Synth90kDataset
from ctc_onnx_decoder import ctc_decode
import glob
import os

def load_model(model_path):
    # Check for CUDA availability and set the appropriate providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    print(providers)
    return ort.InferenceSession(model_path, providers=providers)

def preprocess_image(image_path, img_height, img_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0
    # image /= 255.0
    image = image.reshape(1, 1, img_height, img_width)  # Add batch and channel dimensions
    return image


def predict_onnx(onnx_model, image, label2char, decode_method, beam_size):
    # Convert numpy array to torch tensor and move to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_tensor = torch.from_numpy(image).to(device)

    # Inference with ONNX
    logits = onnx_model.run(None, {'input': image_tensor.cpu().numpy()})[0]
    log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=2).cpu().numpy()
    # log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    # Decode
    preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size, label2char=label2char)
    return preds

def main():
    model_path = 'rgb.onnx'  # Replace with your ONNX model path
    #image_paths = ['/home/dev/Documents/new_crnn/crnn-pytorch/24가8694.jpg', '/home/dev/Documents/new_crnn/crnn-pytorch/전남99라9467.jpg']  # Replace with your image paths
    # Define the path to the directory containing images
    image_directory = '/home/dev/Documents/new_crnn/crnn-pytorch/one_million_lpr/val'

    # List all image files in the directory with multiple extensions using glob.glob
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_directory, '**', ext), recursive=True))
    # image_paths.sort()    
    
    decode_method = 'beam_search'
    beam_size = 10

    img_height = 32  # Replace with your model's input height
    img_width = 100  # Replace with your model's input width

    onnx_model = load_model(model_path)

    for image_path in image_paths:
        image = preprocess_image(image_path, img_height, img_width)
        pred = predict_onnx(onnx_model, image, Synth90kDataset.LABEL2CHAR, decode_method, beam_size)
        print(f'Prediction for {image_path}:', ''.join(pred[0]))

if __name__ == '__main__':
    main()
