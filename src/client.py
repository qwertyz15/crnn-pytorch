import grpc
import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import os
import glob
from dataset import Synth90kDataset
from ctc_onnx_decoder import ctc_decode

def preprocess_image(image_path, img_height, img_width):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image
    image = cv2.resize(image, (img_width, img_height))
    
    # Convert to float32
    image = image.astype(np.float32)
    
    # Normalize the image
    image = (image / 127.5) - 1.0
    
    # Reshape to match the model's input shape
    image = image.reshape(1, 1, img_height, img_width)
    
    return image

def main():
    # Triton server details
    server_url = 'localhost:8001'  # Replace with your server URL and port

    # Connect to the server
    triton_client = grpcclient.InferenceServerClient(url=server_url)

    # Define the path to the directory containing images
    image_directory = '/home/dev/Documents/new_crnn/crnn-pytorch/one_million_lpr/val'

    # List all image files in the directory with multiple extensions using glob.glob
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_directory, '**', ext), recursive=True))

    # Process each image in the directory
    img_height = 32
    img_width = 100

    for image_path in image_paths:
        # Load and preprocess the image
        image = preprocess_image(image_path, img_height, img_width)

        # Prepare the request
        inputs = [grpcclient.InferInput('input', [1, 1, 32, 100], 'FP32')]
        inputs[0].set_data_from_numpy(image)

        # Send the request
        outputs = [grpcclient.InferRequestedOutput('output')]
        response = triton_client.infer('crnn-pytorch-trt', inputs, request_id='1', outputs=outputs)

        # Get and process the response
        output_data = response.as_numpy('output')
        preds = ctc_decode(output_data, method='beam_search', beam_size=10, label2char=Synth90kDataset.LABEL2CHAR)
        print(f'Prediction for {image_path}: {"".join(preds[0])}')

if __name__ == '__main__':
    main()

