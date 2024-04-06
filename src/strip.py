import os
import torch
from model import CRNN  # Import the CRNN class from model.py

from config import common_config

# Extract common configuration parameters
data_dir = common_config['data_dir']
img_width = common_config['img_width']
img_height = common_config['img_height']
map_to_seq_hidden = common_config['map_to_seq_hidden']
rnn_hidden = common_config['rnn_hidden']
leaky_relu = common_config['leaky_relu']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path to the model checkpoint and the output file
input_checkpoint = '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/onemillion_again/best.pt'
output_checkpoint = '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/onemillion_again/stripped_best.pt'

# Define the strip_optimizer function in the same script
def strip_optimizer(f='best.pt', s=''):
    # Load the model checkpoint
    checkpoint = torch.load(f, map_location=torch.device('cpu'))
    print(checkpoint.keys())
    
    # Check if there's an Exponential Moving Average (EMA) model
    if 'ema' in checkpoint:
        checkpoint['model'] = checkpoint['ema']  # Replace model with EMA

    # Remove optimizer-related information
    for key in ['optimizer', 'best_fitness', 'wandb_id', 'ema', 'updates']:
        if key in checkpoint:
            checkpoint[key] = None
    
    # Set the epoch to -1 to indicate it's not related to a specific training epoch
    checkpoint['epoch'] = -1

    
    # Convert the model to half-precision (FP16)
    model = checkpoint['model']
    model.half()

    # Set requires_grad to False for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Save the stripped checkpoint
    torch.save(checkpoint, s or f)
    
    # Calculate the file size of the saved checkpoint
    file_size_mb = os.path.getsize(s or f) / 1E6

    # Print a message indicating the operation is complete
    print(f"Optimizer stripped from '{f}', {'saved as %s,' % s if s else ''} {file_size_mb:.1f}MB")

# Assuming you have 'num_class' defined somewhere
num_class = 79  # Replace with your actual number of classes

# Create the CRNN model
crnn = CRNN(1, img_height, img_width, num_class,
            map_to_seq_hidden=map_to_seq_hidden,
            rnn_hidden=rnn_hidden,
            leaky_relu=leaky_relu)

# Load the pre-trained weights (if applicable)
crnn.load_state_dict(torch.load(input_checkpoint, map_location=device))

# Move the model to the device (CPU or GPU)
crnn.to(device)

# Call the 'strip_optimizer' function to strip the optimizer-related information
strip_optimizer(f=input_checkpoint, s=output_checkpoint)

# Check the file size of the stripped model
mb = os.path.getsize(output_checkpoint) / 1E6  # File size in MB
print(f"Optimizer stripped from '{input_checkpoint}', saved as '{output_checkpoint}', {mb:.1f}MB")

