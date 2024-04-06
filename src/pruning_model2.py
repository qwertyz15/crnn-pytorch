import torch
import torch.nn.utils.prune as prune
from config import evaluate_config as config
from dataset import Synth90kDataset
from model import CRNN

# Function to load the CRNN model based on your previous implementation
def load_model(model_path):
    img_height = config['img_height']
    img_width = config['img_width']
    map_to_seq_hidden = config['map_to_seq_hidden']
    rnn_hidden = config['rnn_hidden']
    leaky_relu = config['leaky_relu']
    num_class = len(Synth90kDataset.LABEL2CHAR) + 1

    model = CRNN(1, img_height, img_width, num_class,
                 map_to_seq_hidden=map_to_seq_hidden,
                 rnn_hidden=rnn_hidden,
                 leaky_relu=leaky_relu)
    
    model.load_state_dict(torch.load(model_path))
    return model

# Load the model
model_path = '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/onemillion_again/best.pt'
model = load_model(model_path)
print("Original Model Loaded")
print(model)

# Pruning 'conv1' layer
parameters_to_prune = [(model.cnn.conv1, 'weight')]

# Prune 30% of the connections in conv1 based on their weight
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

# Remove the pruning re-parametrization
for module, name in parameters_to_prune:
    prune.remove(module, name)

# Save the pruned model
torch.save(model.state_dict(), '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/onemillion_again/best_pruned.pt')
print("Pruned Model Saved")

