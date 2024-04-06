import torch
from torch.nn.utils.prune import global_unstructured
from model import CRNN  # Import your CRNN model class
from config import prune_config  # Import the pruning configuration

def prune_crnn_model(model, pruning_parameters, pruning_rate):
    for parameter, param_name in pruning_parameters:
        global_unstructured([parameter], pruning_method=prune_config['pruning_method'], amount=pruning_rate, name=param_name)
    return model

def main():
    # Load your pre-trained model to prune
    model = CRNN(1, prune_config['img_height'], prune_config['img_width'], 79,  # Set num_class to 79
                map_to_seq_hidden=prune_config['map_to_seq_hidden'],
                rnn_hidden=prune_config['rnn_hidden'],
                leaky_relu=prune_config['leaky_relu'])
    model.load_state_dict(torch.load('/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/onemillion_again/best.pt'))

    # Define a pruning strategy
    pruning_rate = prune_config['pruning_rate']

    # Define your pruning parameters here after model is defined
    pruning_parameters = [
        (model.cnn.conv0.weight, 'weight'),  # Prune the weights of the first convolutional layer
        (model.dense.weight, 'weight'),     # Prune the weights of the dense layer
        # Add more layers as needed
    ]

    # Prune the model
    pruned_model = prune_crnn_model(model, pruning_parameters, pruning_rate)

    # Save the pruned model
    torch.save(pruned_model.state_dict(), 'pruned_model.pth')

if __name__ == '__main__':
    main()

