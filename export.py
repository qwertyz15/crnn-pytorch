import torch
import torch.onnx
import tensorrt as trt
from model import CRNN
from dataset import Synth90kDataset
from config import common_config as config

def load_pytorch_model(checkpoint_path):
    img_height = config['img_height']
    img_width = config['img_width']
    num_class = len(Synth90kDataset.LABEL2CHAR) + 1

    model = CRNN(1, img_height, img_width, num_class,
                 map_to_seq_hidden=config['map_to_seq_hidden'],
                 rnn_hidden=config['rnn_hidden'],
                 leaky_relu=config['leaky_relu'])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model
'''
def export_to_onnx(model, output_onnx_file, img_height, img_width):
    dummy_input = torch.randn(1, 1, img_height, img_width)
    torch.onnx.export(model, dummy_input, output_onnx_file, opset_version=15)
'''
def export_to_onnx(model, output_onnx_file, img_height, img_width):
    # Create a dummy input tensor matching the input format
    dummy_input = torch.randn(1, 1, img_height, img_width)  # 1 batch size, 1 channel, height, width

    # Export the model
    torch.onnx.export(model, 
                      dummy_input, 
                      output_onnx_file,
                      export_params=True, 
                      opset_version=15,  # or a different version depending on your ONNX and PyTorch versions
                      do_constant_folding=True,  
                      input_names=['input'],   
                      output_names=['output'])
''',
dynamic_axes={'input': {0: 'batch_size'},    
'output': {0: 'batch_size'}})
'''
def convert_onnx_to_tensorrt(onnx_file):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('Failed to parse ONNX file')

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, config)
    return engine

def save_tensorrt_model(engine, output_trt_file):
    with open(output_trt_file, "wb") as f:
        f.write(engine.serialize())

def main():
    checkpoint_path = '/home/dev/Documents/new_crnn/crnn-pytorch/checkpoints/another_million_rgb/best.pt'  # Adjust as necessary
    onnx_file = 'rgb.onnx'
    trt_file = 'rgb.trt'

    # Load PyTorch Model
    model = load_pytorch_model(checkpoint_path)

    # Export to ONNX
    export_to_onnx(model, onnx_file, config['img_height'], config['img_width'])

    # Convert to TensorRT
    #engine = convert_onnx_to_tensorrt(onnx_file)

    # Save TensorRT engine to file
    #save_tensorrt_model(engine, trt_file)

    #print(f'TensorRT model saved to {trt_file}')

if __name__ == '__main__':
    main()
