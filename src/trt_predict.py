import tensorrt as trt
import numpy as np
import pycuda.driver as cuda

# Initialize CUDA context
cuda.init()

# Load the saved TensorRT engine
trt_file = 'rgb.trt'
with open(trt_file, 'rb') as f:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())

# Create an execution context
context = engine.create_execution_context()

# Prepare random dummy input data
input_shape = (1, 1, 32, 100)
input_data = np.random.randn(*input_shape).astype(np.float32)

# Allocate device memory for inputs and outputs
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(engine.max_batch_size * config['num_classes'] * 4)

# Transfer input data to device
cuda.memcpy_htod_async(d_input, input_data, stream)

# Execute the model
context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])

# Transfer output data from device
output_data = np.empty((1, config['num_classes']), dtype=np.float32)
cuda.memcpy_dtoh_async(output_data, d_output, stream)
stream.synchronize()

# Process the output data as needed
print("Output shape:", output_data.shape)

