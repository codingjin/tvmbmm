
"""
import torch

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print("Name:", props.name)
#print("Max threads per block:", props.max_threads_per_block)
print("Shared mem per block (bytes):", props.shared_memory_per_block)
print(props)
"""


import pycuda.driver as cuda
cuda.init()
dev = cuda.Device(0)
print("Name:", dev.name())

print("Max threads per block:", dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK))
print("Shared mem per block (bytes):", dev.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK))
#print("Shared mem per block (bytes) OPTIN:", dev.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN))
print("Compute capability:", "%d.%d" % dev.compute_capability())

