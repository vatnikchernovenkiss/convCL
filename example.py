import math
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import torch
from copy import deepcopy
from torch import nn
import sys
import convCL
import convCL_oneimg

platform_num = int(sys.argv[1])
using_gpu_num = int(sys.argv[2])
H = int(sys.argv[3])
W = int(sys.argv[4])
C = int(sys.argv[5])
N_filters = int(sys.argv[6])
N_img = int(sys.argv[7])
krnl_H = int(sys.argv[8])
krnl_W = int(sys.argv[9])
method_name = sys.argv[10]

print(
    f'''
    test: test_multigpu_multiimg_gemm_implicit, using_gpu_num: {using_gpu_num},
    H: {H}, W: {W}, C: {C}, N_filters: {N_filters}, N_img: {N_img},
    krnl_H: {krnl_H}, krnl_W: {krnl_W}, method: {method_name}
    '''
)

print()

if N_img == 1:
    if method_name == 'GEMM':
        method = convCL_oneimg.GEMM
    elif method_name == 'GEMMImplicit':
        method = convCL_oneimg.GEMMImplicit
    elif method_name == 'Winograd':
        method = convCL_oneimg.Winograd
    else:
        raise ValueError('Unknown method name')
else:
    if method_name == 'GEMM':
        method = convCL.GEMM
    elif method_name == 'GEMMImplicit':
        method = convCL.GEMMImplicit
    elif method_name == 'Winograd':
        method = convCL.Winograd
    else:
        raise ValueError('Unknown method name')

platform = cl.get_platforms()[platform_num]

devices = platform.get_devices()

using_devices = devices[:using_gpu_num]

ctx = cl.Context(using_devices)

queue_arr = []

for device in using_devices:
    queue_arr.append(cl.CommandQueue(ctx, device=device))

mf = cl.mem_flags
data = np.random.randn(N_img, C, H, W).astype(np.float32)
dilation_H = 1
stride_H = 1
dilation_W = 1
stride_W = 1
pad_H = (krnl_H // 2) * dilation_H
pad_W = (krnl_W // 2) * dilation_W

dst_H = int(math.floor((H + 2 * pad_H - (dilation_H * (krnl_H - 1) + 1)) / stride_H + 1))
dst_W = int(math.floor((W + 2 * pad_W - (dilation_W * (krnl_W - 1) + 1)) / stride_W + 1))

kernels = np.random.randn(N_filters, C, krnl_H, krnl_W).astype(np.float32)
data_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)

myconv = method(ctx, using_gpu_num, queue_arr, N_img, H, W, C, N_filters,
                dst_H, dst_W, krnl_H, krnl_W, pad_H, pad_W, kernels)

result_buf = myconv.forward(data_gpu)

device = torch.device("cuda")

torch_conv = nn.Conv2d(C, N_filters, kernel_size=(krnl_H, krnl_W), padding=(pad_H, pad_W),
                       stride=(stride_H, stride_W), dilation=(dilation_H, dilation_W), device=device)
torch_conv.weight.data = torch.FloatTensor(kernels).to(device)
torch_conv.bias.data = torch.zeros(N_filters).to(device)
torch_conv = torch_conv

data_torch = torch.FloatTensor(data).to(device)

torch_res = torch_conv(data_torch)

for i in range(using_gpu_num):
    result_arr = cl.array.Array(
        queue_arr[i], N_img * N_filters * dst_H * dst_W, np.float32, data=result_buf
    )
    res_sum = result_arr.get().reshape(N_img, N_filters, dst_H, dst_W).sum()
    torch_sum = torch_res.cpu().detach().numpy().sum()
    print('Relative difference between torch and convCL for device ', i, ': ',
          round(np.abs((res_sum - torch_sum) / torch_sum) * 100, 3), '%')

myconv.release_data()

print('-' * 100)
print()
