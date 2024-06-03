import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import warnings
from typing import Optional

mf = cl.mem_flags


class Winograd():

    def __init__(self, ctx: cl.Context, using_gpu_num: int, queue_arr: list[cl.CommandQueue],
                 N_img: int, H: int, W: int, C: int, N_filters: int, dst_H: int, dst_W: int,
                 krnl_H: int, krnl_W: int, pad_H: int, pad_W: int,
                 kernels: Optional[list[np.ndarray]] = None, dilation_H: int = 1, 
                 dilation_W: int = 1, stride_H: int = 1, stride_W: int = 1, 
                 use_big_tile: bool = False):

        self.using_gpu_num = using_gpu_num
        self.queue_arr = queue_arr
        self.N_filters = N_filters
        self.C = C
        self.H_ = cl.cltypes.int(H)
        self.C_ = cl.cltypes.int(C)
        self.W_ = cl.cltypes.int(W)
        self.dst_H_ = cl.cltypes.int(dst_H)
        self.dst_W_ = cl.cltypes.int(dst_W)
        self.pad_H_ = cl.cltypes.int(pad_H)
        self.pad_W_ = cl.cltypes.int(pad_W)
        self.krnl_H_ = cl.cltypes.int(krnl_H)
        self.krnl_W_ = cl.cltypes.int(krnl_W)
        self.dilation_H_ = cl.cltypes.int(dilation_H)
        self.dilation_W_ = cl.cltypes.int(dilation_W)
        self.stride_H_ = cl.cltypes.int(stride_H)
        self.stride_W_ = cl.cltypes.int(stride_W)
        self.N_filters_ = cl.cltypes.int(N_filters)
        self.res_size_ = cl.cltypes.int(dst_H * dst_W)

        self.bytes_per_img_src = H * W * C * 4
        self.N_img_per_gpu = N_img // using_gpu_num

        self.img_reminder = N_img % using_gpu_num

        if use_big_tile == True and not (krnl_H == krnl_W == 3):
            warnings.warn(
                "Big tile size is usable only in case of kernel size 3 X 3"
            )

        if krnl_H == krnl_W == 3:

            if use_big_tile == True:
                transformed_buf_H = 6
                transformed_buf_W = 6
                tile_H = 4
                tile_W = 4
                self.tiles_in_img = (H // tile_H) * (W // tile_W)

                if (H / 4) * (W / 4) >= 16:
                    winograd_file = 'conv_kernels/winograd/winograd_43.cl'
                    self.TILES_PER_GROUP = 16
                    self.CHANNELS_PER_GROUP = 9
                else:
                    winograd_file = 'conv_kernels/winograd/winograd_43_small.cl'
                    self.TILES_PER_GROUP = 4
                    self.CHANNELS_PER_GROUP = 9

                with open('conv_kernels/winograd/winograd_43_preproc.cl', 'r') as f:
                    winograd_transofrm_weights = f.read()

                with open(winograd_file, 'r') as f:
                    winograd = f.read()
            else:
                transformed_buf_H = 4
                transformed_buf_W = 4
                tile_H = 2
                tile_W = 2
                self.tiles_in_img = (H // tile_H) * (W // tile_W)

                if (H / 2) * (W / 2) >= 32:
                    winograd_file = 'conv_kernels/winograd/winograd_23.cl'
                    self.TILES_PER_GROUP = 32
                    self.CHANNELS_PER_GROUP = 8
                else:
                    winograd_file = 'conv_kernels/winograd/winograd_23_small.cl'
                    self.TILES_PER_GROUP = 4
                    self.CHANNELS_PER_GROUP = 8

                with open('conv_kernels/winograd/winograd_23_preproc.cl', 'r') as f:
                    winograd_transofrm_weights = f.read()

                with open(winograd_file, 'r') as f:
                    winograd = f.read()

        elif krnl_H == krnl_W == 5:
            transformed_buf_H = 6
            transformed_buf_W = 6
            tile_H = 2
            tile_W = 2
            self.tiles_in_img = (H // tile_H) * (W // tile_W)

            if (H / 2) * (W / 2) >= 16:
                winograd_file = 'conv_kernels/winograd/winograd_25.cl'
                self.TILES_PER_GROUP = 16
                self.CHANNELS_PER_GROUP = 9
            else:
                winograd_file = 'conv_kernels/winograd/winograd_25_small.cl'
                self.TILES_PER_GROUP = 4
                self.CHANNELS_PER_GROUP = 9

            with open('conv_kernels/winograd/winograd_25_preproc.cl', 'r') as f:
                winograd_transofrm_weights = f.read()

            with open(winograd_file, 'r') as f:
                winograd = f.read()
        else:
            raise ValueError(
                "Winograd method currenrly supports only 3 X 3 and 5 X 5 kernel sizes"
            )

        prg_trans = cl.Program(ctx, winograd_transofrm_weights).build()
        prg_winograd = cl.Program(ctx, winograd).build()

        self.knl_trans = prg_trans.winograd_transorm_weights
        self.knl_winograd = prg_winograd.winograd

        if kernels is None:
            kernels = np.random.randn(N_filters, C, krnl_H, krnl_W).astype(np.float32)

        self.kernel_arr = []

        for i in range(using_gpu_num):

            cur_kernels = cl_array.to_device(queue_arr[i], kernels)

            self.kernel_arr.append(cur_kernels)

        self.transposed_kernels_arr = []

        self.dest_buf = cl.Buffer(ctx, mf.READ_WRITE,
                                  size=N_img * N_filters * dst_H * dst_W * 4)

        self.dest_buf_arr = []

        bytes_per_img_dest = N_filters * dst_H * dst_W * 4

        for i in range(using_gpu_num):
            if i < self.img_reminder:
                dest_buf_сur = self.dest_buf.get_sub_region(
                    ((self.N_img_per_gpu + 1) * i) * bytes_per_img_dest,
                    ((self.N_img_per_gpu + 1) * (i + 1) * bytes_per_img_dest)
                )
            else:
                dest_buf_сur = self.dest_buf.get_sub_region(
                    (self.N_img_per_gpu * i + self.img_reminder) * bytes_per_img_dest,
                    (((self.N_img_per_gpu) * (i + 1) + self.img_reminder) * bytes_per_img_dest)
                )
            self.dest_buf_arr.append(dest_buf_сur)

            transposed_kernels_buf_cur = cl_array.empty(
                self.queue_arr[i],
                N_filters * C * transformed_buf_H * transformed_buf_W,
                np.float32
            )
            self.transposed_kernels_arr.append(transposed_kernels_buf_cur)

    def forward(self, data: cl.Buffer):

        gpu_data_arr = []

        for i in range(self.using_gpu_num):
            if i < self.img_reminder:
                gpu_data = data.get_sub_region(
                    ((self.N_img_per_gpu + 1) * i) * self.bytes_per_img_src,
                    ((self.N_img_per_gpu + 1) * (i + 1)) * self.bytes_per_img_src
                )
            else:
                gpu_data = data.get_sub_region(
                    (self.N_img_per_gpu * i + self.img_reminder) * self.bytes_per_img_src,
                    (self.N_img_per_gpu * (i + 1) + self.img_reminder) * self.bytes_per_img_src
                )
            gpu_data_arr.append(gpu_data)

        event_arr = []

        for i in range(self.using_gpu_num):

            event = self.knl_trans(
                self.queue_arr[i],
                (self.N_filters * self.C, ), (1, ),
                self.kernel_arr[i].data,
                self.transposed_kernels_arr[i].data
            )
            event_arr.append(event)

        for event in event_arr:
            event.wait()

        event_arr = []

        for i in range(self.using_gpu_num):
            N_img_ = cl.cltypes.int(self.N_img_per_gpu + (i < self.img_reminder))

            event = self.knl_winograd(
                self.queue_arr[i],
                (self.tiles_in_img * (self.N_img_per_gpu +
                 (i < self.img_reminder)) * self.CHANNELS_PER_GROUP,
                 self.N_filters // self.TILES_PER_GROUP +
                 ((self.N_filters % self.TILES_PER_GROUP) > 0)),
                (self.TILES_PER_GROUP * self.CHANNELS_PER_GROUP, 1),
                gpu_data_arr[i], self.transposed_kernels_arr[i].data, self.H_,
                self.W_, self.pad_H_, self.pad_W_, N_img_, self.C_,
                self.N_filters_, self.dest_buf_arr[i]
            )
            
            event_arr.append(event)

        for event in event_arr:
            event.wait()
        return self.dest_buf

    def release_data(self):
        for i in range(self.using_gpu_num):
            self.dest_buf_arr[i].release()
            self.transposed_kernels_arr[i].data.release()
            self.kernel_arr[i].data.release()


class GEMM():

    def __init__(self, ctx: cl.Context, using_gpu_num: int, queue_arr: list[cl.CommandQueue],
                 N_img: int, H: int, W: int, C: int, N_filters: int, dst_H: int, dst_W: int,
                 krnl_H: int, krnl_W: int, pad_H: int, pad_W: int,
                 kernels: Optional[list[np.ndarray]] = None, dilation_H: int = 1, 
                 dilation_W: int = 1, stride_H: int = 1, stride_W: int = 1):

        self.using_gpu_num = using_gpu_num
        self.queue_arr = queue_arr
        self.dst_H = dst_H
        self.dst_W = dst_W
        self.N_filters = N_filters
        self.C = C
        self.dst_H = dst_H
        self.dst_W = dst_W
        self.H_ = cl.cltypes.int(H)
        self.C_ = cl.cltypes.int(C)
        self.W_ = cl.cltypes.int(W)
        self.dst_H_ = cl.cltypes.int(dst_H)
        self.dst_W_ = cl.cltypes.int(dst_W)
        self.pad_H_ = cl.cltypes.int(pad_H)
        self.pad_W_ = cl.cltypes.int(pad_W)
        self.N_img_ = cl.cltypes.int(N_img)
        self.krnl_H_ = cl.cltypes.int(krnl_H)
        self.krnl_W_ = cl.cltypes.int(krnl_W)
        self.dilation_H_ = cl.cltypes.int(dilation_H)
        self.dilation_W_ = cl.cltypes.int(dilation_W)
        self.stride_H_ = cl.cltypes.int(stride_H)
        self.stride_W_ = cl.cltypes.int(stride_W)
        self.res_size_ = cl.cltypes.int(dst_H * dst_W)
        self.M_ = cl.cltypes.int(N_filters)
        self.K_ = cl.cltypes.int(krnl_H * krnl_W * C)

        self.TILE_SIZE_N = 64
        self.TILE_SIZE_M = 64
        self.ELEMENTS_PER_THREAD_N = 8
        self.ELEMENTS_PER_THREAD_M = 8
        self.TILE_SIZE_K = 7
        self.THREADS_PER_TILE_M = self.TILE_SIZE_M // self.ELEMENTS_PER_THREAD_M
        self.THREADS_PER_TILE_N = self.TILE_SIZE_N // self.ELEMENTS_PER_THREAD_N

        with open('conv_kernels/gemm/gemm_preproc.cl', 'r') as f:
            GEMM_preproc = f.read()

        with open('conv_kernels/gemm/gemm.cl', 'r') as f:
            GEMM = f.read()

        prg_preproc = cl.Program(ctx, GEMM_preproc).build()
        prg_gemm = cl.Program(ctx, GEMM).build()

        self.knl_preproc = prg_preproc.GEMM_preproc_data
        self.knl_gemm = prg_gemm.GEMM

        if kernels is None:
            kernels = np.random.randn(N_filters, C, krnl_H, krnl_W).astype(np.float32)

        kernels_t = np.transpose(kernels.reshape(N_filters, -1)).reshape(-1)

        self.kernel_arr = []

        for i in range(using_gpu_num):

            cur_kernels = cl_array.to_device(queue_arr[i], kernels_t)

            self.kernel_arr.append(cur_kernels)

        self.bytes_per_img_src = H * W * C * 4
        self.N_img_per_gpu = N_img // using_gpu_num
        self.img_reminder = N_img % using_gpu_num

        self.transposed_dev_arr = []

        for i in range(self.using_gpu_num):
            self.transposed_dev_arr.append(
                cl_array.empty(
                    self.queue_arr[i],
                    dst_H * dst_W * (self.N_img_per_gpu + (i < self.img_reminder)) *
                    krnl_H * krnl_W * C, np.float32)
            )

        self.dest_buf = cl.Buffer(ctx, mf.READ_WRITE, 
                                  size=N_img * N_filters * dst_H * dst_W * 4)

        self.dest_buf_arr = []

        bytes_per_img_dest = N_filters * dst_H * dst_W * 4

        for i in range(using_gpu_num):
            if i < self.img_reminder:
                dest_buf_сur = self.dest_buf.get_sub_region(
                    ((self.N_img_per_gpu + 1) * i) * bytes_per_img_dest,
                    ((self.N_img_per_gpu + 1) * (i + 1) * bytes_per_img_dest)
                )
            else:
                dest_buf_сur = self.dest_buf.get_sub_region(
                    (self.N_img_per_gpu * i + self.img_reminder) * bytes_per_img_dest,
                    (((self.N_img_per_gpu) * (i + 1) + self.img_reminder) * bytes_per_img_dest)
                )
            self.dest_buf_arr.append(dest_buf_сur)

    def forward(self, data: cl.Buffer):

        gpu_data_arr = []
        for i in range(self.using_gpu_num):
            if i < self.img_reminder:
                gpu_data = data.get_sub_region(
                    ((self.N_img_per_gpu + 1) * i) * self.bytes_per_img_src,
                    ((self.N_img_per_gpu + 1) * (i + 1)) * self.bytes_per_img_src
                )
            else:
                gpu_data = data.get_sub_region(
                    (self.N_img_per_gpu * i + self.img_reminder) * self.bytes_per_img_src,
                    (self.N_img_per_gpu * (i + 1) + self.img_reminder) * self.bytes_per_img_src
                )
                
            gpu_data_arr.append(gpu_data)

        event_arr = []

        for i in range(self.using_gpu_num):
            N_img_ = cl.cltypes.int(self.N_img_per_gpu + (i < self.img_reminder))

            event = self.knl_preproc(
                self.queue_arr[i],
                ((self.N_img_per_gpu + (i < self.img_reminder)) * self.C *
                 self.dst_H * self.dst_W,), (64,), gpu_data_arr[i], self.H_,
                self.W_, self.dst_H_, self.dst_W_, self.pad_H_, self.pad_W_,
                N_img_, self.C_, self.krnl_H_, self.krnl_W_, self.dilation_H_,
                self.dilation_W_, self.stride_H_, self.stride_W_,
                self.transposed_dev_arr[i].data
            )
            
            event_arr.append(event)

        for event in event_arr:
            event.wait()

        event_arr = []

        for i in range(self.using_gpu_num):
            N_ = cl.cltypes.int(self.dst_H * self.dst_W *
                                (self.N_img_per_gpu + (i < self.img_reminder)))

            event = self.knl_gemm(
                self.queue_arr[i],
                (max(self.N_filters // self.ELEMENTS_PER_THREAD_M, self.THREADS_PER_TILE_M),
                 max(self.dst_H * self.dst_W *
                     (self.N_img_per_gpu + (i < self.img_reminder)) // self.ELEMENTS_PER_THREAD_N,
                     self.THREADS_PER_TILE_N)),
                (self.THREADS_PER_TILE_M, self.THREADS_PER_TILE_N), self.M_, N_, self.K_, 
                self.res_size_, self.kernel_arr[i].data, self.transposed_dev_arr[i].data, 
                self.dest_buf_arr[i]
            )
            
            event_arr.append(event)

        for event in event_arr:
            event.wait()

        return self.dest_buf

    def release_data(self):
        for i in range(self.using_gpu_num):
            self.dest_buf_arr[i].release()
            self.transposed_dev_arr[i].data.release()
            self.kernel_arr[i].data.release()


class GEMMImplicit():

    def __init__(self, ctx: cl.Context, using_gpu_num: int, queue_arr: list[cl.CommandQueue],
                 N_img: int, H: int, W: int, C: int, N_filters: int, dst_H: int, dst_W: int,
                 krnl_H: int, krnl_W: int, pad_H: int, pad_W: int,
                 kernels: Optional[list[np.ndarray]] = None, dilation_H: int = 1, 
                 dilation_W: int = 1, stride_H: int = 1, stride_W: int = 1):

        self.N_filters = N_filters
        self.bytes_per_img_src = H * W * C * 4
        self.N_img_per_gpu = N_img // using_gpu_num
        self.img_reminder = N_img % using_gpu_num
        self.using_gpu_num = using_gpu_num
        self.queue_arr = queue_arr
        self.dst_H = dst_H
        self.dst_W = dst_W
        self.H_ = cl.cltypes.int(H)
        self.C_ = cl.cltypes.int(C)
        self.W_ = cl.cltypes.int(W)
        self.dst_H_ = cl.cltypes.int(dst_H)
        self.dst_W_ = cl.cltypes.int(dst_W)
        self.pad_H_ = cl.cltypes.int(pad_H)
        self.pad_W_ = cl.cltypes.int(pad_W)
        self.N_img_ = cl.cltypes.int(N_img)
        self.krnl_H_ = cl.cltypes.int(krnl_H)
        self.krnl_W_ = cl.cltypes.int(krnl_W)
        self.dilation_H_ = cl.cltypes.int(dilation_H)
        self.dilation_W_ = cl.cltypes.int(dilation_W)
        self.stride_H_ = cl.cltypes.int(stride_H)
        self.stride_W_ = cl.cltypes.int(stride_W)
        self.N_filters_ = cl.cltypes.int(N_filters)

        self.TILE_SIZE_N = 64
        self.TILE_SIZE_M = 64
        self.ELEMENTS_PER_THREAD_N = 8
        self.ELEMENTS_PER_THREAD_M = 8
        self.TILE_SIZE_K = 7
        self.THREADS_PER_TILE_M = self.TILE_SIZE_M // self.ELEMENTS_PER_THREAD_M
        self.THREADS_PER_TILE_N = self.TILE_SIZE_N // self.ELEMENTS_PER_THREAD_N

        with open('conv_kernels/gemm_implicit/gemm_implicit.cl', 'r') as f:
            gemm_implicit = f.read()

        prg = cl.Program(ctx, gemm_implicit).build()
        self.knl = prg.GEMM_implicit

        if kernels is None:
            kernels = np.random.randn(N_filters, C, krnl_H, krnl_W).astype(np.float32)

        kernels_t = np.transpose(kernels.reshape(N_filters, -1)).reshape(-1)

        self.kernel_arr = []

        for i in range(using_gpu_num):

            cur_kernels = cl_array.to_device(queue_arr[i], kernels_t)

            self.kernel_arr.append(cur_kernels)

        self.dest_buf = cl.Buffer(ctx, mf.READ_WRITE,
                                  size=N_img * N_filters * dst_H * dst_W * 4)

        self.dest_buf_arr = []

        bytes_per_img_dest = N_filters * dst_H * dst_W * 4

        self.dest_buf = cl.Buffer(ctx, mf.READ_WRITE,
                                  size=N_img * N_filters * dst_H * dst_W * 4)

        for i in range(using_gpu_num):
            if i < self.img_reminder:
                dest_buf_сur = self.dest_buf.get_sub_region(
                    ((self.N_img_per_gpu + 1) * i) * bytes_per_img_dest,
                    ((self.N_img_per_gpu + 1) * (i + 1) * bytes_per_img_dest)
                )
            else:
                dest_buf_сur = self.dest_buf.get_sub_region(
                    (self.N_img_per_gpu * i + self.img_reminder) * bytes_per_img_dest,
                    (((self.N_img_per_gpu) * (i + 1) + self.img_reminder) * bytes_per_img_dest)
                )
            self.dest_buf_arr.append(dest_buf_сur)

    def forward(self, data: cl.Buffer):

        gpu_data_arr = []
        for i in range(self.using_gpu_num):
            if i < self.img_reminder:
                gpu_data = data.get_sub_region(
                    ((self.N_img_per_gpu + 1) * i) * self.bytes_per_img_src,
                    ((self.N_img_per_gpu + 1) * (i + 1)) * self.bytes_per_img_src
                )
            else:
                gpu_data = data.get_sub_region(
                    (self.N_img_per_gpu * i + self.img_reminder) * self.bytes_per_img_src,
                    (self.N_img_per_gpu * (i + 1) + self.img_reminder) * self.bytes_per_img_src
                )
            gpu_data_arr.append(gpu_data)

        event_arr = []

        for i in range(self.using_gpu_num):
            N_img_ = cl.cltypes.int(self.N_img_per_gpu + (i < self.img_reminder))

            event = self.knl(
                self.queue_arr[i],
                (max(self.N_filters // self.ELEMENTS_PER_THREAD_M, self.THREADS_PER_TILE_M),
                 max(self.dst_H * self.dst_W *
                     (self.N_img_per_gpu + (i < self.img_reminder)) // self.ELEMENTS_PER_THREAD_N, 
                     self.THREADS_PER_TILE_N)),
                (self.THREADS_PER_TILE_M, self.THREADS_PER_TILE_N), gpu_data_arr[i],
                self.kernel_arr[i].data, self.H_, self.W_, self.dst_H_, self.dst_W_, 
                self.pad_H_, self.pad_W_, N_img_, self.C_, self.N_filters_, self.krnl_H_,
                self.krnl_W_, self.dilation_H_, self.dilation_W_, self.stride_H_, 
                self.stride_W_, self.dest_buf_arr[i]
            )
            
            event_arr.append(event)

        for event in event_arr:
            event.wait()

        return self.dest_buf

    def release_data(self):
        for i in range(self.using_gpu_num):
            self.dest_buf_arr[i].release()
            self.kernel_arr[i].data.release()
