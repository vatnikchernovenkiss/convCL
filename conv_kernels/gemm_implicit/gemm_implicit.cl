#define TileSizeK 7
#define TileSizeN 64
#define TileSizeM 64
#define ThreadBufSizeN 8
#define ThreadBufSizeM 8

kernel void GEMM_implicit(const __global float *data, const __global float *filters, 
                          const int H, const int W, const int dst_H, const int dst_W, 
                          const int pad_H, const int pad_W, const int N_objects,
                          const int N_channels, const int N_filters, const int kern_H,
                          const int kern_W, const int dilation_H, const int dilation_W,
                          const int stride_H, const int stride_W, float __global *output) {

    const int res_img_size = dst_H * dst_W;

    const int M = N_filters, N = res_img_size * N_objects, K = kern_H * kern_W * N_channels;

    const int GroupSizeM = get_local_size(0);        
    const int GroupSizeN = get_local_size(1);        
    const int ThreadLoadsA = ((TileSizeK * TileSizeM) / (GroupSizeM * GroupSizeN));
	const int ThreadLoadsB = ((TileSizeK * TileSizeN) / (GroupSizeM * GroupSizeN)); 

    const int ThreadIdM = get_local_id(0); 
    const int ThreadIdN = get_local_id(1); 
    const int offsetM = TileSizeM * get_group_id(0); 
    const int offsetN = TileSizeN * get_group_id(1);
    __local float Alocal[TileSizeK][TileSizeM];
    __local float Blocal[TileSizeN][TileSizeK];

    float Apriv;
    float Bpriv[ThreadBufSizeN];
    float res[ThreadBufSizeM][ThreadBufSizeN];

    for (int m = 0; m < ThreadBufSizeM; m++) {
        for (int n = 0; n < ThreadBufSizeN; n++) {
            res[m][n] = 0;
        }
    }
    
    int tid = ThreadIdN*GroupSizeM + ThreadIdM;
    int row_im2col = tid % TileSizeN;
            
    int img_num = (row_im2col + offsetN) / res_img_size;
    int img_offset =  img_num * H * W * N_channels;
    int conv_num = (row_im2col + offsetN) % res_img_size;
    int row_num = (conv_num / dst_W) * stride_H - pad_H;
    int col_num = (conv_num % dst_W) * stride_W - pad_W;
    
    int numTiles = K / TileSizeK + ((K % TileSizeK) > 0);
    for (int t = 0; t < numTiles; t++) {

        for (int l = 0; l < ThreadLoadsA; l++) {
            int col_im2col = tid / TileSizeN + l;
            int tiledIndex = TileSizeK*t + col_im2col;
            
            if ((tiledIndex >= K) || ((offsetM + row_im2col) >= M)) {
                Alocal[col_im2col][row_im2col] = 0;
            } else {
                Alocal[col_im2col][row_im2col] = filters[tiledIndex * M + offsetM + row_im2col];
            }
        }

        for (int l = 0; l < ThreadLoadsB; l++) {
            int col_im2col = tid / TileSizeN + l;
            
            int tiledIndex = TileSizeK*t + col_im2col;
            int channel_num = tiledIndex / (kern_H * kern_W);
            int elem_num = tiledIndex % (kern_H * kern_W);

            int cur_row_num = row_num + (elem_num / kern_W) * dilation_H;
            int cur_col_num = col_num + (elem_num % kern_W) * dilation_W;
            
            int need_elem_idx = img_offset + channel_num * H * W + cur_row_num * W + cur_col_num;
            
            if ((tiledIndex >= K) || (cur_row_num < 0) || (cur_row_num >= H) || (cur_col_num < 0) || 
                (cur_col_num >= W)) {
                Blocal[row_im2col][col_im2col] = 0;
            } else {
                Blocal[row_im2col][col_im2col] = data[need_elem_idx];
            }
        }


        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TileSizeK; k++) {
            for (int n = 0; n < ThreadBufSizeN; n++) {
                int col = ThreadIdN + n * GroupSizeN;
                Bpriv[n] = Blocal[col][k];
            }

            for (int m = 0; m < ThreadBufSizeM; m++) {
                int row = ThreadIdM + m * GroupSizeM;
                Apriv = Alocal[k][row];
                for (int n = 0; n < ThreadBufSizeN; n++) {
                    res[m][n] += Apriv * Bpriv[n];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int m = 0; m < ThreadBufSizeM; m++) {
        int globalRow = offsetM + ThreadIdM + m * GroupSizeM;
        for (int n = 0; n < ThreadBufSizeN; n++) {
            int globalCol = offsetN + ThreadIdN + n * GroupSizeN,
                cur_img_el_num = globalCol % res_img_size,
                cur_img_num = globalCol / res_img_size, cur_filter_num = globalRow;
                 
            if ((cur_img_num < N_objects) && (cur_filter_num < N_filters)
            && (cur_img_el_num < res_img_size)) { 
        		output[cur_img_num * res_img_size * M + cur_filter_num * res_img_size + cur_img_el_num]
                = res[m][n];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
