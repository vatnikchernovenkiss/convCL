#define  img_buf_H 6
#define  img_buf_W 6

#define  transformed_buf_H 6
#define  transformed_buf_W 6

#define output_H 2
#define output_W 2

#define TILES_PER_GROUP 16
#define CHANNELS_PER_GROUP 9

#define ACCUM_BUF_SIZE_IMG 8
#define ACCUM_BUF_SIZE_FILTER 8

#define N_ITER_PER_ONE_BUF_ELEM_IMG  2
#define N_ITER_PER_ONE_BUF_ELEM_FILTER  2

#define THREADS_PER_ONE_BUF_ELEM 4
#define transformed_buf_size 36

#define local_buf_size 2 * CHANNELS_PER_GROUP * TILES_PER_GROUP * transformed_buf_size


kernel void winograd(const __global float *data, const __global float *filters, const int H, const int W,
                     const int pad_H, const int pad_W, const int N_objects, const int N_channels, 
                     const int N_filters, float __global *output) {

    const int img_group_size = get_local_size(0);
    const int img_group_id = get_group_id(0);
    const int img_local_id = get_local_id(0);
    const int img_local_tile_id = img_local_id % TILES_PER_GROUP;
    int img_tile_offset_group = TILES_PER_GROUP * img_group_id;
    int img_tile_offset = img_tile_offset_group + img_local_tile_id;

    const int tiles_per_row = W / output_W, tiles_per_col = H / output_H;
    const int img_tile_num = tiles_per_row * tiles_per_col;
    const int img_num = img_tile_offset / img_tile_num;
    img_tile_offset_group = img_tile_offset_group % img_tile_num;
    img_tile_offset = img_tile_offset % img_tile_num;

    const int row_to_process_init_group = (img_tile_offset_group / tiles_per_row) * output_H;
    const int col_to_process_init_group = (img_tile_offset_group % tiles_per_row) * output_W;
    const int row_to_process_init = (img_tile_offset / tiles_per_row) * output_H - pad_H;
    const int col_to_process_init = (img_tile_offset % tiles_per_row) * output_W - pad_W;
    const int img_channel_begin = img_local_id / TILES_PER_GROUP;

    const int filter_group_id = get_group_id(1);
    const int filter_local_id = img_local_id;
    const int filter_local_tile_id = img_local_tile_id; 

    const int filter_tile_offset_group = TILES_PER_GROUP * filter_group_id;
    int filter_tile_offset = filter_tile_offset_group + filter_local_tile_id;
    const int filter_channel_begin = img_channel_begin;

    float matrix_buf[transformed_buf_size] = {0}, transformed_matrix_buf[transformed_buf_size] = {0};
    const int img_size = H * W, img_buf_size = img_buf_H * img_buf_W;

    const int filter_num = filter_tile_offset_group + img_local_id;

    float accum_buf[ACCUM_BUF_SIZE_FILTER][ACCUM_BUF_SIZE_IMG] = {{0}};
    float accum_img[ACCUM_BUF_SIZE_IMG] = {0}, accum_kern[ACCUM_BUF_SIZE_FILTER] = {0};

    __local float local_buf[local_buf_size];

    const int local_img_buf_size = transformed_buf_size * CHANNELS_PER_GROUP * TILES_PER_GROUP;

    int img_offset_c = img_num * img_size * N_channels + img_size * img_channel_begin;
    int filter_offset = filter_tile_offset * transformed_buf_size * N_channels + 
                        transformed_buf_size * filter_channel_begin;
    const int local_el_stride = TILES_PER_GROUP * CHANNELS_PER_GROUP;
    
    int cur_row = row_to_process_init;
    int cur_col = col_to_process_init;
    for (int c = 0; c < N_channels; c += CHANNELS_PER_GROUP) {

        if ((row_to_process_init < H) && (col_to_process_init < W) && (img_num < N_objects) && 
             ((img_channel_begin + c) < N_channels)) {

            for (int my_el = 0; my_el < img_buf_size; ++my_el) {
                cur_row = row_to_process_init + my_el / img_buf_W;
                cur_col = col_to_process_init + my_el % img_buf_W;
                
                if ((cur_row >= 0) && (cur_row < H) && (cur_col >= 0) && (cur_col < W)) {
                    matrix_buf[(my_el % img_buf_W) * img_buf_W + my_el / img_buf_W] =
                    data[img_offset_c + c * img_size + (row_to_process_init + (my_el / img_buf_W)) * W + 
                         col_to_process_init + my_el % img_buf_W];
                } else {
                    matrix_buf[(my_el % img_buf_W) * img_buf_W + my_el / img_buf_W] = 0;
                }
            }

            transformed_matrix_buf[0] = 4 * matrix_buf[0] - 5 * matrix_buf[2] + matrix_buf[4];
            transformed_matrix_buf[1] = 4 * matrix_buf[6] - 5 * matrix_buf[8] + matrix_buf[10];
            transformed_matrix_buf[2] = 4 * matrix_buf[12] - 5 * matrix_buf[14] + matrix_buf[16];
            transformed_matrix_buf[3] = 4 * matrix_buf[18] - 5 * matrix_buf[20] + matrix_buf[22];
            transformed_matrix_buf[4] = 4 * matrix_buf[24] - 5 * matrix_buf[26] + matrix_buf[28];
            transformed_matrix_buf[5] = 4 * matrix_buf[30] - 5 * matrix_buf[32] + matrix_buf[34];
            transformed_matrix_buf[6] = -4 * (matrix_buf[1] + matrix_buf[2]) + matrix_buf[3] + matrix_buf[4];
            transformed_matrix_buf[7] = -4 * (matrix_buf[7] + matrix_buf[8]) + matrix_buf[9] + matrix_buf[10];
            transformed_matrix_buf[8] = -4 * (matrix_buf[13] + matrix_buf[14]) + matrix_buf[15] + 
                                        matrix_buf[16];
            transformed_matrix_buf[9] = -4 * (matrix_buf[19] + matrix_buf[20]) + matrix_buf[21] + 
                                        matrix_buf[22];
            transformed_matrix_buf[10] = -4 * (matrix_buf[25] + matrix_buf[26]) + matrix_buf[27] + 
                                         matrix_buf[28];
            transformed_matrix_buf[11] = -4 * (matrix_buf[31] + matrix_buf[32]) + matrix_buf[33] + 
                                         matrix_buf[34];
            transformed_matrix_buf[12] = 4 * (matrix_buf[1] - matrix_buf[2]) - matrix_buf[3] + matrix_buf[4];
            transformed_matrix_buf[13] = 4 * (matrix_buf[7] - matrix_buf[8]) - matrix_buf[9] + matrix_buf[10];
            transformed_matrix_buf[14] = 4 * (matrix_buf[13] - matrix_buf[14]) - matrix_buf[15] + 
                                         matrix_buf[16];
            transformed_matrix_buf[15] = 4 * (matrix_buf[19] - matrix_buf[20]) - matrix_buf[21] + 
                                         matrix_buf[22];
            transformed_matrix_buf[16] = 4 * (matrix_buf[25] - matrix_buf[26]) - matrix_buf[27] + 
                                         matrix_buf[28];
            transformed_matrix_buf[17] = 4 * (matrix_buf[31] - matrix_buf[32]) - matrix_buf[33] + 
                                         matrix_buf[34];
            transformed_matrix_buf[18] = 2 * (matrix_buf[3] - matrix_buf[1]) + matrix_buf[4] - matrix_buf[2];
            transformed_matrix_buf[19] = 2 * (matrix_buf[9] - matrix_buf[7]) + matrix_buf[10] - matrix_buf[8];
            transformed_matrix_buf[20] = 2 * (matrix_buf[15] - matrix_buf[13]) + matrix_buf[16] - 
                                         matrix_buf[14];
            transformed_matrix_buf[21] = 2 * (matrix_buf[21] - matrix_buf[19]) + matrix_buf[22] - 
                                         matrix_buf[20];
            transformed_matrix_buf[22] = 2 * (matrix_buf[27] - matrix_buf[25]) + matrix_buf[28] - 
                                         matrix_buf[26];
            transformed_matrix_buf[23] = 2 * (matrix_buf[33] - matrix_buf[31]) + matrix_buf[34] - 
                                         matrix_buf[32];
            transformed_matrix_buf[24] = 2 * (matrix_buf[1] - matrix_buf[3]) + matrix_buf[4] - matrix_buf[2];
            transformed_matrix_buf[25] = 2 * (matrix_buf[7] - matrix_buf[9]) + matrix_buf[10] - matrix_buf[8];
            transformed_matrix_buf[26] = 2 * (matrix_buf[13] - matrix_buf[15]) + matrix_buf[16] - 
                                         matrix_buf[14];
            transformed_matrix_buf[27] = 2 * (matrix_buf[19] - matrix_buf[21]) + matrix_buf[22] - 
                                         matrix_buf[20];
            transformed_matrix_buf[28] = 2 * (matrix_buf[25] - matrix_buf[27]) + matrix_buf[28] - 
                                         matrix_buf[26];
            transformed_matrix_buf[29] = 2 * (matrix_buf[31] - matrix_buf[33]) + matrix_buf[34] - 
                                        matrix_buf[32];
            transformed_matrix_buf[30] = 4 * matrix_buf[1] - 5 * matrix_buf[3] + matrix_buf[5];
            transformed_matrix_buf[31] = 4 * matrix_buf[7] - 5 * matrix_buf[9] + matrix_buf[11];
            transformed_matrix_buf[32] = 4 * matrix_buf[13] - 5 * matrix_buf[15] + matrix_buf[17];
            transformed_matrix_buf[33] = 4 * matrix_buf[19] - 5 * matrix_buf[21] + matrix_buf[23];
            transformed_matrix_buf[34] = 4 * matrix_buf[25] - 5 * matrix_buf[27] + matrix_buf[29];
            transformed_matrix_buf[35] = 4 * matrix_buf[31] - 5 * matrix_buf[33] + matrix_buf[35];

            matrix_buf[0] = 4 * transformed_matrix_buf[0] - 5 * transformed_matrix_buf[2] + 
                            transformed_matrix_buf[4];
            matrix_buf[1] = -4 * (transformed_matrix_buf[1] + transformed_matrix_buf[2]) + 
                             transformed_matrix_buf[3] + transformed_matrix_buf[4];
            matrix_buf[2] = 4 * (transformed_matrix_buf[1] - transformed_matrix_buf[2]) - 
                            transformed_matrix_buf[3] + transformed_matrix_buf[4];
            matrix_buf[3] = 2 * (transformed_matrix_buf[3] - transformed_matrix_buf[1]) + 
                            transformed_matrix_buf[4] - transformed_matrix_buf[2];
            matrix_buf[4] = 2 * (transformed_matrix_buf[1] - transformed_matrix_buf[3]) + 
                            transformed_matrix_buf[4] - transformed_matrix_buf[2];
            matrix_buf[5] = 4 * transformed_matrix_buf[1] - 5 * transformed_matrix_buf[3] + 
                            transformed_matrix_buf[5];
            matrix_buf[6] = 4 * transformed_matrix_buf[6] - 5 * transformed_matrix_buf[8] + 
                            transformed_matrix_buf[10];
            matrix_buf[7] = -4 * (transformed_matrix_buf[7] + transformed_matrix_buf[8]) + 
                            transformed_matrix_buf[9] + transformed_matrix_buf[10];
            matrix_buf[8] = 4 * (transformed_matrix_buf[7] - transformed_matrix_buf[8]) - 
                            transformed_matrix_buf[9] + transformed_matrix_buf[10];
            matrix_buf[9] = 2 * (transformed_matrix_buf[9] - transformed_matrix_buf[7]) + 
                            transformed_matrix_buf[10] - transformed_matrix_buf[8];
            matrix_buf[10] = 2 * (transformed_matrix_buf[7] - transformed_matrix_buf[9]) + 
                             transformed_matrix_buf[10] - transformed_matrix_buf[8];
            matrix_buf[11] = 4 * transformed_matrix_buf[7] - 5 * transformed_matrix_buf[9] +
                             transformed_matrix_buf[11];
            matrix_buf[12] = 4 * transformed_matrix_buf[12] - 5 * transformed_matrix_buf[14] +
                             transformed_matrix_buf[16];
            matrix_buf[13] = -4 * (transformed_matrix_buf[13] + transformed_matrix_buf[14]) + 
                             transformed_matrix_buf[15] + transformed_matrix_buf[16];
            matrix_buf[14] = 4 * (transformed_matrix_buf[13] - transformed_matrix_buf[14]) - 
                             transformed_matrix_buf[15] + transformed_matrix_buf[16];
            matrix_buf[15] = 2 * (transformed_matrix_buf[15] - transformed_matrix_buf[13]) + 
                             transformed_matrix_buf[16] - transformed_matrix_buf[14];
            matrix_buf[16] = 2 * (transformed_matrix_buf[13] - transformed_matrix_buf[15]) + 
                             transformed_matrix_buf[16] - transformed_matrix_buf[14];
            matrix_buf[17] = 4 * transformed_matrix_buf[13] - 5 * transformed_matrix_buf[15] + 
                             transformed_matrix_buf[17];
            matrix_buf[18] = 4 * transformed_matrix_buf[18] - 5 * transformed_matrix_buf[20] + 
                             transformed_matrix_buf[22];
            matrix_buf[19] = -4 * (transformed_matrix_buf[19] + transformed_matrix_buf[20]) + 
                             transformed_matrix_buf[21] + transformed_matrix_buf[22];
            matrix_buf[20] = 4 * (transformed_matrix_buf[19] - transformed_matrix_buf[20]) - 
                             transformed_matrix_buf[21] + transformed_matrix_buf[22];
            matrix_buf[21] = 2 * (transformed_matrix_buf[21] - transformed_matrix_buf[19]) + 
                             transformed_matrix_buf[22] - transformed_matrix_buf[20];
            matrix_buf[22] = 2 * (transformed_matrix_buf[19] - transformed_matrix_buf[21]) + 
                             transformed_matrix_buf[22] - transformed_matrix_buf[20];
            matrix_buf[23] = 4 * transformed_matrix_buf[19] - 5 * transformed_matrix_buf[21] + 
                             transformed_matrix_buf[23];
            matrix_buf[24] = 4 * transformed_matrix_buf[24] - 5 * transformed_matrix_buf[26] + 
                             transformed_matrix_buf[28];
            matrix_buf[25] = -4 * (transformed_matrix_buf[25] + transformed_matrix_buf[26]) + 
                             transformed_matrix_buf[27] + transformed_matrix_buf[28];
            matrix_buf[26] = 4 * (transformed_matrix_buf[25] - transformed_matrix_buf[26]) - 
                             transformed_matrix_buf[27] + transformed_matrix_buf[28];
            matrix_buf[27] = 2 * (transformed_matrix_buf[27] - transformed_matrix_buf[25]) + 
                             transformed_matrix_buf[28] - transformed_matrix_buf[26];
            matrix_buf[28] = 2 * (transformed_matrix_buf[25] - transformed_matrix_buf[27]) + 
                             transformed_matrix_buf[28] - transformed_matrix_buf[26];
            matrix_buf[29] = 4 * transformed_matrix_buf[25] - 5 * transformed_matrix_buf[27] + 
                             transformed_matrix_buf[29];
            matrix_buf[30] = 4 * transformed_matrix_buf[30] - 5 * transformed_matrix_buf[32] + 
                             transformed_matrix_buf[34];
            matrix_buf[31] = -4 * (transformed_matrix_buf[31] + transformed_matrix_buf[32]) + 
                             transformed_matrix_buf[33] + transformed_matrix_buf[34];
            matrix_buf[32] = 4 * (transformed_matrix_buf[31] - transformed_matrix_buf[32]) - 
                             transformed_matrix_buf[33] + transformed_matrix_buf[34];
            matrix_buf[33] = 2 * (transformed_matrix_buf[33] - transformed_matrix_buf[31]) + 
                             transformed_matrix_buf[34] - transformed_matrix_buf[32];
            matrix_buf[34] = 2 * (transformed_matrix_buf[31] - transformed_matrix_buf[33]) +
                             transformed_matrix_buf[34] - transformed_matrix_buf[32];
            matrix_buf[35] = 4 * transformed_matrix_buf[31] - 5 * transformed_matrix_buf[33] + 
                             transformed_matrix_buf[35];
        }

        if ((filter_tile_offset < N_filters) && ((img_channel_begin + c) < N_channels)) {

            for (int i = 0; i < transformed_buf_H; ++ i) {
                for (int j = 0; j < transformed_buf_W; ++j) {
                  transformed_matrix_buf[i * transformed_buf_W + j] = filters[filter_offset + c * 
                                                                              transformed_buf_size + i *  
                                                                              transformed_buf_W + j];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < transformed_buf_size; ++i) {
          local_buf[i * CHANNELS_PER_GROUP * TILES_PER_GROUP + img_channel_begin * TILES_PER_GROUP + 
                    img_local_tile_id] = matrix_buf[i];
        }
        
        for (int i = 0; i < transformed_buf_size; ++i) {
          local_buf[local_img_buf_size + i * CHANNELS_PER_GROUP * TILES_PER_GROUP + filter_channel_begin * 
                    TILES_PER_GROUP + filter_local_tile_id] = transformed_matrix_buf[i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int channel_num = CHANNELS_PER_GROUP > N_channels - c ? N_channels - c : CHANNELS_PER_GROUP;

        for (int k = 0; k < channel_num; ++k) {
            for (int el = 0; el < ACCUM_BUF_SIZE_IMG; ++el) {

               accum_img[el] = local_buf[(img_local_id / (img_group_size / transformed_buf_size)) * 
                                         CHANNELS_PER_GROUP * TILES_PER_GROUP + k * TILES_PER_GROUP + 
                                         ((img_local_id % THREADS_PER_ONE_BUF_ELEM) / 
                                          N_ITER_PER_ONE_BUF_ELEM_IMG) * ACCUM_BUF_SIZE_IMG + el];
            }
            for (int el = 0; el < ACCUM_BUF_SIZE_FILTER; ++el) {

               accum_kern[el] = local_buf[local_img_buf_size + 
                                          (filter_local_id / (img_group_size / transformed_buf_size)) * 
                                          CHANNELS_PER_GROUP * TILES_PER_GROUP + k * TILES_PER_GROUP + 
                                          ((filter_local_id % THREADS_PER_ONE_BUF_ELEM) % 
                                           N_ITER_PER_ONE_BUF_ELEM_FILTER) + 
                                          el * N_ITER_PER_ONE_BUF_ELEM_FILTER];
            }
            
            for (int m = 0; m < ACCUM_BUF_SIZE_FILTER; ++m) {
                for (int n = 0; n < ACCUM_BUF_SIZE_IMG; ++n) {
                    accum_buf[m][n] += accum_kern[m] * accum_img[n];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    const int filters_in_buf = 2 * CHANNELS_PER_GROUP > TILES_PER_GROUP ?
                               TILES_PER_GROUP : 2 * CHANNELS_PER_GROUP ;
    const int n_output_iter = TILES_PER_GROUP / (filters_in_buf) + ((TILES_PER_GROUP % (filters_in_buf)) > 0);
    const int n_filters_per_buf_iter = filters_in_buf / N_ITER_PER_ONE_BUF_ELEM_FILTER;
    for (int filter_count = 0, accum_buf_beg=0; filter_count < TILES_PER_GROUP; 
         filter_count += filters_in_buf, accum_buf_beg++) {
        for (int m = 0; m < n_filters_per_buf_iter; ++m) {
            for (int n = 0; n < ACCUM_BUF_SIZE_IMG; ++n) {
    
                local_buf[(((filter_local_id % THREADS_PER_ONE_BUF_ELEM) % N_ITER_PER_ONE_BUF_ELEM_FILTER) +
                            m * N_ITER_PER_ONE_BUF_ELEM_FILTER) * TILES_PER_GROUP * transformed_buf_size +
                            (((img_local_id % THREADS_PER_ONE_BUF_ELEM) / N_ITER_PER_ONE_BUF_ELEM_IMG) *
                             ACCUM_BUF_SIZE_IMG + n) * transformed_buf_size + img_local_id /
                            (img_group_size / transformed_buf_size)] = accum_buf[n_filters_per_buf_iter * 
                                                                                 accum_buf_beg + m][n];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    
        for (int i = img_local_id; i < filters_in_buf * TILES_PER_GROUP; 
             i += TILES_PER_GROUP * CHANNELS_PER_GROUP) {

            for (int j = 0; j < transformed_buf_size; ++j) {
                transformed_matrix_buf[(j % transformed_buf_W) * transformed_buf_W + j / transformed_buf_W] = 
                local_buf[(i / TILES_PER_GROUP) * TILES_PER_GROUP * transformed_buf_size +
                               (i % TILES_PER_GROUP) * transformed_buf_size + j];
            }
    
            matrix_buf[0] = transformed_matrix_buf[0] + transformed_matrix_buf[1] + transformed_matrix_buf[2] +
                            transformed_matrix_buf[3] + transformed_matrix_buf[4];
            matrix_buf[1] = transformed_matrix_buf[6] + transformed_matrix_buf[7] + transformed_matrix_buf[8] + 
                            transformed_matrix_buf[9] + transformed_matrix_buf[10];
            matrix_buf[2] = transformed_matrix_buf[12] + transformed_matrix_buf[13] + 
                            transformed_matrix_buf[14] + transformed_matrix_buf[15] + 
                            transformed_matrix_buf[16];
            matrix_buf[3] = transformed_matrix_buf[18] + transformed_matrix_buf[19] + 
                            transformed_matrix_buf[20] + transformed_matrix_buf[21] + 
                            transformed_matrix_buf[22];
            matrix_buf[4] = transformed_matrix_buf[24] + transformed_matrix_buf[25] + 
                            transformed_matrix_buf[26] + transformed_matrix_buf[27] + 
                            transformed_matrix_buf[28];
            matrix_buf[5] = transformed_matrix_buf[30] + transformed_matrix_buf[31] + 
                            transformed_matrix_buf[32] + transformed_matrix_buf[33] + 
                            transformed_matrix_buf[34];
            matrix_buf[6] = transformed_matrix_buf[1] - transformed_matrix_buf[2] + transformed_matrix_buf[5] +
                            2 * (transformed_matrix_buf[3] - transformed_matrix_buf[4]);
            matrix_buf[7] = transformed_matrix_buf[7] - transformed_matrix_buf[8] + transformed_matrix_buf[11]+
                            2 * (transformed_matrix_buf[9] - transformed_matrix_buf[10]);
            matrix_buf[8] = transformed_matrix_buf[13] - transformed_matrix_buf[14] + 
                            transformed_matrix_buf[17] +
                            2 * (transformed_matrix_buf[15] - transformed_matrix_buf[16]);
            matrix_buf[9] = transformed_matrix_buf[19] - transformed_matrix_buf[20] + 
                            transformed_matrix_buf[23] +
                            2 * (transformed_matrix_buf[21] - transformed_matrix_buf[22]);
            matrix_buf[10] = transformed_matrix_buf[25] - transformed_matrix_buf[26] + 
                             transformed_matrix_buf[29] + 
                             2 * (transformed_matrix_buf[27] - transformed_matrix_buf[28]);
            matrix_buf[11] = transformed_matrix_buf[31] - transformed_matrix_buf[32] + 
                             transformed_matrix_buf[35] +
                             2 * (transformed_matrix_buf[33] - transformed_matrix_buf[34]);

            const int output_filter_num =  i / TILES_PER_GROUP + filter_count, 
                      output_tile_num = i % TILES_PER_GROUP;
    
            const int output_row = ((img_tile_offset_group + output_tile_num) / tiles_per_row) * output_H;
            const int output_col = ((img_tile_offset_group + output_tile_num) % tiles_per_row) * output_W;
            const int output_filter = filter_tile_offset_group + output_filter_num;
    
            if ((output_row < H) && (img_num < N_objects) && (output_filter < N_filters)) {
    
                const int output_offset = img_num * img_size * N_filters + output_filter * img_size + 
                                          output_row * W + output_col;
    
                output[output_offset] = matrix_buf[0] + matrix_buf[1] + matrix_buf[2] + matrix_buf[3] +
                                        matrix_buf[4];
                output[output_offset + 1] = matrix_buf[1] - matrix_buf[2] + matrix_buf[5] +
                                            2 * (matrix_buf[3] - matrix_buf[4]);
                output[output_offset + W + 0] = matrix_buf[6] + matrix_buf[7] + matrix_buf[8] + matrix_buf[9] + 
                                                matrix_buf[10];
                output[output_offset + W + 1] = matrix_buf[7] - matrix_buf[8] + matrix_buf[11] +
                                                2 * (matrix_buf[9] - matrix_buf[10]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
