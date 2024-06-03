#define  img_buf_H 4
#define  img_buf_W 4

#define  transformed_buf_H 4
#define  transformed_buf_W 4

#define output_H 2
#define output_W 2

#define TILES_PER_GROUP 4
#define CHANNELS_PER_GROUP 8

#define ACCUM_BUF_SIZE_IMG 2
#define ACCUM_BUF_SIZE_FILTER 4

#define N_ITER_PER_ONE_BUF_ELEM_IMG  1
#define N_ITER_PER_ONE_BUF_ELEM_FILTER  1

#define THREADS_PER_ONE_BUF_ELEM 2

#define transformed_buf_size 16

#define local_buf_size 2 * CHANNELS_PER_GROUP * TILES_PER_GROUP * transformed_buf_size
#define local_img_buf_size transformed_buf_size * CHANNELS_PER_GROUP * TILES_PER_GROUP
#define img_buf_size img_buf_H * img_buf_W

kernel void winograd(const __global float *data, const __global float *filters, const int H,const int W,
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

    float transformed_img_buf[transformed_buf_size] = {0};
    float filter_buf[transformed_buf_size] = {0};
    const int img_size = H * W;
    float img_buf[img_buf_size];

    const int filter_num = filter_tile_offset_group + img_local_id;

    float accum_buf[ACCUM_BUF_SIZE_FILTER][ACCUM_BUF_SIZE_IMG] ={{0}};
    float accum_img[ACCUM_BUF_SIZE_IMG] = {0}, accum_kern[ACCUM_BUF_SIZE_FILTER] = {0};

    __local float local_buf[local_buf_size];

    int img_offset_c = img_num * img_size * N_channels + img_size * img_channel_begin;
    int filter_offset = filter_tile_offset * transformed_buf_size * N_channels + transformed_buf_size * 
                        filter_channel_begin;
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
                    img_buf[(my_el / img_buf_W) * img_buf_W + my_el % img_buf_W] =
                    data[img_offset_c + c * img_size + (row_to_process_init + (my_el / img_buf_W)) * W + 
                         col_to_process_init + my_el % img_buf_W];
                } else {
                    img_buf[(my_el / img_buf_W) * img_buf_W + my_el % img_buf_W] = 0;
                }
            }

            transformed_img_buf[0] = (img_buf[0] - img_buf[8]) - (img_buf[2] - img_buf[10]);
            transformed_img_buf[1] = (img_buf[1] - img_buf[9]) + (img_buf[2] - img_buf[10]);
            transformed_img_buf[2] = (img_buf[2] - img_buf[10]) - (img_buf[1] - img_buf[9]);
            transformed_img_buf[3] = (img_buf[1] - img_buf[9]) - (img_buf[3] - img_buf[11]);
            transformed_img_buf[4] = (img_buf[4] + img_buf[8]) - (img_buf[6] + img_buf[10]);
            transformed_img_buf[5] = (img_buf[5] + img_buf[9]) + (img_buf[6] + img_buf[10]);
            transformed_img_buf[6] = (img_buf[6] + img_buf[10]) - (img_buf[5] + img_buf[9]);
            transformed_img_buf[7] = (img_buf[5] + img_buf[9]) - (img_buf[7] + img_buf[11]);
            transformed_img_buf[8] = (img_buf[8] - img_buf[4]) - (img_buf[10] - img_buf[6]);
            transformed_img_buf[9] = (img_buf[9] - img_buf[5]) + (img_buf[10] - img_buf[6]);
            transformed_img_buf[10] = (img_buf[10] - img_buf[6]) - (img_buf[9] - img_buf[5]);
            transformed_img_buf[11] = (img_buf[9] - img_buf[5]) - (img_buf[11] - img_buf[7]);
            transformed_img_buf[12] = (img_buf[4] - img_buf[12]) - (img_buf[6] - img_buf[14]);
            transformed_img_buf[13] = (img_buf[5] - img_buf[13]) + (img_buf[6] - img_buf[14]);
            transformed_img_buf[14] = (img_buf[6] - img_buf[14]) - (img_buf[5] - img_buf[13]);
            transformed_img_buf[15] = (img_buf[5] - img_buf[13]) - (img_buf[7] - img_buf[15]);
        }

        if ((filter_tile_offset < N_filters) && ((img_channel_begin + c) < N_channels)) {

            for (int i = 0; i < transformed_buf_H; ++ i) {
                for (int j = 0; j < transformed_buf_W; ++j) {
                  filter_buf[i * transformed_buf_W + j] = filters[filter_offset + c * transformed_buf_size + 
                                                                  i * transformed_buf_W + j];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < transformed_buf_size; ++i) {
          local_buf[i * CHANNELS_PER_GROUP * TILES_PER_GROUP + img_channel_begin * TILES_PER_GROUP +
                    img_local_tile_id] = transformed_img_buf[i];
        }
        for (int i = 0; i < transformed_buf_size; ++i) {
          local_buf[local_img_buf_size + i * CHANNELS_PER_GROUP * TILES_PER_GROUP +
                    filter_channel_begin * TILES_PER_GROUP + filter_local_tile_id] = filter_buf[i];
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
                                           el *  N_ITER_PER_ONE_BUF_ELEM_FILTER];
            }
            
            for (int m = 0; m < ACCUM_BUF_SIZE_FILTER; ++m) {
                for (int n = 0; n < ACCUM_BUF_SIZE_IMG; ++n) {
                    accum_buf[m][n] += accum_kern[m] * accum_img[n];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int filters_in_buf = 2 * CHANNELS_PER_GROUP > TILES_PER_GROUP ? TILES_PER_GROUP :
                               2 * CHANNELS_PER_GROUP ;
    const int n_output_iter = TILES_PER_GROUP / (filters_in_buf) + ((TILES_PER_GROUP % (filters_in_buf)) > 0);
    const int n_filters_per_buf_iter = filters_in_buf / N_ITER_PER_ONE_BUF_ELEM_FILTER;
    for (int filter_count = 0, accum_buf_beg=0; filter_count < TILES_PER_GROUP; 
         filter_count += filters_in_buf, accum_buf_beg++) {
        for (int m = 0; m < n_filters_per_buf_iter; ++m) {
            for (int n = 0; n < ACCUM_BUF_SIZE_IMG; ++n) {
    
                local_buf[(((filter_local_id % THREADS_PER_ONE_BUF_ELEM) % N_ITER_PER_ONE_BUF_ELEM_FILTER) + m  
                             * N_ITER_PER_ONE_BUF_ELEM_FILTER) * TILES_PER_GROUP * transformed_buf_size +
                             (((img_local_id % THREADS_PER_ONE_BUF_ELEM) / N_ITER_PER_ONE_BUF_ELEM_IMG) * 
                              ACCUM_BUF_SIZE_IMG + n) * transformed_buf_size + img_local_id / 
                             (img_group_size / transformed_buf_size)] = accum_buf[n_filters_per_buf_iter *   
                                                                                  accum_buf_beg + m][n];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    
        float transform_accum_buf[transformed_buf_size] = {0};
    
    
        for (int i = img_local_id; i < filters_in_buf * TILES_PER_GROUP; 
             i += TILES_PER_GROUP * CHANNELS_PER_GROUP) {
    
            for (int j = 0; j < transformed_buf_size; ++j) {
                transform_accum_buf[j] = local_buf[(i / TILES_PER_GROUP) * TILES_PER_GROUP *   
                                                    transformed_buf_size +(i % TILES_PER_GROUP) *  
                                                   transformed_buf_size + j];
            }
    
            float tmp[8];
            tmp[0] = transform_accum_buf[0] + transform_accum_buf[1] + transform_accum_buf[2];
            tmp[1] = transform_accum_buf[1] - transform_accum_buf[2] - transform_accum_buf[3];
            tmp[2] = transform_accum_buf[4] + transform_accum_buf[5] + transform_accum_buf[6];
            tmp[3] = transform_accum_buf[5] - transform_accum_buf[6] - transform_accum_buf[7];
            tmp[4] = transform_accum_buf[8] + transform_accum_buf[9] + transform_accum_buf[10];
            tmp[5] = transform_accum_buf[9] - transform_accum_buf[10] - transform_accum_buf[11];
            tmp[6] = transform_accum_buf[12] + transform_accum_buf[13] + transform_accum_buf[14];
            tmp[7] = transform_accum_buf[13] - transform_accum_buf[14] - transform_accum_buf[15];
    
            const int output_filter_num =  i / TILES_PER_GROUP + filter_count, 
                      output_tile_num = i % TILES_PER_GROUP;
    
            const int output_row = ((img_tile_offset_group + output_tile_num) / tiles_per_row) * output_H;
            const int output_col = ((img_tile_offset_group + output_tile_num) % tiles_per_row) * output_W;
            const int output_filter = filter_tile_offset_group + output_filter_num;
    
            if ((output_row < H) && (img_num < N_objects) && (output_filter < N_filters)) {
    
                const int output_offset = img_num * img_size * N_filters + output_filter * img_size +     
                                          output_row * W + output_col;
    
                output[output_offset] = tmp[0] + tmp[2] + tmp[4];
                output[output_offset + 1] = tmp[1] + tmp[3] + tmp[5];
                output[output_offset + W + 0] = tmp[2] - tmp[4] - tmp[6];
                output[output_offset + W + 1] = tmp[3] - tmp[5] - tmp[7];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}
