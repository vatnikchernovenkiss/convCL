kernel void GEMM_preproc_data(const __global float *data, const int H, const int W, const int dst_H,
                              const int dst_W, const int pad_H, const int pad_W, const int N_objects,
                              const int N_channels, const int kern_H, const int kern_W, const int dilation_H, 
                              const int dilation_W, const int stride_H,  const int stride_W, 
                              float global *output) {

    const int one_channel_conv_num = dst_H * dst_W;
    const int one_conv_elements_num =  kern_H * kern_W * N_channels;
    const int input_img_size = H * W * N_channels;
    const int im2col_channel_size = kern_H * kern_W * one_channel_conv_num * N_objects;
    const int num_el = N_objects * N_channels * dst_H * dst_W;
    const int num_workers = get_global_size(0);
    const int global_id = get_global_id(0);

    for (int cur_el = global_id; cur_el < num_el; cur_el += num_workers) {
        int id = cur_el;
        const int img_num = id / (one_channel_conv_num * N_channels);
        id = id % (one_channel_conv_num * N_channels);
        const int channel_num = id / one_channel_conv_num;
        id = id % one_channel_conv_num;

        const int col_to_process = id % dst_W;
        const int row_to_process = id / dst_W;
        const int row_ind_input_init = row_to_process * stride_H - pad_H;
        const int col_ind_input_init = col_to_process * stride_W - pad_W;

        int im2col_offset = (one_channel_conv_num * img_num + id)  + channel_num * im2col_channel_size;
        const int input_img_offset = input_img_size * img_num + H * W * channel_num;
        for (int h = 0; h < kern_H; ++h) {
            const int input_row_offset = row_ind_input_init + h * dilation_H;
            for (int w = 0; w < kern_W; ++w) {
                const int input_col_offset = col_ind_input_init + w * dilation_W;
                float element_to_write = 0;
                if ((input_row_offset >= 0) && (input_row_offset < H) && (input_col_offset >= 0) &&
                    (input_col_offset < W)) {
                    element_to_write = data[input_img_offset +  input_row_offset * W  + input_col_offset];
                }
                output[im2col_offset] = element_to_write;
                im2col_offset += one_channel_conv_num * N_objects;
            }
        }
    }
}