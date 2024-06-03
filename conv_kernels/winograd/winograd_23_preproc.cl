#define  kern_H 3
#define  kern_W 3

#define  transformed_buf_H 4
#define  transformed_buf_W 4

kernel void winograd_transorm_weights( const __global float *filters, float __global *output) {

    const int group_id = get_group_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int filter_size = kern_H * kern_W;
    float filter_buf[kern_H * kern_W];
    const int filter_offset = (group_id * group_size + local_id) * filter_size;

    for (int i = 0; i < filter_size; ++ i) {
        filter_buf[i] = filters[filter_offset + i];
    }

    const int output_offset = (group_id * group_size + local_id) * transformed_buf_W * transformed_buf_H;
    output[output_offset + 0] = filter_buf[0];
    output[output_offset + 1] = 0.5 * (filter_buf[0] + filter_buf[2] + filter_buf[1]);
    output[output_offset + 2] = 0.5 * (filter_buf[0] + filter_buf[2] - filter_buf[1]);
    output[output_offset + 3] = filter_buf[2];
    output[output_offset + 4] = 0.5 * (filter_buf[0] + filter_buf[6] + filter_buf[3]);
    output[output_offset + 5] = 0.25 * ((filter_buf[0] + filter_buf[6] + filter_buf[3]) +
        (filter_buf[2] + filter_buf[8] + filter_buf[5]) + (filter_buf[1] + filter_buf[7] + filter_buf[4]));
    output[output_offset + 6] = 0.25 * ((filter_buf[0] + filter_buf[6] + filter_buf[3]) +
        (filter_buf[2] + filter_buf[8] + filter_buf[5]) - (filter_buf[1] + filter_buf[7] + filter_buf[4]));
    output[output_offset + 7] = 0.5 * (filter_buf[2] + filter_buf[8] + filter_buf[5]);
    output[output_offset + 8] = 0.5 * (filter_buf[0] + filter_buf[6] - filter_buf[3]);
    output[output_offset + 9] = 0.25 * ((filter_buf[0] + filter_buf[6] - filter_buf[3]) +
        (filter_buf[2] + filter_buf[8] - filter_buf[5]) + (filter_buf[1] + filter_buf[7] - filter_buf[4]));
    output[output_offset + 10] = 0.25 * ((filter_buf[0] + filter_buf[6] - filter_buf[3]) +
        (filter_buf[2] + filter_buf[8] - filter_buf[5]) - (filter_buf[1] + filter_buf[7] - filter_buf[4]));
    output[output_offset + 11] = 0.5 * (filter_buf[2] + filter_buf[8] - filter_buf[5]);
    output[output_offset + 12] = filter_buf[6];
    output[output_offset + 13] = 0.5 * (filter_buf[6] + filter_buf[8] + filter_buf[7]);
    output[output_offset + 14] = 0.5 * (filter_buf[6] + filter_buf[8] - filter_buf[7]);
    output[output_offset + 15] = filter_buf[8];
}