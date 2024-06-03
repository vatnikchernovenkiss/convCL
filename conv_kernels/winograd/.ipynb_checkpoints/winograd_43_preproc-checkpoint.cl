#define  kern_H 3
#define  kern_W 3

#define  transformed_buf_H 6
#define  transformed_buf_W 6

kernel void winograd_transorm_weights(const __global float *filters, float __global *output) {

    const int group_id = get_group_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int filter_size = kern_H * kern_W;
    float filter_buf[kern_H * kern_W];
    const int filter_offset = (group_id * group_size + local_id) * filter_size;

    for (int i = 0; i < filter_size; ++ i) {
        filter_buf[(i % kern_H) * kern_W + i / kern_W] = filters[filter_offset + i];
    }

    float matrix_buf[transformed_buf_H * kern_W];

    matrix_buf[0] = 1.0/4 * filter_buf[0];
    matrix_buf[1] = 1.0/4 * filter_buf[3];
    matrix_buf[2] = 1.0/4 * filter_buf[6];

    matrix_buf[3] = -1.0/6 * (filter_buf[0] + filter_buf[1] + filter_buf[2]);
    matrix_buf[4] = -1.0/6 * (filter_buf[3] + filter_buf[4] + filter_buf[5]);
    matrix_buf[5] = -1.0/6 * (filter_buf[6] + filter_buf[7] + filter_buf[8]);

    matrix_buf[6] = 1.0/6 * (-filter_buf[0] + filter_buf[1] - filter_buf[2]);
    matrix_buf[7] = 1.0/6 * (-filter_buf[3] + filter_buf[4] - filter_buf[5]);
    matrix_buf[8] = 1.0/6 * (-filter_buf[6] + filter_buf[7] - filter_buf[8]);

    matrix_buf[9] = 1.0/24 * filter_buf[0] + 1.0/12 * filter_buf[1] + 1.0/6 * filter_buf[2];
    matrix_buf[10] = 1.0/24 * filter_buf[3] + 1.0/12 * filter_buf[4] + 1.0/6 * filter_buf[5];
    matrix_buf[11] = 1.0/24 * filter_buf[6] + 1.0/12 * filter_buf[7] + 1.0/6 * filter_buf[8];

    matrix_buf[12] = 1.0/24 * filter_buf[0] - 1.0/12 * filter_buf[1] + 1.0/6 * filter_buf[2];
    matrix_buf[13] = 1.0/24 * filter_buf[3] - 1.0/12 * filter_buf[4] + 1.0/6 * filter_buf[5];
    matrix_buf[14] = 1.0/24 * filter_buf[6] - 1.0/12 * filter_buf[7] + 1.0/6 * filter_buf[8];

    matrix_buf[15] = filter_buf[2];
    matrix_buf[16] = filter_buf[5];
    matrix_buf[17] = filter_buf[8];

    const int output_offset = (group_id * group_size + local_id) * transformed_buf_W * transformed_buf_H;
    output[output_offset + 0] = 1.0/4 * matrix_buf[0];
    output[output_offset + 1] = -1.0/6 * (matrix_buf[0] + matrix_buf[1] + matrix_buf[2]);
    output[output_offset + 2] = 1.0/6 * (-matrix_buf[0] + matrix_buf[1] - matrix_buf[2]);
    output[output_offset + 3] = 1.0/24 * matrix_buf[0] + 1.0/12 * matrix_buf[1] + 1.0/6 * matrix_buf[2];
    output[output_offset + 4] = 1.0/24 * matrix_buf[0] - 1.0/12 * matrix_buf[1] + 1.0/6 * matrix_buf[2];
    output[output_offset + 5] = matrix_buf[2];

    output[output_offset + 6] = 1.0/4 * matrix_buf[3];
    output[output_offset + 7] = -1.0/6 * (matrix_buf[3] + matrix_buf[4] + matrix_buf[5]);
    output[output_offset + 8] = 1.0/6 * (-matrix_buf[3] + matrix_buf[4] - matrix_buf[5]);
    output[output_offset + 9] = 1.0/24 * matrix_buf[3] + 1.0/12 * matrix_buf[4] + 1.0/6 * matrix_buf[5];
    output[output_offset + 10] = 1.0/24 * matrix_buf[3] - 1.0/12 * matrix_buf[4] + 1.0/6 * matrix_buf[5];
    output[output_offset + 11] = matrix_buf[5];

    output[output_offset + 12] = 1.0/4 * matrix_buf[6];
    output[output_offset + 13] = -1.0/6 * (matrix_buf[6] + matrix_buf[7] + matrix_buf[8]);
    output[output_offset + 14] = 1.0/6 * (-matrix_buf[6] + matrix_buf[7] - matrix_buf[8]);
    output[output_offset + 15] = 1.0/24 * matrix_buf[6] + 1.0/12 * matrix_buf[7] + 1.0/6 * matrix_buf[8];
    output[output_offset + 16] = 1.0/24 * matrix_buf[6] - 1.0/12 * matrix_buf[7] + 1.0/6 * matrix_buf[8];
    output[output_offset + 17] = matrix_buf[8];

    output[output_offset + 18] = 1.0/4 * matrix_buf[9];
    output[output_offset + 19] = -1.0/6 * (matrix_buf[9] + matrix_buf[10] + matrix_buf[11]);
    output[output_offset + 20] = 1.0/6 * (-matrix_buf[9] + matrix_buf[10] - matrix_buf[11]);
    output[output_offset + 21] = 1.0/24 * matrix_buf[9] + 1.0/12 * matrix_buf[10] + 1.0/6 * matrix_buf[11];
    output[output_offset + 22] = 1.0/24 * matrix_buf[9] - 1.0/12 * matrix_buf[10] + 1.0/6 * matrix_buf[11];
    output[output_offset + 23] = matrix_buf[11];

    output[output_offset + 24] = 1.0/4 * matrix_buf[12];
    output[output_offset + 25] = -1.0/6 * (matrix_buf[12] + matrix_buf[13] + matrix_buf[14]);
    output[output_offset + 26] = 1.0/6 * (-matrix_buf[12] + matrix_buf[13] - matrix_buf[14]);
    output[output_offset + 27] = 1.0/24 * matrix_buf[12] + 1.0/12 * matrix_buf[13] + 1.0/6 * matrix_buf[14];
    output[output_offset + 28] = 1.0/24 * matrix_buf[12] - 1.0/12 * matrix_buf[13] + 1.0/6 * matrix_buf[14];
    output[output_offset + 29] = matrix_buf[14];

    output[output_offset + 30] = 1.0/4 * matrix_buf[15];
    output[output_offset + 31] = -1.0/6 * (matrix_buf[15] + matrix_buf[16] + matrix_buf[17]);
    output[output_offset + 32] = 1.0/6 * (-matrix_buf[15] + matrix_buf[16] - matrix_buf[17]);
    output[output_offset + 33] = 1.0/24 * matrix_buf[15] + 1.0/12 * matrix_buf[16] + 1.0/6 * matrix_buf[17];
    output[output_offset + 34] = 1.0/24 * matrix_buf[15] - 1.0/12 * matrix_buf[16] + 1.0/6 * matrix_buf[17];
    output[output_offset + 35] = matrix_buf[17];
}