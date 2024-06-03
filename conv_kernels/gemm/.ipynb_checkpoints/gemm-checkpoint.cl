#define TileSizeK 7
#define TileSizeN 64
#define TileSizeM 64
#define ThreadBufSizeN 8
#define ThreadBufSizeM 8

kernel void GEMM(const int M, const int N, const int K, const int res_img_size,
                 const __global float* A, const __global float* B, __global float* C) {
                   
    const int GroupSizeM = get_local_size(0);   
    const int GroupSizeN = get_local_size(1);   

    const int ThreadLoadsA = ((TileSizeK * TileSizeM) / (GroupSizeM * GroupSizeN));
	const int ThreadLoadsB = ((TileSizeK * TileSizeN) / (GroupSizeM * GroupSizeN));
    const int ThreadIdM = get_local_id(0); 
    const int ThreadIdN = get_local_id(1); 
    const int offsetM = TileSizeM * get_group_id(0);
    const int offsetN = TileSizeN * get_group_id(1); 

    const int N_filters = M;
    const int N_objects = N / res_img_size;

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

    int numTiles = K / TileSizeK + ((K % TileSizeK) > 0);
    int tid = ThreadIdN * GroupSizeM + ThreadIdM;
    int row = tid;

    for (int t = 0; t < numTiles; t++) {

        for (int l = 0; l < ThreadLoadsA; l++) {
            int col = tid / TileSizeN + l;
            int tiledIndex = TileSizeK * t + col;
            if ((tiledIndex >= K) || ((offsetM + row) >= M)) {
                Alocal[col][row] = 0;
            } else {
                Alocal[col][row] = A[tiledIndex * M + offsetM + row];
            }
        }

        for (int l = 0; l < ThreadLoadsB; l++) {
            int col = tid / TileSizeN + l;
            int tiledIndex = TileSizeK * t + col;
            if ((tiledIndex >= K) || ((offsetN + row) >= N)) {
                Blocal[row][col] = 0;
            } else {
                Blocal[row][col] = B[tiledIndex * N + offsetN + row];
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
                C[cur_img_num * res_img_size * M + cur_filter_num * res_img_size + cur_img_el_num]
                = res[m][n];
            };
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
