/*
 * gemmv2.cu
 * The Matrix M and N don't have to be square matrices.
 * The size of the shared memory is not determined until runtime.
*/

#include"gemm.hh"


__global__ void gemm_v2(float *M, float *N, float *P, int r, int s, int t, 
                        unsigned int Mds_sz, unsigned int Nds_sz) {
    
    extern __shared__ char Mds_Nds[];

    float *Mds = (float *) Mds_Nds;
    float *Nds = (float *) (Mds_Nds + Mds_sz);

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0;
    for(int phase = 0; phase < ceil(s/(float)blockDim.x); phase ++){
        if(Row < r && (phase * blockDim.x + tx) < s)
            Mds[ty * blockDim.x + tx] = M[Row * s + phase * blockDim.x + tx];
        else
            Mds[ty * blockDim.x + tx] = 0.0f;
        
        if(Col < t && (phase * blockDim.x + ty) < s)
            Nds[ty * blockDim.x + tx] = N[(phase * blockDim.x + ty) * t + Col];
        else
            Nds[ty * blockDim.x + tx] = 0.0f;

        __syncthreads();

        if(Row < r && Col < t){
            for(int k = 0; k < blockDim.x; k++){
                Pvalue += Mds[ty * blockDim.x + k] * Nds[k * blockDim.x + tx];
            }
        }
        __syncthreads();
    }

    if(Row < r && Col < t)
        P[Row * t + Col] = Pvalue;
}


void gemm_v2_invok(uint32_t n1, uint32_t n2, uint32_t n3,
                    float *a, float *b, float *c) {

    // Adjust this value to fit the shared memory size
    int local_tile_width = 32; 
    size_t size = local_tile_width * local_tile_width * 2 * sizeof(float);

    dim3 blockDim(local_tile_width, local_tile_width);
    dim3 gridDim(ceil((float)n3 / blockDim.x), ceil((float)n1 / blockDim.y));

    gemm_v2<<<gridDim, blockDim, size>>>(a, b, c, n1, n2, n3, 
                                        size / 2, size / 2);

}