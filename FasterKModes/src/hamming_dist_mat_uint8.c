#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void compute_dist_mat(
    uint8_t *X, int32_t N, int32_t K,
    uint8_t *Y, int32_t M,
    int32_t *D, int32_t n_jobs){

    uint8_t *x_ptr, *y_ptr;
    int32_t hamming_distance = 0;

    omp_set_num_threads(n_jobs);
    #pragma omp parallel for private(x_ptr, y_ptr, hamming_distance)
    for (int32_t i = 0; i < N; i++){
        x_ptr = &X[i*K];

        for (int32_t j = 0; j < M; j++){
            hamming_distance = 0;
            y_ptr = &Y[j*K];

            for (int32_t k = 0; k < K; k++){
                hamming_distance += (x_ptr[k] != y_ptr[k]);
            }

            D[i * M + j] = hamming_distance;
        }
    }
}
            