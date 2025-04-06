#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void get_nearest_dist(
    uint8_t *X, int32_t N, int32_t K,
    uint8_t *y,
    int32_t *d, int32_t n_jobs){

    int32_t tmp_dist = 0; 
    for (int32_t i=0; i<N; i++){
        int32_t *x_i = &X[i*K];
        for (int32_t j=0; j<K; j++){
            tmp_dist += x_i[j] != y[j];
        }
        d[i] = ((tmp_dist < d[i]) ? tmp_dist : d[i]);
    }
}

// void get_nearest_dist(
//     uint8_t *X, int32_t N, int32_t K,
//     uint8_t *y,
//     int32_t *d, int32_t n_jobs){
//     // 2行ずつ処理するためのループ（外側ループのアンローリング）
//     for (int32_t i = 0; i < N - 1; i += 2) {
//         int32_t tmp_dist1 = 0, tmp_dist2 = 0;
//         uint8_t *x1 = &X[i * K];
//         uint8_t *x2 = &X[(i + 1) * K];
//         int32_t j = 0;
        
//         // 内側ループを2回分ずつ処理（内側ループのアンローリング）
//         for (; j <= K - 2; j += 2) {
//             tmp_dist1 += (x1[j] != y[j]) + (x1[j + 1] != y[j + 1]);
//             tmp_dist2 += (x2[j] != y[j]) + (x2[j + 1] != y[j + 1]);
//         }
//         // K が奇数の場合、残りの1要素を処理
//         if (j < K) {
//             tmp_dist1 += (x1[j] != y[j]);
//             tmp_dist2 += (x2[j] != y[j]);
//         }
//         // 既存の d[i] と比較して、より小さいハミング距離をセット
//         d[i]   = (tmp_dist1 < d[i]) ? tmp_dist1 : d[i];
//         d[i+1] = (tmp_dist2 < d[i+1]) ? tmp_dist2 : d[i+1];
//     }
    
//     // N が奇数の場合、最後の1行を処理
//     if (N % 2 != 0) {
//         int32_t tmp_dist = 0;
//         uint8_t *x = &X[(N - 1) * K];
//         for (int32_t j = 0; j < K; j++) {
//             tmp_dist += (x[j] != y[j]);
//         }
//         d[N - 1] = (tmp_dist < d[N - 1]) ? tmp_dist : d[N - 1];
//     }
// }