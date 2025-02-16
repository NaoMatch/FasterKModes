#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void matrix_accumulater_float(
    float *X, int32_t N, int32_t K,
    int32_t *I, 
    float *S, int32_t *C, int32_t n_clusters, int32_t n_jobs){

    omp_set_num_threads(n_jobs);
    #pragma omp parallel for reduction(+:S[:n_clusters*K]) reduction(+:C[:n_clusters])
    for(int32_t i=0; i<N; i++){
        int32_t cidx = I[i];
        for(int32_t k=0; k<K; k++){
            S[cidx*K+k] += X[i*K+k];
        }
        C[cidx] += 1;
    }
}

void matrix_accumulater_double(
    double *X, int32_t N, int32_t K,
    int32_t *I, 
    double *S, int32_t *C, int32_t n_clusters, int32_t n_jobs){

    omp_set_num_threads(n_jobs);
    #pragma omp parallel for reduction(+:S[:n_clusters*K]) reduction(+:C[:n_clusters])
    for(int32_t i=0; i<N; i++){
        int32_t cidx = I[i];
        for(int32_t k=0; k<K; k++){
            S[cidx*K+k] += X[i*K+k];
        }
        C[cidx] += 1;
    }
}



// void matrix_counter_uint8(
//     uint8_t *X, int32_t N, int32_t K,
//     int32_t *offset, 
//     int32_t *I, 
//     int32_t *C, int32_t n_clusters, int32_t n_categories, int32_t n_jobs){

//     int32_t shifted_val;
//     int32_t shift = K*(n_categories+1);
//     uint8_t *vals;

//     for(int32_t i=0; i<N; i++){
//         int32_t c=I[i];
//         vals = &X[i*K];
//         for(int32_t k=0; k<K; k++){
//             shifted_val = vals[k]+(n_categories+1)*k;
//             C[c*shift+shifted_val] += 1;
//         }
//     }
// }

// void matrix_counter_uint16(
//     uint16_t *X, int32_t N, int32_t K,
//     int32_t *offset, 
//     int32_t *I, 
//     int32_t *C, int32_t n_clusters, int32_t n_categories, int32_t n_jobs){

//     int32_t shifted_val;
//     int32_t shift = K*(n_categories+1);
//     uint16_t *vals;

//     omp_set_num_threads(n_jobs);
//     #pragma omp parallel for private(vals, shifted_val) reduction(+:C[:n_clusters*K*(n_categories+1)])
//     for(int32_t i=0; i<N; i++){
//         vals = &X[i*K];
//         for(int32_t k=0; k<K; k++){
//             shifted_val = vals[k]+offset[k];
//             C[shifted_val] += 1;
//         }
//     }
// }

void matrix_counter_uint8(
    uint8_t *X, int32_t N, int32_t K,
    int32_t *offset, 
    int32_t *I, 
    int32_t *C, int32_t n_clusters, int32_t n_categories, int32_t n_jobs){
    
    int32_t shifted_val;
    int32_t shift = K*(n_categories+1);
    uint8_t *vals;
    int32_t tmp[shift];

    for(int32_t c=0; c<n_clusters; c++){
        // zero clear
        for(int32_t s=0; s<shift; s++){
            tmp[s] = 0;
        }

        // クラスターごとにカウント
        omp_set_num_threads(n_jobs);
        #pragma omp parallel for private(vals, shifted_val) reduction(+:tmp[:shift])
        for(int32_t i=0; i<N; i++){
            if(I[i] == c){
                vals = &X[i*K];
                for(int32_t k=0; k<K; k++){
                    shifted_val = vals[k]+offset[k];
                    tmp[shifted_val] += 1;
                }
            }
        }

        // カウント結果を格納
        for(int32_t s=0; s<shift; s++){
            C[c*shift+s] = tmp[s];
        }
    }
}

void matrix_counter_uint16(
    uint16_t *X, int32_t N, int32_t K,
    int32_t *offset, 
    int32_t *L, 
    int32_t *C, int32_t n_clusters, int32_t n_categories, int32_t n_jobs){
    
    int32_t shifted_val;
    int32_t shift = K*(n_categories+1);
    uint16_t *vals;
    int32_t tmp[shift];

    for(int32_t c=0; c<n_clusters; c++){
        // zero clear
        for(int32_t s=0; s<shift; s++){
            tmp[s] = 0;
        }

        // クラスターごとにカウント
        omp_set_num_threads(n_jobs);
        #pragma omp parallel for private(vals, shifted_val) reduction(+:tmp[:shift])
        for(int32_t i=0; i<N; i++){
            if(L[i] == c){
                vals = &X[i*K];
                for(int32_t k=0; k<K; k++){
                    shifted_val = vals[k]+offset[k];
                    tmp[shifted_val] += 1;
                }
            }
        }

        // カウント結果を格納
        for(int32_t s=0; s<shift; s++){
            C[c*shift+s] = tmp[s];
        }
    }
}

void sample_density_uint8(
    uint8_t *X, int32_t N, int32_t K,
    float *D, 
    float *S, 
    int32_t *offset){

    int32_t shifted_val;
    uint8_t *vals;

    for(int32_t i=0; i<N; i++){
        float dens = 0;
        vals = &X[i*K];
        for(int32_t k=0; k<K; k++){
            shifted_val = vals[k] + offset[k];
            dens += D[shifted_val];
        }
        S[i] = dens / K;
    }
}

void sample_density_uint16(
    uint16_t *X, int32_t N, int32_t K,
    float *D, 
    float *S, 
    int32_t *offset){

    int32_t shifted_val;
    uint16_t *vals;

    for(int32_t i=0; i<N; i++){
        float dens = 0;
        vals = &X[i*K];
        for(int32_t k=0; k<K; k++){
            shifted_val = vals[k] + offset[k];
            dens += D[shifted_val];
        }
        S[i] = dens / K;
    }
}

void dist_x_dens(
    int32_t *D, int32_t N, int32_t n_clusters, 
    int32_t n_selected_cluster, 
    float *S, float*T){

    float dense;
    for(int32_t i=0; i<N; i++){
        dense = S[i];
        T[n_selected_cluster*N + i] = D[n_selected_cluster*N + i] * dense;
    }
}
