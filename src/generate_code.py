from .utils_kmodes import create_accumulated_operation, can_use_immintrin

REGISTER_ACCUMURATOR = {
    "uint8":"""
    static inline uint64_t sum_of_uint8_in_m256i(__m256i v)
    {
        // まずは全要素との「絶対差」(=そのままの値) を取り、その合計を計算
        __m256i zero = _mm256_setzero_si256();
        // sad_epu8: v と zero の差分(=v自身)をバイトごとに絶対値をとり、合計する
        // ただし、内部的には 128 ビットごとに 2 つの 64 ビット値として合計するため、
        // 結果は [sum0, sum1, sum2, sum3] (64bit x 4) の形になる
        __m256i sum = _mm256_sad_epu8(v, zero);

        // sum（256 ビット）から、下位128ビット (sum0, sum1) と上位128ビット (sum2, sum3) を取り出す
        __m128i sum_lo = _mm256_castsi256_si128(sum);          // 下位128ビット
        __m128i sum_hi = _mm256_extracti128_si256(sum, 1);     // 上位128ビット

        // 下位128ビットと上位128ビットを64ビット加算すると、
        // まだ64ビット要素が2つ残る(__m128i)になる (例: (sum0+sum2, sum1+sum3))
        __m128i s = _mm_add_epi64(sum_lo, sum_hi);

        // s の中には 64 ビット要素が2つあるので、それらを最終的に足し合わせる
        uint64_t val0 = (uint64_t)_mm_cvtsi128_si64(s); // s の下位 64ビット
        // s の上位64ビットを下位に持ってくるため shuffle する
        s = _mm_shuffle_epi32(s, 0xEE);                 // 0xEE = (3,3,2,2) の意味
        uint64_t val1 = (uint64_t)_mm_cvtsi128_si64(s); // s の（元々の）上位64ビット

        return val0 + val1;
    }
    """, 
    "uint16": """
    static inline int32_t sum_of_uint16_in_m256i(__m256i v)
    {
        // 下位128ビットと上位128ビットをそれぞれ __m128i として取り出す
        __m128i lo_128 = _mm256_castsi256_si128(v);        
        __m128i hi_128 = _mm256_extracti128_si256(v, 1);   

        // (v[i], v[i+1]) を (v[i]*1 + v[i+1]*1) の 32ビットにまとめるための "1" を16ビットで詰めたもの
        __m128i ones = _mm_set1_epi16(1);

        //------------------------------------------------------------
        // 1) 下位128ビット(8要素のuint16_t)の合計
        //------------------------------------------------------------
        // madd_epi16:
        //   lo_128 の (16ビット×2要素) をまとめて「乗算→加算」し、32ビット要素に変換
        //   ここでは全て "×1" なので、単なるペア加算になります
        __m128i lo_pairs = _mm_madd_epi16(lo_128, ones);  
        // lo_pairs は 32ビット要素が 4 個 (8個のuint16_t→4個の32bit) になりました

        // 以下、水平加算を2回行うことで 4→2→1 個の32ビット要素に畳み込みます
        lo_pairs = _mm_hadd_epi32(lo_pairs, lo_pairs); // 4->2 要素
        lo_pairs = _mm_hadd_epi32(lo_pairs, lo_pairs); // 2->1 要素
        uint32_t s_lo = (uint32_t)_mm_cvtsi128_si32(lo_pairs);

        //------------------------------------------------------------
        // 2) 上位128ビット(8要素のuint16_t)の合計
        //------------------------------------------------------------
        __m128i hi_pairs = _mm_madd_epi16(hi_128, ones);  
        hi_pairs = _mm_hadd_epi32(hi_pairs, hi_pairs);
        hi_pairs = _mm_hadd_epi32(hi_pairs, hi_pairs);
        uint32_t s_hi = (uint32_t)_mm_cvtsi128_si32(hi_pairs);

        // 呼び出し側に返す
        return s_lo + s_hi;
    }
    """,
}

def generate_get_nearest_cat_dist_code(input_dtype, n_cols_simd, simd_size, fn_get_nearest_cat_dist):
    # Set1
    _mm256_set1 = "_mm256_set1_epi8" if input_dtype == "uint8" else "_mm256_set1_epi16"
    _mm256_add = "_mm256_add_epi8" if input_dtype == "uint8" else "_mm256_add_epi16"

    # 事前ロードするy_ptrの生成
    y_vecs = [f"y_vec_{i:0=3}" for i in range(0, n_cols_simd, simd_size)]

    # x_ptrのロード
    load_y_vecs = "\n        ".join(
        [f"__m256i {x} = _mm256_loadu_si256((__m256i*)&y[{i:=3}]);" for x, i in zip(y_vecs, range(0, n_cols_simd, simd_size))]
    )

    # ロードするx_ptrの生成
    x_vecs = [f"x_vec_{i:0=3}" for i in range(0, n_cols_simd, simd_size)]

    # x_ptrのロード
    load_x_vecs = [f"__m256i {x} = _mm256_loadu_si256((__m256i*)&x_ptr[{i:=3}]);" for x, i in zip(x_vecs, range(0, n_cols_simd, simd_size))]

    # 比較結果の生成
    comp_vecs = [f"__m256i compare_result_{i:0=3} = _mm256_cmpeq_epi8(x_vec_{i:0=3}, y_vec_{i:0=3});" for i in range(0, n_cols_simd, simd_size)]

    # 2個づつまとめて、load and compを実行する
    rem = len(x_vecs) % 2
    load_and_comp = "\n                \n            ".join([
        "\n            ".join([load_x_vecs[i], load_x_vecs[i+1], comp_vecs[i], comp_vecs[i+1]]) 
        for i in range(0, len(x_vecs)-rem, 2)
    ])
    if rem:
        load_and_comp += "\n        \n            "
        load_and_comp += "\n            ".join([load_x_vecs[-1], comp_vecs[-1]])
        

    # compとone の and積をとる
    result_vecs = [f"result_{i:0=3}"  for i in range(0, n_cols_simd, simd_size)]
    comp_and_ones = "\n            ".join(
        [f"__m256i result_{i:0=3} =  _mm256_and_si256(one, compare_result_{i:0=3});" for i in range(0, n_cols_simd, simd_size)]
    )

    accm_op = []
    create_accumulated_operation(accm_op, result_vecs, indent_level=0, op_format="{} = "+_mm256_add+"({}, {});")

    last_accm = accm_op[-1]
    accm_op = "\n            ".join(accm_op[:-2])

    reg_accm = REGISTER_ACCUMURATOR[input_dtype]

    base_code_for_get_nearest_hamming_dist = f"""

    {reg_accm}

    void get_nearest_cat_dist(
        {input_dtype}_t *X, int32_t N, int32_t K,
        {input_dtype}_t *y,
        int32_t *d, int32_t n_jobs){{

        {input_dtype}_t *x_ptr;
        uint64_t sum_result[2];

        __m256i one = {_mm256_set1}(1);
        {load_y_vecs}

        omp_set_num_threads(n_jobs);
        #pragma omp parallel for private(x_ptr, sum_result)
        for (int32_t i = 0; i < N; ++i) {{
            x_ptr = &X[i*K];

            int32_t hamming_simirality = 0;
            
            {load_and_comp}

            {comp_and_ones}

            {accm_op}

            hamming_simirality = sum_of_{input_dtype}_in_m256i({last_accm});

            for(int32_t k={n_cols_simd}; k<K; k++){{
                hamming_simirality += (x_ptr[k] == y[k]);
            }}
            d[i] = ((d[i] > K-hamming_simirality) ? K-hamming_simirality : d[i]);
        }}
    }}
    """

    with open(f"./src/{fn_get_nearest_cat_dist}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <stdint.h>\n")
        if can_use_immintrin(): f.write("#include <immintrin.h>\n")
        
        f.write(base_code_for_get_nearest_hamming_dist)

def generate_naive_get_nearest_cat_dist_code(input_dtype, n_cols, fn_get_nearest_cat_dist):
    ys = [f"y_{i:0=2}" for i in range(0, n_cols)]
    xs = [f"x_{i:0=2}" for i in range(0, n_cols)]
    cs = [f"comp_{i:0=2}" for i in range(0, n_cols)]

    load_y = "\n    ".join([f"{input_dtype}_t y_{i:0=2} = y[{i}];" for i in range(0, n_cols)])
    load_x = "\n        ".join([f"{input_dtype}_t x_{i:0=2} = X[i*K+{i}];" for i in range(0, n_cols)])
    comp_xy = "\n        ".join([f"int32_t comp_{i:0=2} = (x_{i:0=2} != y_{i:0=2});" for i in range(0, n_cols)])

    accm_op = []
    create_accumulated_operation(accm_op, cs, indent_level=2, op_format="{} = {} + {};")
    last_op = accm_op[-1]
    accm_op = "\n".join(accm_op[:-3])

    base_code = f"""
    void get_nearest_cat_dist(
        {input_dtype}_t *X, int32_t N, int32_t K,
        {input_dtype}_t *y,
        int32_t *d, int32_t n_jobs){{

        {load_y}

        for(int32_t i=0; i<N; i++){{
            {load_x}

            {comp_xy}

            {accm_op}

            d[i] = ((d[i] > {last_op}) ? {last_op} : d[i]);
        }}
    }}
                """
    with open(f"./src/{fn_get_nearest_cat_dist}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <stdint.h>\n")
        if can_use_immintrin(): f.write("#include <immintrin.h>\n")
        
        f.write(base_code)


def generate_dist_vec_code(input_dtype, n_cols_simd, simd_size, fn_dist_vec):
    # Set1
    _mm256_set1 = "_mm256_set1_epi8" if input_dtype == "uint8" else "_mm256_set1_epi16"
    _mm256_add = "_mm256_add_epi8" if input_dtype == "uint8" else "_mm256_add_epi16"

    # 事前ロードするy_ptrの生成
    y_vecs = [f"y_vec_{i:0=3}" for i in range(0, n_cols_simd, simd_size)]

    # x_ptrのロード
    load_y_vecs = "\n        ".join(
        [f"__m256i {x} = _mm256_loadu_si256((__m256i*)&y[{i:=3}]);" for x, i in zip(y_vecs, range(0, n_cols_simd, simd_size))]
    )

    # ロードするx_ptrの生成
    x_vecs = [f"x_vec_{i:0=3}" for i in range(0, n_cols_simd, simd_size)]

    # x_ptrのロード
    load_x_vecs = [f"__m256i {x} = _mm256_loadu_si256((__m256i*)&x_ptr[{i:=3}]);" for x, i in zip(x_vecs, range(0, n_cols_simd, simd_size))]

    # 比較結果の生成
    comp_vecs = [f"__m256i compare_result_{i:0=3} = _mm256_cmpeq_epi8(x_vec_{i:0=3}, y_vec_{i:0=3});" for i in range(0, n_cols_simd, simd_size)]

    # 2個づつまとめて、load and compを実行する
    rem = len(x_vecs) % 2
    load_and_comp = "\n                \n            ".join([
        "\n            ".join([load_x_vecs[i], load_x_vecs[i+1], comp_vecs[i], comp_vecs[i+1]]) 
        for i in range(0, len(x_vecs)-rem, 2)
    ])
    if rem:
        load_and_comp += "\n        \n            "
        load_and_comp += "\n            ".join([load_x_vecs[-1], comp_vecs[-1]])
        
    # compとone の and積をとる
    result_vecs = [f"result_{i:0=3}"  for i in range(0, n_cols_simd, simd_size)]
    comp_and_ones = "\n            ".join(
        [f"__m256i result_{i:0=3} =  _mm256_and_si256(one, compare_result_{i:0=3});" for i in range(0, n_cols_simd, simd_size)]
    )

    accm_op = []
    create_accumulated_operation(accm_op, result_vecs, indent_level=0, op_format="{} = "+_mm256_add+"({}, {});")

    last_accm = accm_op[-1]
    accm_op = "\n            ".join(accm_op[:-2])
    reg_accm = REGISTER_ACCUMURATOR[input_dtype]

    base_code_for_hamm_dist_vec = f"""

    {reg_accm}

    void compute_cat_dist_vec(
        {input_dtype}_t *X, int32_t N, int32_t K,
        {input_dtype}_t *y,
        int32_t *d, int32_t n_jobs){{

        {input_dtype}_t *x_ptr;
        uint64_t sum_result[2];

        __m256i one = {_mm256_set1}(1);
        {load_y_vecs}

        omp_set_num_threads(n_jobs);
        #pragma omp parallel for private(x_ptr, sum_result)
        for (int32_t i = 0; i < N; ++i) {{
            x_ptr = &X[i*K];

            int32_t hamming_simirality = 0;
            
            {load_and_comp}

            {comp_and_ones}

            {accm_op}

            hamming_simirality = sum_of_{input_dtype}_in_m256i({last_accm});

            for(int32_t k={n_cols_simd}; k<K; k++){{
                hamming_simirality += (x_ptr[k] == y[k]);
            }}
            d[i] = K-hamming_simirality;
        }}
    }}
    """
    with open(f"./src/{fn_dist_vec}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <stdint.h>\n")
        if can_use_immintrin(): f.write("#include <immintrin.h>\n")
        
        f.write(base_code_for_hamm_dist_vec)

def generate_naive_dist_vec_code(input_dtype, n_cols, fn_dist_vec):
    ys = [f"y_{i:0=2}" for i in range(0, n_cols)]
    xs = [f"x_{i:0=2}" for i in range(0, n_cols)]
    cs = [f"comp_{i:0=2}" for i in range(0, n_cols)]

    load_y = "\n    ".join([f"{input_dtype}_t y_{i:0=2} = y[{i}];" for i in range(0, n_cols)])
    load_x = "\n        ".join([f"{input_dtype}_t x_{i:0=2} = X[i*K+{i}];" for i in range(0, n_cols)])
    comp_xy = "\n        ".join([f"int32_t comp_{i:0=2} = (x_{i:0=2} != y_{i:0=2});" for i in range(0, n_cols)])

    accm_op = []
    create_accumulated_operation(accm_op, cs, indent_level=2, op_format="{} = {} + {};")
    last_op = accm_op[-1]
    accm_op = "\n".join(accm_op[:-3])

    base_code = f"""
    void compute_cat_dist_vec(
        {input_dtype}_t *X, int32_t N, int32_t K,
        {input_dtype}_t *y,
        int32_t *d, int32_t n_jobs){{

        {load_y}

        for(int32_t i=0; i<N; i++){{
            {load_x}

            {comp_xy}

            {accm_op}

            d[i] = {last_op};
        }}
    }}
                """
    with open(f"./src/{fn_dist_vec}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <stdint.h>\n")
        if can_use_immintrin(): f.write("#include <immintrin.h>\n")
        
        f.write(base_code)


def generate_dist_mat_code(input_dtype, n_cols_simd, simd_size, fn_dist_mat):
    # Set1
    _mm256_set1 = "_mm256_set1_epi8" if input_dtype == "uint8" else "_mm256_set1_epi16"
    _mm256_add = "_mm256_add_epi8" if input_dtype == "uint8" else "_mm256_add_epi16"


    # ロードするx_ptrの生成
    x_vecs = [f"x_vec_{i:0=3}" for i in range(0, n_cols_simd, simd_size)]

    # x_ptrのロード
    load_x_vecs = "\n        ".join([f"__m256i {x} = _mm256_loadu_si256((__m256i*)&x_ptr[{i:=3}]);" for x, i in zip(x_vecs, range(0, n_cols_simd, simd_size))])

    # 事前ロードするy_ptrの生成
    y_vecs = [f"y_vec_{i:0=3}" for i in range(0, n_cols_simd, simd_size)]

    # x_ptrのロード
    load_y_vecs = [f"__m256i {x} = _mm256_loadu_si256((__m256i*)&y_ptr[{i:=3}]);" for x, i in zip(y_vecs, range(0, n_cols_simd, simd_size))]

    # 比較結果の生成
    comp_vecs = [f"__m256i compare_result_{i:0=3} = _mm256_cmpeq_epi8(x_vec_{i:0=3}, y_vec_{i:0=3});" for i in range(0, n_cols_simd, simd_size)]

    # 2個づつまとめて、load and compを実行する
    rem = len(x_vecs) % 2
    load_and_comp = "\n                \n            ".join([
        "\n            ".join([load_y_vecs[i], load_y_vecs[i+1], comp_vecs[i], comp_vecs[i+1]]) 
        for i in range(0, len(x_vecs)-rem, 2)
    ])
    if rem:
        load_and_comp += "\n        \n            "
        load_and_comp += "\n            ".join([load_y_vecs[-1], comp_vecs[-1]])
        
    # compとone の and積をとる
    result_vecs = [f"result_{i:0=3}"  for i in range(0, n_cols_simd, simd_size)]
    comp_and_ones = "\n            ".join(
        [f"__m256i result_{i:0=3} =  _mm256_and_si256(one, compare_result_{i:0=3});" for i in range(0, n_cols_simd, simd_size)]
    )

    accm_op = []
    create_accumulated_operation(accm_op, result_vecs, indent_level=0, op_format="{} = "+_mm256_add+"({}, {});")

    last_accm = accm_op[-1]
    accm_op = "\n            ".join(accm_op[:-2])

    reg_accm = REGISTER_ACCUMURATOR[input_dtype]

    base_code_for_hamming_dist_matrix = f"""

    {reg_accm}

    void compute_cat_dist_mat(
        {input_dtype}_t *X, int32_t N, int32_t K,
        {input_dtype}_t *Y, int32_t M,
        int32_t *D, int32_t n_jobs){{

        {input_dtype}_t *x_ptr, *y_ptr;
        __m256i one = {_mm256_set1}(1);
        uint64_t sum_result[2];

        omp_set_num_threads(n_jobs);
        #pragma omp parallel for private(x_ptr, sum_result)
        for (int32_t i = 0; i < N; i++){{
            x_ptr = &X[i*K];
            {load_x_vecs}

            for (int32_t j = 0; j < M; j++){{
                y_ptr = &Y[j*K];
                int32_t hamming_simirality = 0;

                {load_and_comp}

                {comp_and_ones}

                {accm_op}

                hamming_simirality = sum_of_{input_dtype}_in_m256i({last_accm});

                for(int32_t k={n_cols_simd}; k<K; k++){{
                    hamming_simirality += (x_ptr[k] == y_ptr[k]);
                }}
                
                D[i * M + j] = K - hamming_simirality;
            }}
        }}
    }}
    """

    with open(f"./src/{fn_dist_mat}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <stdint.h>\n")
        if can_use_immintrin(): f.write("#include <immintrin.h>\n")
        
        f.write(base_code_for_hamming_dist_matrix)

def generate_naive_dist_mat_code(input_dtype, n_cols, fn_hamm_dist_mat):
    load_x = "\n        ".join([f"{input_dtype}_t x_{i:0=2} = x_ptr[{i}];" for i in range(0, n_cols)])
    ys = [f"y_{i:0=2}" for i in range(0, n_cols)]
    reses = [f"r_{i:0=2}" for i in range(0, n_cols)]

    load_y = [f"{input_dtype}_t y_{i:0=2} = y_ptr[{i}];" for i in range(0, n_cols)]
    comp_res_xy = [f"int32_t r_{i:0=2} = (x_{i:0=2} != y_{i:0=2});" for i in range(0, n_cols)]

    rem = len(ys) % 2
    load_and_comp = "\n                \n            ".join([
        "\n            ".join([load_y[i], load_y[i+1], comp_res_xy[i], comp_res_xy[i+1]]) 
        for i in range(0, len(ys)-rem, 2)
    ])
    if rem:
        load_and_comp += "\n        \n            "
        load_and_comp += "\n            ".join([load_y[-1], comp_res_xy[-1]])


    accm_op = []
    create_accumulated_operation(accm_op, reses, indent_level=0, op_format="{} = {} + {};")

    last_accm = accm_op[-1]
    accm_op = "\n            ".join(accm_op[:-2])

    base_code = f"""
void compute_cat_dist_mat(
    {input_dtype}_t *X, int32_t N, int32_t K,
    {input_dtype}_t *Y, int32_t M,
    int32_t *D, int32_t n_jobs){{

    {input_dtype}_t *x_ptr, *y_ptr;
    int32_t hamming_distance = 0;

    omp_set_num_threads(n_jobs);
    #pragma omp parallel for private(x_ptr, y_ptr, hamming_distance)
    for (int32_t i = 0; i < N; i++){{
        x_ptr = &X[i*K];
        {load_x}
        for (int32_t j = 0; j < M; j++){{
            hamming_distance = 0;
            y_ptr = &Y[j*K];

            {load_and_comp}

            {accm_op}
            
            D[i * M + j] = {last_accm};
        }}
    }}
}}
            """

    with open(f"./src/{fn_hamm_dist_mat}.c", "w") as f:
        f.write("#include <omp.h>\n")
        f.write("#include <stdio.h>\n")
        f.write("#include <stdlib.h>\n")
        f.write("#include <stdint.h>\n")
        if can_use_immintrin(): f.write("#include <immintrin.h>\n")
        
        f.write(base_code)
