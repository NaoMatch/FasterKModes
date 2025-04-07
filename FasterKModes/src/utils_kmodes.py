import numpy as np
import ctypes
import subprocess
import tempfile
import os


def create_accumulated_operation(accm_op, variables, indent_level=1, op_format="{} = _mm256_or_si256({}, {});"):
    indent = " " * 4 * indent_level
    
    if len(variables) == 1:
        accm_op.append("\n")
        accm_op.append(variables[0])
        return 

    variables_next_ = []
    variables_ = variables[:-1] if len(variables) % 2 == 1 else variables
    for i in range(0, len(variables_), 2):
        accm_op.append(f"{indent}" + op_format.format(variables_[i], variables_[i], variables_[i+1]))
        variables_next_.append(f"{variables_[i]}")
        
    if len(variables) % 2 == 1:
        variables_next_.append(f"{variables[-1]}")

    accm_op.append("")
    create_accumulated_operation(accm_op, variables_next_, indent_level = indent_level, op_format=op_format)

def return_cat_argtypes(input_dtype):
    if input_dtype=="uint8":
        ARGTYPES_HAMM_DIST_VEC = [
                np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                ctypes.c_int32,
            ]

        ARGTYPES_HAMM_DIST_MAT = [
                np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2), 
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=2), 
                ctypes.c_int32,
            ]

        ARGTYPES_MATRIX_COUNTER = [
                np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=3), 
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
            ]

        ARGTYPES_SAMPLE_DENSITY = [
                np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
            ]
    else:
        ARGTYPES_HAMM_DIST_VEC = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                ctypes.c_int32,
            ]

        ARGTYPES_HAMM_DIST_MAT = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=2), 
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=2), 
                ctypes.c_int32,
            ]

        ARGTYPES_MATRIX_COUNTER = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=3), 
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
            ]

        ARGTYPES_SAMPLE_DENSITY = [
                np.ctypeslib.ndpointer(dtype=np.uint16, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
            ]

    ARGTYPES_DIST_X_DENS = [
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=2), 
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
        ]

    return ARGTYPES_HAMM_DIST_VEC, ARGTYPES_HAMM_DIST_MAT, ARGTYPES_MATRIX_COUNTER, ARGTYPES_SAMPLE_DENSITY, ARGTYPES_DIST_X_DENS

def return_num_argtypes(input_dtype):
    if input_dtype=="float":
        ARGTYPES_MATRIX_ACCUMULATOR = [
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.float32, ndim=2), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                ctypes.c_int32,
                ctypes.c_int32,
            ]
    else:
        ARGTYPES_MATRIX_ACCUMULATOR = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), 
                ctypes.c_int32,
                ctypes.c_int32,
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), 
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1), 
                ctypes.c_int32,
                ctypes.c_int32,
            ]
    return ARGTYPES_MATRIX_ACCUMULATOR

def can_use_immintrin():
    # テスト用のコード
    test_code = r'''
#include <immintrin.h>
int main(void) {
    __m256 x = _mm256_set1_ps(1.0f); // AVX 命令の一例
    return 0;
}
'''
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
        f.write(test_code.encode('utf-8'))
        f.flush()
        test_c_path = f.name

    test_exe_path = test_c_path + ".exe"
    try:
        # -mavx などのフラグが必要な場合もある (環境による)
        subprocess.check_call(
            ["gcc", "-o", test_exe_path, test_c_path, "-march=native"], 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        # 後始末（不要なら消してください）
        if os.path.exists(test_c_path):
            os.remove(test_c_path)
        if os.path.exists(test_exe_path):
            os.remove(test_exe_path)