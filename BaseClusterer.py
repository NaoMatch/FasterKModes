import os
import numpy as np
import subprocess
import ctypes
import datetime
from typing import Tuple
from numpy.typing import NDArray
import inspect

from src.generate_code import generate_get_nearest_dist_code, generate_dist_vec_code, generate_dist_mat_code
from src.generate_code import generate_naive_dist_mat_code, generate_naive_get_nearest_dist_code, generate_naive_dist_vec_code
from src.utils_kmodes import return_cat_argtypes

class BaseClusterer:
    def __init__(
            self, 
            n_clusters=8, 
            max_iter=100, 
            min_n_moves=0, 
            n_init=1, 
            random_state=None, 
            init="k-means++", 
            categorical_measure="hamming", 
            numerical_measure="euclidean", 
            n_jobs=None, 
            print_log=True, 
            gamma=1.0, 
            recompile=False, 
            use_simd=True, 
            max_tol=None): 
        
        self.__validate_hyper_parameters(
            n_clusters, 
            max_iter, 
            min_n_moves, 
            n_init, 
            random_state, 
            init,
            categorical_measure, 
            numerical_measure, 
            n_jobs, 
            print_log, 
            gamma, 
            recompile, 
            use_simd, 
            max_tol)
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.min_n_moves = min_n_moves
        self.n_init = n_init
        self.random_state = random_state
        self.init = init
        self.categorical_measure = categorical_measure
        self.numerical_measure = numerical_measure
        self.n_jobs = os.cpu_count() if n_jobs is None else n_jobs
        self.print_log = print_log
        self.recompile = recompile
        self.gamma = gamma
        self.use_simd = use_simd
        self.max_tol = max_tol
        self.is_loaded = False
        
        self._compile_lib(fn="common_funcs")
        self.lib = ctypes.CDLL("./src/common_funcs.so")
        self.is_fitted = False

    def __validate_hyper_parameters(
            self,
            n_clusters, 
            max_iter, 
            min_n_moves, 
            n_init, 
            random_state, 
            init,
            categorical_measure, 
            numerical_measure, 
            n_jobs, 
            print_log, 
            gamma, 
            recompile, 
            use_simd, 
            max_tol):
        # n_clusters: 整数でかつ2以上
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError(f"n_clusters must be an integer >= 2, but got {n_clusters} (type: {type(n_clusters)}).")
        
        # max_iter: 整数でかつ1以上
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError(f"max_iter must be an integer >= 1, but got {max_iter} (type: {type(max_iter)}).")
        
        # min_n_moves: 整数でかつ0以上
        if not isinstance(min_n_moves, int) or min_n_moves < 0:
            raise ValueError(f"min_n_moves must be an integer >= 0, but got {min_n_moves} (type: {type(min_n_moves)}).")
        
        # n_init: 整数でかつ1以上
        if not isinstance(n_init, int) or n_init < 1:
            raise ValueError(f"n_init must be an integer >= 1, but got {n_init} (type: {type(n_init)}).")
        
        # random_state: np.random.seed() が受け取れるかチェック
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(f"random_state must be either None or an integer, but got {random_state} (type: {type(random_state)}).")
        
        # init: 文字列の場合は VALID_INIT_METHODS、または callable で引数が ["X", "n_clusters"]
        VALID_INIT_METHODS = ["random", "k-means++", "huang", "cao"]
        if isinstance(init, str):
            if init not in VALID_INIT_METHODS:
                raise ValueError(f"init must be one of {VALID_INIT_METHODS}, but got '{init}'.")
        elif callable(init):
            sig = inspect.signature(init)
            if list(sig.parameters.keys()) != ["X", "n_clusters"]:
                raise ValueError(f"Custom init function must accept exactly two arguments: 'X' and 'n_clusters'. Got parameters: {list(sig.parameters.keys())}")
        else:
            raise ValueError(f"init must be a string or a callable function, but got {init} (type: {type(init)}).")
        
        # categorical_measure: 文字列なら "hamming"、または callable で引数が ["Xcat", "Ccat"]
        VALID_CATEGORICAL_MEASURES = ["hamming"]
        if isinstance(categorical_measure, str):
            if categorical_measure not in VALID_CATEGORICAL_MEASURES:
                raise ValueError(f"categorical_measure must be one of {VALID_CATEGORICAL_MEASURES}, but got '{categorical_measure}'.")
        elif callable(categorical_measure):
            sig = inspect.signature(categorical_measure)
            if list(sig.parameters.keys()) != ["Xcat", "Ccat"]:
                raise ValueError(f"Custom categorical_measure function must accept exactly two arguments: 'Xcat' and 'Ccat'. Got parameters: {list(sig.parameters.keys())}")
        else:
            raise ValueError(f"categorical_measure must be a string from {VALID_CATEGORICAL_MEASURES} or a callable function, but got {categorical_measure} (type: {type(categorical_measure)}).")
        
        # numerical_measure: 文字列なら "euclidean"、または callable で引数が ["Xnum", "Cnum"]
        VALID_NUMERICAL_MEASURES = ["euclidean"]
        if isinstance(numerical_measure, str):
            if numerical_measure not in VALID_NUMERICAL_MEASURES:
                raise ValueError(f"numerical_measure must be one of {VALID_NUMERICAL_MEASURES}, but got '{numerical_measure}'.")
        elif callable(numerical_measure):
            sig = inspect.signature(numerical_measure)
            if list(sig.parameters.keys()) != ["Xnum", "Cnum"]:
                raise ValueError(f"Custom numerical_measure function must accept exactly two arguments: 'Xnum' and 'Cnum'. Got parameters: {list(sig.parameters.keys())}")
        else:
            raise ValueError(f"numerical_measure must be a string from {VALID_NUMERICAL_MEASURES} or a callable function, but got {numerical_measure} (type: {type(numerical_measure)}).")
        
        # n_jobs: None または 1以上の整数
        if n_jobs is None:
            n_jobs = os.cpu_count()
        elif isinstance(n_jobs, int):
            if n_jobs < 1:
                raise ValueError(f"n_jobs must be an integer >= 1 or None, but got {n_jobs}.")
            n_jobs = min(n_jobs, os.cpu_count())
        else:
            raise ValueError(f"n_jobs must be an integer or None, but got {n_jobs} (type: {type(n_jobs)}).")
        
        # print_log: Boolean
        if not isinstance(print_log, bool):
            raise ValueError(f"print_log must be a boolean value, but got {print_log} (type: {type(print_log)}).")
        
        # gamma: 非負の int/float（None も可）
        if gamma is not None:
            if not isinstance(gamma, (int, float)) or gamma < 0:
                raise ValueError(f"gamma must be a non-negative int or float (or None), but got {gamma} (type: {type(gamma)}).")
        
        # recompile: Boolean
        if not isinstance(recompile, bool):
            raise ValueError(f"recompile must be a boolean value, but got {recompile} (type: {type(recompile)}).")
        
        # use_simd: Boolean
        if not isinstance(use_simd, bool):
            raise ValueError(f"use_simd must be a boolean value, but got {use_simd} (type: {type(use_simd)}).")
        
        # max_tol: None または非負の int/float
        if max_tol is not None:
            if not isinstance(max_tol, (int, float)) or max_tol < 0:
                raise ValueError(f"max_tol must be a non-negative float, but got {max_tol} (type: {type(max_tol)}).")
            
    def _compile_lib(self, fn: str):
        cmd = f"gcc -cpp -fPIC -fopenmp -march=native -shared ./src/{fn}.c -lm -o ./src/{fn}.so -O3 -Ofast"

        if (not os.path.exists(f"{fn}.so")) | (self.recompile):
            if self.print_log:
                print(cmd)
            result = subprocess.run(cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                print("Command failed with error:")
                print(result.stderr)


    def _generate_compile_load_get_nearest_dist(self):
        if not os.path.exists(self.fn_get_nearest_dist):
            if (self.n_cat_cols>=32) & (self.use_simd):
                generate_get_nearest_dist_code(self.input_cat_dtype, self.n_cat_cols_simd, self.simd_size, self.fn_get_nearest_dist)
            else:
                generate_naive_get_nearest_dist_code(self.input_cat_dtype, self.n_cat_cols, self.fn_get_nearest_dist)
        self._compile_lib(fn=self.fn_get_nearest_dist)
        self.lib_get_nearest_dist = ctypes.CDLL(f"./src/{self.fn_get_nearest_dist}.so")
        self.lib_get_nearest_dist.get_nearest_dist.argtypes = self.arg_dist_vec
        self.get_nearest_dist = self.lib_get_nearest_dist.get_nearest_dist

    def _generate_compile_load_dist_vec(self):
        if not os.path.exists(self.fn_dist_vec):
            if (self.n_cat_cols>=32) & (self.use_simd):
                generate_dist_vec_code(self.input_cat_dtype, self.n_cat_cols_simd, self.simd_size, self.fn_dist_vec)
            else:
                generate_naive_dist_vec_code(self.input_cat_dtype, self.n_cat_cols, self.fn_dist_vec)
        self._compile_lib(fn=self.fn_dist_vec)
        self.lib_dist_vec = ctypes.CDLL(f"./src/{self.fn_dist_vec}.so")
        self.lib_dist_vec.compute_dist_vec.argtypes = self.arg_dist_vec
        self.compute_dist_vec = self.lib_dist_vec.compute_dist_vec

    def _generate_compile_load_dist_mat(self):
        if not os.path.exists(self.fn_dist_mat):
            if (self.n_cat_cols>=32) & (self.use_simd):
                generate_dist_mat_code(self.input_cat_dtype, self.n_cat_cols_simd, self.simd_size, self.fn_dist_mat)
            else:
                generate_naive_dist_mat_code(self.input_cat_dtype, self.n_cat_cols, self.fn_dist_mat)
        self._compile_lib(fn=self.fn_dist_mat)
        self.lib_dist_mat = ctypes.CDLL(f"./src/{self.fn_dist_mat}.so")
        self.lib_dist_mat.compute_dist_mat.argtypes = self.arg_dist_mat
        self.compute_dist_mat = self.lib_dist_mat.compute_dist_mat

    def _compute_distance_vector(self, Xcat: np.ndarray, c: np.ndarray):
        N = len(Xcat)
        dist_vec = np.zeros(N, dtype=np.int32)
        self.compute_dist_vec(Xcat, N, self.n_cat_cols, c, dist_vec, self.n_jobs)
        return dist_vec

    def _compute_density_matrix(self, Xcat: np.ndarray):
        N = len(Xcat)
        max_val = self.max_vals.max().astype(np.int32)
        count_matrix = np.zeros((1, self.n_cat_cols, (max_val+1)), dtype=np.int32)

        centroids = np.array(np.zeros(N), dtype=np.int32)
        self.matrix_counter(Xcat, N, self.n_cat_cols, self.offset, centroids, count_matrix, 1, max_val, self.n_jobs)

        density_matrix = (count_matrix[0, :, :] / N).astype(np.float32)
        return density_matrix
