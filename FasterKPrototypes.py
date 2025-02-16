import os
import numpy as np
import subprocess
import ctypes
import datetime
from copy import deepcopy
from scipy.linalg.blas import sgemm, sgemv, dgemm, dgemv
from typing import Tuple
from numpy.typing import NDArray
import inspect

from src.generate_code import generate_get_nearest_dist_code, generate_dist_vec_code, generate_dist_mat_code
from src.generate_code import generate_naive_dist_mat_code, generate_naive_get_nearest_dist_code, generate_naive_dist_vec_code
from src.utils_kmodes import return_cat_argtypes, return_num_argtypes

Array1D = NDArray[Tuple[int]]  # 任意の長さの1次元配列

class FasterKPrototypes:
    def __init__(
            self, 
            n_clusters=8, max_iter=100, min_n_moves=0, n_init=1, random_state=None, init="k-means++", categorical_measure="hamming", numerical_measure="euclidean", 
            n_jobs=None, print_log=True, gamma=1.0, recompile=False, use_simd=True, max_tol=None):
        
        self.__validate_params(
            n_clusters, max_iter, min_n_moves, n_init, random_state, init,
            categorical_measure, numerical_measure, n_jobs, print_log, gamma, recompile, use_simd)
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
        
        self.__compile_lib(fn="common_funcs")
        self.lib = ctypes.CDLL("./src/common_funcs.so")

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.arg_dist_vec, self.arg_dist_mat, self.arg_matrix_counter, \
            self.arg_sample_density, self.arg_dist_x_dens = return_cat_argtypes(self.input_cat_dtype)
        self.arg_matrix_accumulator = return_num_argtypes(self.input_num_dtype)

        self.__compile_lib(fn="common_funcs")
        self.lib = ctypes.CDLL("./src/common_funcs.so")
        self.__generate_compile_load_get_nearest_dist()
        self.__generate_compile_load_dist_vec()
        self.__generate_compile_load_dist_mat()
        if self.input_num_dtype == "double":
            self.gemm = dgemm
            self.gemv = dgemv
        else:
            self.gemm = sgemm
            self.gemv = sgemv

    def __getstate__(self):
        state = self.__dict__.copy()

        del state["arg_dist_mat"]
        del state["arg_dist_vec"]
        del state["arg_dist_x_dens"]
        del state["arg_matrix_accumulator"]
        del state["arg_matrix_counter"]
        del state["arg_sample_density"]
        del state["compute_dist_mat"]
        del state["compute_dist_vec"]
        del state["get_nearest_dist"]
        del state["lib"]
        del state["lib_dist_mat"]
        del state["lib_dist_vec"]
        del state["lib_get_nearest_dist"]
        del state["matrix_accumulater"]
        del state["matrix_counter"]
        del state["sample_density"]
        del state["gemm"]
        del state["gemv"]
        return state

    def __validate_params(
        self, n_clusters, max_iter, min_n_moves, n_init, random_state, init,
        categorical_measure, numerical_measure, n_jobs, print_log, gamma, recompile, use_simd):
        # n_clusters
        if not isinstance(n_clusters, int) or n_clusters < 0:
            raise ValueError("n_clusters must be a non-negative integer.")

        # max_iter
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer.")

        # min_n_moves
        if not isinstance(min_n_moves, int) or min_n_moves < 0:
            raise ValueError("min_n_moves must be a non-negative integer.")

        # n_init
        if not isinstance(n_init, int) or n_init < 0:
            raise ValueError("n_init must be a non-negative integer.")

        # random_state
        try:
            np.random.seed(random_state)
        except Exception as e:
            raise ValueError(f"random_state must be an int, None, or a compatible type for np.random.seed(). Error: {e}")

        # init
        VALID_INIT_METHODS = ["random", "k-means++", "huang", "cao"]
        if isinstance(init, str):
            if init not in VALID_INIT_METHODS:
                raise ValueError(f"init must be one of {VALID_INIT_METHODS}, but got '{init}'.")
        elif callable(init):
            sig = inspect.signature(init)
            if list(sig.parameters.keys()) != ["X", "n_clusters"]:
                raise ValueError("Custom init function must accept exactly two arguments: 'X' and 'n_clusters'.")
        else:
            raise ValueError("init must be a string or a callable function.")

        # categorical_measure
        VALID_CATEGORICAL_MEASURES = ["hamming"]
        if isinstance(categorical_measure, str):
            if categorical_measure not in VALID_CATEGORICAL_MEASURES:
                raise ValueError(f"categorical_measure must be one of {VALID_CATEGORICAL_MEASURES}, but got '{categorical_measure}'.")
        elif callable(categorical_measure):
            sig = inspect.signature(categorical_measure)
            if list(sig.parameters.keys()) != ["Xcat", "Xnum", "C"]:
                raise ValueError("Custom categorical_measure function must accept exactly two arguments: 'Xcat', 'Xnum' and 'C'.")
        else:
            raise ValueError(f"categorical_measure must be a string from {VALID_CATEGORICAL_MEASURES} or a callable function.")

        # numerical_measure
        VALID_NUMERICAL_MEASURES = ["euclidean"]
        if isinstance(numerical_measure, str):
            if numerical_measure not in VALID_NUMERICAL_MEASURES:
                raise ValueError(f"numerical_measure must be one of {VALID_NUMERICAL_MEASURES}, but got '{numerical_measure}'.")
        elif callable(numerical_measure):
            sig = inspect.signature(numerical_measure)
            if list(sig.parameters.keys()) != ["X", "C"]:
                raise ValueError("Custom numerical_measure function must accept exactly two arguments: 'X' and 'C'.")
        else:
            raise ValueError(f"numerical_measure must be a string from {VALID_NUMERICAL_MEASURES} or a callable function.")

        # n_jobs
        if n_jobs is None:
            n_jobs = os.cpu_count()
        elif isinstance(n_jobs, int):
            if n_jobs < 0:
                raise ValueError("n_jobs must be a non-negative integer or None.")
            n_jobs = min(n_jobs, os.cpu_count())
        else:
            raise ValueError("n_jobs must be an integer or None.")

        # print_log
        if not isinstance(print_log, bool):
            raise ValueError("print_log must be a boolean value.")

        # gamma
        if gamma is not None:
            if not isinstance(gamma, (int, float)) or gamma < 0:
                raise ValueError("gamma must be a non-negative int, float, or None.")

        # recompile
        if not isinstance(recompile, bool):
            raise ValueError("recompile must be a boolean value.")

        # use_simd
        if not isinstance(use_simd, bool):
            raise ValueError("use_simd must be a boolean value.")

    def __compile_lib(self, fn: str):
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

    def __generate_compile_load_get_nearest_dist(self):
        if not os.path.exists(self.fn_get_nearest_dist):
            if (self.n_cat_cols>=32) & (self.use_simd):
                generate_get_nearest_dist_code(self.input_cat_dtype, self.n_cat_cols_simd, self.simd_size, self.fn_get_nearest_dist)
            else:
                generate_naive_get_nearest_dist_code(self.input_cat_dtype, self.n_cat_cols, self.fn_get_nearest_dist)
        self.__compile_lib(fn=self.fn_get_nearest_dist)
        self.lib_get_nearest_dist = ctypes.CDLL(f"./src/{self.fn_get_nearest_dist}.so")
        self.lib_get_nearest_dist.get_nearest_dist.argtypes = self.arg_dist_vec
        self.get_nearest_dist = self.lib_get_nearest_dist.get_nearest_dist

    def __generate_compile_load_dist_vec(self):
        if not os.path.exists(self.fn_dist_vec):
            if (self.n_cat_cols>=32) & (self.use_simd):
                generate_dist_vec_code(self.input_cat_dtype, self.n_cat_cols_simd, self.simd_size, self.fn_dist_vec)
            else:
                generate_naive_dist_vec_code(self.input_cat_dtype, self.n_cat_cols, self.fn_dist_vec)
        self.__compile_lib(fn=self.fn_dist_vec)
        self.lib_dist_vec = ctypes.CDLL(f"./src/{self.fn_dist_vec}.so")
        self.lib_dist_vec.compute_dist_vec.argtypes = self.arg_dist_vec
        self.compute_dist_vec = self.lib_dist_vec.compute_dist_vec

    def __generate_compile_load_dist_mat(self):
        if not os.path.exists(self.fn_dist_mat):
            if (self.n_cat_cols>=32) & (self.use_simd):
                generate_dist_mat_code(self.input_cat_dtype, self.n_cat_cols_simd, self.simd_size, self.fn_dist_mat)
            else:
                generate_naive_dist_mat_code(self.input_cat_dtype, self.n_cat_cols, self.fn_dist_mat)
        self.__compile_lib(fn=self.fn_dist_mat)
        self.lib_dist_mat = ctypes.CDLL(f"./src/{self.fn_dist_mat}.so")
        self.lib_dist_mat.compute_dist_mat.argtypes = self.arg_dist_mat
        self.compute_dist_mat = self.lib_dist_mat.compute_dist_mat

    def __compute_distance_matrix(self, Xcat: np.ndarray, Xnum: np.ndarray):
        N = len(Xcat)
        cat_dist_mat = np.zeros((N, self.n_clusters), dtype=np.int32, order="C")
        self.compute_dist_mat(Xcat, N, self.n_cat_cols, self.Ccat, self.n_clusters, cat_dist_mat, self.n_jobs)

        if self.__is_train:
            num_dist_mat_ = (self.Xnum_sqsum + self.Cnum_sqsum).astype(self.ftype)
        else:
            Xnum_sqsum = (Xnum*Xnum).sum(axis=1).reshape((-1, 1))
            num_dist_mat_ = (Xnum_sqsum + self.Cnum_sqsum).astype(self.ftype)

        num_dist_mat = self.gemm(alpha=-2.0, a=Xnum, b=self.Cnum, c=num_dist_mat_, beta=1.0, trans_b=True)

        return num_dist_mat + self.gamma*cat_dist_mat

    def __compute_distance_vector(self, Xcat: np.ndarray, c: np.ndarray):
        N = len(Xcat)
        dist_vec = np.zeros(N, dtype=np.int32)
        self.compute_dist_vec(Xcat, N, self.n_cat_cols, c, dist_vec, self.n_jobs)
        return dist_vec

    def __compute_density_matrix(self, Xcat: np.ndarray):
        N = len(Xcat)
        max_val = self.max_vals.max()
        count_matrix = np.zeros((1, self.n_cat_cols, (max_val+1)), dtype=np.int32)

        centroids = np.array(np.zeros(N), dtype=np.int32)
        self.matrix_counter(Xcat, N, self.n_cat_cols, self.offset, centroids, count_matrix, 1, max_val, self.n_jobs)

        density_matrix = (count_matrix[0, :, :] / N).astype(np.float32)
        return density_matrix
    
    def __predict_centroid_indices(self, Xcat: np.ndarray, Xnum: np.ndarray, return_distance=False):
        dist_mat = self.__compute_distance_matrix(Xcat, Xnum)
        nearest_centroid_indices = dist_mat.argmin(axis=1).astype(np.int32)
        if return_distance:
            return nearest_centroid_indices, dist_mat.min(axis=1)
        return nearest_centroid_indices

    def __select_initial_centroids(self, Xcat: np.ndarray, Xnum: np.ndarray, init_Ccat: np.ndarray, init_Cnum: np.ndarray):
        N = len(Xcat)
        if init_Ccat is not None:
            self.Ccat = init_Ccat.astype(self.itype)
            self.Cnum = init_Cnum.astype(self.ftype)
        elif self.init == "random":
            indices = N
            shuffled_indices = np.random.permutation(indices)
            centroid_indices = shuffled_indices[:self.n_clusters]
            self.Cnum = Xnum[centroid_indices,:].astype(self.ftype)

        elif self.init == "k-means++":
            centroid_indices = []

            # Define 
            cat_dist_mat = np.zeros((N, self.n_clusters), dtype=np.int32)
            num_dist_mat = np.zeros((N, self.n_clusters), dtype=np.float32)

            cat_dist_vec = np.zeros(N, dtype=np.int32)
            num_dist_vec = np.zeros(N, dtype=np.float32)

            # Random Selected Centroid
            rnd_idx = np.random.randint(0, N)
            centroid_indices.append(rnd_idx)
            cat_centroid_vec = Xcat[rnd_idx,:]
            num_centroid_vec = Xnum[rnd_idx,:]

            self.compute_dist_vec(Xcat, N, self.n_cat_cols, cat_centroid_vec, cat_dist_vec, self.n_jobs)
            cat_dist_mat[:,0] = cat_dist_vec[:]

            num_dist_vec = self.Xnum_sqsum + (num_centroid_vec*num_centroid_vec).sum()
            num_dist_vec = self.gemv(alpha=-2.0, a=Xnum, x=num_centroid_vec, y=num_dist_vec, beta=1.0)
            num_dist_mat[:,0] = num_dist_vec[:,0].clip(min=0)

            # Select 2nd - N-th Centroids
            for c in range(1, self.n_clusters):
                min_dist_vec = (self.gamma*cat_dist_mat[:,0:c] + num_dist_mat[:,0:c]).min(axis=1)
                weights = min_dist_vec / np.sum(min_dist_vec)
                rnd_idx = np.random.choice(N, p=weights)
                centroid_indices.append(rnd_idx)
                cat_centroid_vec = Xcat[rnd_idx,:]
                num_centroid_vec = Xnum[rnd_idx,:]

                self.compute_dist_vec(Xcat, N, self.n_cat_cols, cat_centroid_vec, cat_dist_vec, self.n_jobs)
                cat_dist_mat[:,c] = cat_dist_vec[:]

                num_dist_vec = self.Xnum_sqsum + (num_centroid_vec*num_centroid_vec).sum()
                num_dist_vec = self.gemv(alpha=-2.0, a=Xnum, x=num_centroid_vec, y=num_dist_vec, beta=1.0)
                num_dist_mat[:,c] = num_dist_vec[:,0].clip(min=0)
            self.Cnum = Xnum[centroid_indices,:].astype(self.ftype)

        elif self.init == "huang":
            self.Ccat = np.zeros((self.n_clusters, self.n_cat_cols), dtype=Xcat.dtype)

            # ランダムに選択されたセントロイドを生成
            for k in range(self.n_cat_cols):
                X_k = Xcat[:,k]
                x_k = np.random.choice(X_k, size=self.n_clusters)
                self.Ccat[:,k] = x_k

            # セントロイドを修正
            centroid_indices = []
            dist_mat = np.zeros((N, self.n_clusters), dtype=np.int32)
            self.compute_dist_mat(Xcat, N, self.n_cat_cols, self.Ccat, self.n_clusters, dist_mat, self.n_jobs)
            for k in range(self.n_clusters):
                dist_vec = dist_mat[:,k]
                ranking = np.argsort(dist_vec)
                for rnk in ranking:
                    if rnk not in centroid_indices:
                        centroid_indices.append(rnk)
                        break

            self.Cnum = (np.random.randn(self.n_clusters, self.n_num_cols) * self.Xnum_std).astype(self.ftype)

        elif self.init == "cao":
            centroid_indices = []

            # 1番目のセントロイドを選択
            density_matrix = self.__compute_density_matrix(Xcat)
            sample_densities = np.zeros((N,1), dtype=np.float32)
            self.sample_density(Xcat, N, self.n_cat_cols, density_matrix, sample_densities, self.offset)
            centroid_idx = np.argmax(sample_densities[:,0])
            centroid_indices.append(centroid_idx)
            distance_matrix = np.asfortranarray(np.zeros((N, self.n_clusters), dtype=np.int32))

            # 2個目のセントロイドを選択
            c_vector = Xcat[centroid_indices[0],:]
            dist_vec = self.__compute_distance_vector(Xcat, c_vector)
            distance_matrix[:,0] = dist_vec
            centroid_idx = np.argmax(dist_vec * sample_densities[:,0])
            centroid_indices.append(centroid_idx)
            c_vector = Xcat[centroid_indices[1],:]
            dist_vec = self.__compute_distance_vector(Xcat, c_vector)
            distance_matrix[:,1] = dist_vec

            # 3個目以降のセントロイドを選択
            tmp_dens_mat = np.asfortranarray(np.zeros((N, self.n_clusters), dtype=np.float32))
            tmp_dens_mat[:,0] = distance_matrix[:,0] * sample_densities[:,0]
            for c in range(2, self.n_clusters):
                self.lib.dist_x_dens(distance_matrix, N, self.n_clusters, c-1, sample_densities, tmp_dens_mat)
                centroid_idx = np.argmax(np.min(tmp_dens_mat[:,:c], axis=1))
                centroid_indices.append(centroid_idx)
                c_vector = Xcat[centroid_idx,:]
                dist_vec = self.__compute_distance_vector(Xcat, c_vector)
                distance_matrix[:,c] = dist_vec

            self.Cnum = (np.random.randn(self.n_clusters, self.n_num_cols) * self.Xnum_std).astype(self.ftype)
        elif callable(self.init):
            self.Ccat, self.Cnum = self.init(Xcat, Xnum, self.n_clusters)
            self.__check_custom_init_method_output(Xcat, Xnum)
        else:
            raise NotImplementedError
        self.Ccat = Xcat[centroid_indices,:].astype(self.itype)

        self.Cnum_sqsum = np.sum(self.Cnum*self.Cnum, axis=1).reshape((1, -1))

    def __check_custom_init_method_output(self, Xcat, Xnum):
        # Validate self.Ccat
        if not isinstance(self.Ccat, np.ndarray):
            raise ValueError("self.Ccat must be a numpy ndarray.")
        if self.Ccat.ndim != 2:
            raise ValueError("self.Ccat must be a 2-dimensional array.")
        if self.Ccat.shape[1] != len(self.cat_feat_idxs):
            raise ValueError("self.Ccat must have the same number of columns as the categorical features.")
        if not np.issubdtype(self.Ccat.dtype, np.integer):
            raise ValueError(f"self.C must contain integer values, {np.uint8} or {np.uint16}, not {self.Ccat.dtype}.")
        if np.any(self.Ccat < 0) or np.any(self.Ccat >= Xcat[:, self.cat_feat_idxs].max() + 1):
            raise ValueError("self.Ccat contains invalid values for categorical features.")

        # Validate self.Cnum
        if not isinstance(self.Cnum, np.ndarray):
            raise ValueError("self.Cnum must be a numpy ndarray.")
        if self.Cnum.ndim != 2:
            raise ValueError("self.Cnum must be a 2-dimensional array.")
        if self.Cnum.shape[1] != len(self.num_feat_idxs):
            raise ValueError("self.Cnum must have the same number of columns as the numerical features.")
        if not np.issubdtype(self.Cnum.dtype, np.floating):
            raise ValueError(f"self.C must contain float values, {np.float32} or {np.float64}, not {self.Cnum.dtype}.")

        # Check consistency of self.Ccat and self.Cnum with n_clusters
        if self.Ccat.shape[0] != self.Cnum.shape[0]:
            raise ValueError("self.Ccat and self.Cnum must have the same number of rows (centroids).")
        if self.Ccat.shape[0] != self.n_clusters:
            raise ValueError("The number of rows in self.Ccat and self.Cnum must match n_clusters.")

    def __update_centroids(self, Xcat: np.ndarray, Xnum: np.ndarray, old_centroid_inds: Array1D, new_centroid_inds: Array1D):
        N = len(Xcat)

        # Xcat側更新
        max_val = self.max_vals.max().astype(np.int32)
        s = datetime.datetime.now()
        if old_centroid_inds[0] == -1:
            # 初回の処理
            self.count_matrix = np.zeros((self.n_clusters, self.n_cat_cols, (max_val+1)), dtype=np.int32) # Centroids x Category x Feature
        else:
            self.count_matrix = self.count_matrix * 0

        self.matrix_counter(Xcat, N, self.n_cat_cols, self.offset, new_centroid_inds, self.count_matrix, self.n_clusters, max_val, self.n_jobs)

        # Centroidの更新
        is_exist = [False] * self.n_clusters
        for c_ind in new_centroid_inds:
            is_exist[c_ind] = True
        for c_ind in range(self.n_clusters):
            if is_exist[c_ind]:
                self.Ccat[c_ind,:] = self.count_matrix[c_ind,:,:].argmax(axis=1)
            else: # Empty Cluster
                self.Ccat[c_ind,:] = np.random.randint(0, max_val+1, self.n_cat_cols)
        e = datetime.datetime.now()
        time_cat = (e-s).total_seconds()

        # Xnum側更新
        s = datetime.datetime.now()
        if old_centroid_inds[0] == -1:
            self.Cnum_ = np.zeros((self.n_clusters, self.n_num_cols), dtype=self.ftype)
            self.counter = np.zeros(self.n_clusters, dtype=np.int32)
        else:
            self.Cnum_ = self.Cnum_*0.0
            self.counter = self.counter*0

        self.matrix_accumulater(Xnum, N, self.n_num_cols, new_centroid_inds, self.Cnum_, self.counter, self.n_clusters, self.n_jobs)

        for c_ind in range(self.n_clusters):
            count = self.counter[c_ind]
            if count>0:
                self.Cnum[c_ind,:] = self.Cnum_[c_ind,:] / count
            else:
                self.Cnum[c_ind,:] = np.random.randn(self.n_num_cols)

        self.Cnum_sqsum = np.sum(self.Cnum*self.Cnum, axis=1).reshape((1, -1))
        e = datetime.datetime.now()
        time_num = (e-s).total_seconds()
        # print("                                                                    time cat num", time_cat, time_num)

    def __validate_train_X(self, X: np.ndarray, categorical_feature_indices: list, init_Ccat: np.ndarray, init_Cnum: np.ndarray):
        """
        Validate input data and categorical feature indices for k-prototypes clustering.
        
        Parameters:
            X (np.ndarray): The input data array.
            indices (list or np.ndarray): Indices of categorical features.
        
        Raises:
            ValueError: If any validation check fails.
        """

        self.n_cols = X.shape[1]
        self.cat_feat_idxs = sorted(deepcopy(categorical_feature_indices))
        self.num_feat_idxs = [i for i in range(self.n_cols) if i not in self.cat_feat_idxs]

        # Check if X is a numpy array
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy ndarray.")
        
        # Check if X is a 2-dimensional array
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")
        
        # Check if the data type of X is one of the allowed types
        allowed_dtypes = [np.uint8, np.uint16, np.float32, np.float64]
        if X.dtype not in allowed_dtypes:
            raise ValueError("X must have a dtype of uint8, uint16, float32, or float64.")
        
        # If dtype is float, check the conditions for categorical features
        if np.issubdtype(X.dtype, np.floating):
            if np.any(X[:, self.cat_feat_idxs] < 0):
                raise ValueError("Categorical features must be non-negative when dtype is float.")
            max_value = np.iinfo(np.uint16).max
            if np.any(X[:, self.cat_feat_idxs] > max_value):
                raise ValueError(f"Categorical features must not exceed {max_value} when dtype is float.")
        
        # Check if indices are within valid range
        if not all(0 <= idx < X.shape[1] for idx in self.cat_feat_idxs):
            raise ValueError("All indices must be within the range of X's column indices.")
        
        # Check if indices contain negative values
        if any(idx < 0 for idx in self.cat_feat_idxs):
            raise ValueError("Indices must not contain negative values.")
        
        # Check if X is in C order
        if not X.flags['C_CONTIGUOUS']:
            raise ValueError("X must have order='C'.")

        # Check the number of unique rows in X
        n_unique_rows = np.unique(X, axis=0).shape[0]
        if n_unique_rows < self.n_clusters:
            raise ValueError(
                f"The number of unique rows in X ({n_unique_rows}) must be greater than or equal to n_clusters ({self.n_clusters}).")
        
        # Decision logic for clustering algorithm
        if self.cat_feat_idxs is None or len(self.cat_feat_idxs) == 0:
            raise ValueError("No categorical features provided. Using sklearn.cluster.KMeans.")
        elif len(self.cat_feat_idxs) == X.shape[1]:
            raise ValueError("All features are categorical. Using FasterKModes.")
        
        # Validate init_Ccat
        if (init_Ccat is None) != (init_Cnum is None):
            raise ValueError("init_Ccat and init_Cnum must either both be None or both be provided.")

        if init_Ccat is not None:
            if not isinstance(init_Ccat, np.ndarray):
                raise ValueError("init_Ccat must be a numpy ndarray.")
            if init_Ccat.ndim != 2:
                raise ValueError("init_Ccat must be a 2-dimensional array.")
            if init_Ccat.shape[1] != len(self.cat_feat_idxs):
                raise ValueError("init_Ccat must have the same number of columns as the categorical features.")
            if not np.issubdtype(init_Ccat.dtype, np.integer):
                raise ValueError("init_Ccat must contain integer values.")
            if np.any(init_Ccat < 0) or np.any(init_Ccat >= X[:, self.cat_feat_idxs].max() + 1):
                raise ValueError("init_Ccat contains invalid values for categorical features.")

        # Validate init_Cnum
        if init_Cnum is not None:
            if not isinstance(init_Cnum, np.ndarray):
                raise ValueError("init_Cnum must be a numpy ndarray.")
            if init_Cnum.ndim != 2:
                raise ValueError("init_Cnum must be a 2-dimensional array.")
            if init_Cnum.shape[1] != len(self.num_feat_idxs):
                raise ValueError("init_Cnum must have the same number of columns as the numerical features.")
            if not np.issubdtype(init_Cnum.dtype, np.floating):
                raise ValueError("init_Cnum must contain float values.")

        # Check consistency of init_Ccat and init_Cnum with n_clusters
        if init_Ccat is not None and init_Cnum is not None:
            if init_Ccat.shape[0] != init_Cnum.shape[0]:
                raise ValueError("init_Ccat and init_Cnum must have the same number of rows (centroids).")
            if init_Ccat.shape[0] != self.n_clusters:
                raise ValueError("The number of rows in init_Ccat and init_Cnum must match n_clusters.")

        if X[:,self.cat_feat_idxs].max() > np.iinfo(np.uint8).max:
            self.input_cat_dtype = "uint16"
            self.itype = np.uint16
        else:
            self.input_cat_dtype = "uint8"
            self.itype = np.uint8

        if X[:,self.num_feat_idxs].max() > np.finfo(np.float32).max:
            self.input_num_dtype = "double"
            self.ftype = np.float64
            self.gemm = dgemm
            self.gemv = dgemv
        else:
            self.input_num_dtype = "float"
            self.ftype = np.float32
            self.gemm = sgemm
            self.gemv = sgemv

        self.n_cat_cols = len(self.cat_feat_idxs)
        self.n_num_cols = len(self.num_feat_idxs)

        self.simd_size = 32 if self.input_cat_dtype == "uint8" else 16
        self.n_cat_cols_simd = (self.n_cat_cols // self.simd_size) * self.simd_size
        self.n_cat_cols_remain = self.n_cat_cols % self.simd_size
        self.arg_dist_vec, self.arg_dist_mat, self.arg_matrix_counter, \
            self.arg_sample_density, self.arg_dist_x_dens = return_cat_argtypes(self.input_cat_dtype)
        self.arg_matrix_accumulator = return_num_argtypes(self.input_num_dtype)    

    def __validate_predict_X(self, X: np.ndarray):
        """
        Validate input data for prediction in k-prototypes clustering.

        Parameters:
            X (np.ndarray): The input data array for prediction.

        Raises:
            ValueError: If any validation check fails.
        """

        # Ensure the model has been fitted
        if not hasattr(self, "Ccat"):
            raise ValueError("Model has not been fitted. Please call 'fit' before using 'predict'.")

        # Check if X is a numpy array
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy ndarray.")

        # Check if X is a 2-dimensional array
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        # Check if the number of features in X matches the number of features in the training data
        if X.shape[1] != self.n_cols:
            raise ValueError(
                f"X must have the same number of columns as the training data. "
                f"Expected {self.n_cols}, got {X.shape[1]}."
            )

        # Check if the data type of X matches the expected data types
        allowed_dtypes = [np.uint8, np.uint16, np.float32, np.float64]
        if X.dtype not in allowed_dtypes:
            raise ValueError("X must have a dtype of uint8, uint16, float32, or float64.")

        # Check if X's dtype matches the dtype used in training
        if np.issubdtype(X.dtype, np.integer) and self.input_cat_dtype not in ["uint8", "uint16"]:
            raise ValueError(
                f"X contains integer values, but the model was trained with categorical data type {self.input_cat_dtype}."
            )
        if np.issubdtype(X.dtype, np.floating) and self.input_num_dtype not in ["float", "double"]:
            raise ValueError(
                f"X contains floating-point values, but the model was trained with numerical data type {self.input_num_dtype}."
            )

        # Validate categorical features
        if np.any(X[:, self.cat_feat_idxs] < 0):
            raise ValueError("Categorical features in X must be non-negative.")
        max_value = np.iinfo(np.uint16).max if self.input_cat_dtype == "uint16" else np.iinfo(np.uint8).max
        if np.any(X[:, self.cat_feat_idxs] > max_value):
            raise ValueError(
                f"Categorical features in X must not exceed {max_value}. "
                f"Current max value: {X[:, self.cat_feat_idxs].max()}."
            )

        # Check if numerical features are within a valid range
        max_value = np.finfo(np.float32).max if self.input_num_dtype == "float" else np.finfo(np.float64).max
        if np.any(np.abs(X[:, self.num_feat_idxs]) > max_value):
            raise ValueError(
                f"Numerical features in X exceed the maximum allowed value for {self.input_num_dtype}."
            )

        # Check if X is in C order
        if not X.flags["C_CONTIGUOUS"]:
            raise ValueError("X must have order='C'. Ensure the array is row-major memory layout.")

    def __select_common_funcs(self):
        self.lib.dist_x_dens.argtypes = self.arg_dist_x_dens

        if self.input_cat_dtype == "uint8":
            self.lib.matrix_counter_uint8.argtypes = self.arg_matrix_counter
            self.lib.sample_density_uint8.argtypes = self.arg_sample_density
            self.matrix_counter = self.lib.matrix_counter_uint8
            self.sample_density = self.lib.sample_density_uint8
        else:
            self.lib.matrix_counter_uint16.argtypes = self.arg_matrix_counter
            self.lib.sample_density_uint16.argtypes = self.arg_sample_density
            self.matrix_counter = self.lib.matrix_counter_uint16
            self.sample_density = self.lib.sample_density_uint16

        if self.input_num_dtype == "float":
            self.lib.matrix_accumulater_float.argtypes = self.arg_matrix_accumulator
            self.matrix_accumulater = self.lib.matrix_accumulater_float
        else:
            self.lib.matrix_accumulater_double.argtypes = self.arg_matrix_accumulator
            self.matrix_accumulater = self.lib.matrix_accumulater_double

    def __create_file_names(self):
        suffix = ""
        if (self.n_cat_cols >= 32) & (self.use_simd): suffix = "_SIMD"
        self.fn_get_nearest_dist = f"get_nearest_{self.categorical_measure}_dist_{self.input_cat_dtype}_{self.n_cat_cols}{suffix}"
        self.fn_dist_mat = f"{self.categorical_measure}_dist_mat_{self.input_cat_dtype}_{self.n_cat_cols}{suffix}"
        self.fn_dist_vec = f"{self.categorical_measure}_dist_vec_{self.input_cat_dtype}_{self.n_cat_cols}{suffix}"

    def fit(self, X: np.ndarray, categorical: list, init_Ccat: np.ndarray = None, init_Cnum: np.ndarray = None):
        self.__is_train = True
        self.__validate_train_X(X, categorical, init_Ccat, init_Cnum)
        self.__select_common_funcs()
        self.__create_file_names()

        self.__generate_compile_load_get_nearest_dist()
        self.__generate_compile_load_dist_vec()
        self.__generate_compile_load_dist_mat()

        Xnum = X[:, self.num_feat_idxs].copy().astype(self.ftype)
        Xcat = X[:, self.cat_feat_idxs].copy().astype(self.itype)

        self.Xnum_mean = Xnum.mean(axis=0).astype(self.ftype)
        self.Xnum_std = Xnum.std(axis=0).astype(self.ftype)

        Xnum = Xnum - self.Xnum_mean

        self.Xnum_sqsum = np.sum(Xnum*Xnum, axis=1).reshape((-1, 1))

        self.max_vals = Xcat.max(axis=0)
        self.offset = [0] + [self.max_vals.max()+1] * (self.n_cat_cols-1)
        self.offset = np.cumsum(self.offset).astype(np.int32)

        best_cost = np.finfo(np.float64).max
        best_cluster_cat = None
        best_cluster_num = None
        fast_break = init_Ccat is not None
        N = len(X)
        np.random.seed(self.random_state)
        for init in range(self.n_init):
            s = datetime.datetime.now()
            self.__select_initial_centroids(Xcat, Xnum, init_Ccat, init_Cnum)
            e = datetime.datetime.now()
            if self.print_log:
                print(f"Selected {self.n_clusters:>5}: ", (e-s).total_seconds())
            old_centroid_inds = - np.ones(N)

            n_no_update = 0
            for iter in range(self.max_iter):
                s = datetime.datetime.now()
                new_centroid_inds = self.__predict_centroid_indices(Xcat, Xnum, return_distance=False)
                e = datetime.datetime.now(); time_distance = (e-s).total_seconds()

                n_moves = np.sum(new_centroid_inds != old_centroid_inds)

                s = datetime.datetime.now()
                self.__update_centroids(Xcat, Xnum, old_centroid_inds, new_centroid_inds)
                e = datetime.datetime.now(); time_update = (e-s).total_seconds()

                cost = self.compute_score(X)
                if self.print_log:
                    print(f"{iter+1:=4}/{self.max_iter:=4} : N_MOVES = {n_moves:=10}, Cost = {cost:.3f}, Time-Distance={time_distance:10.5f}, Time-Update={time_update:10.5f}")

                old_centroid_inds = new_centroid_inds

                if cost < best_cost:
                    best_cost = cost
                    best_cluster_cat = self.Ccat
                    best_cluster_num = self.Cnum
                    self.best_cost = best_cost
                    n_no_update = 0
                else:
                    n_no_update += 1

                if n_moves <= self.min_n_moves:
                    break

                if self.max_tol is not None:
                    if n_no_update>self.max_tol:
                        break

            if fast_break: break

        self.Ccat = best_cluster_cat.astype(self.itype)
        self.Cnum = best_cluster_num.astype(self.ftype)
        np.random.seed(None)
        self.__is_train = False

    def compute_score(self, X: np.ndarray):
        self.__validate_predict_X(X)
        Xnum = X[:, self.num_feat_idxs].copy().astype(self.ftype) - self.Xnum_mean
        Xcat = X[:, self.cat_feat_idxs].copy().astype(self.itype)
        return self.__compute_distance_matrix(Xcat, Xnum).min(axis=1).sum()

    def predict(self, X: np.ndarray, return_distance: bool = False):
        self.__validate_predict_X(X)
        Xnum = np.array(X[:, self.num_feat_idxs].copy(), dtype=self.ftype, order="C") - self.Xnum_mean.astype(self.ftype)
        Xcat = np.array(X[:, self.cat_feat_idxs].copy(), dtype=self.itype, order="C")
        return self.__predict_centroid_indices(Xcat, Xnum, return_distance=return_distance)
