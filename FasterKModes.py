import os
import numpy as np
import subprocess
import ctypes
import datetime
from typing import Tuple
from numpy.typing import NDArray
import inspect

from src.generate_code import generate_get_nearest_cat_dist_code, generate_dist_vec_code, generate_dist_mat_code
from src.generate_code import generate_naive_dist_mat_code, generate_naive_get_nearest_cat_dist_code, generate_naive_dist_vec_code
from src.utils_kmodes import return_cat_argtypes
from config import VALID_INIT_METHODS

from BaseClusterer import BaseClusterer

Array1D = NDArray[Tuple[int]]  # 任意の長さの1次元配列

class FasterKModes(BaseClusterer):
    def __init__(
            self, 
            n_clusters=8, 
            max_iter=100, 
            min_n_moves=0, 
            n_init=10, 
            random_state=None, 
            init="k-means++", 
            categorical_measure="hamming", 
            n_jobs=None, 
            print_log=True, 
            recompile=False, 
            use_simd=False, 
            max_tol=None):
        
        super().__init__(
            n_clusters=n_clusters, 
            max_iter=max_iter, 
            min_n_moves=min_n_moves, 
            n_init=n_init, 
            random_state=random_state, 
            init=init, 
            categorical_measure=categorical_measure, 
            numerical_measure="euclidean", # <- dummy
            n_jobs=n_jobs, 
            print_log=print_log, 
            gamma=1.0,                     # <- dummy
            recompile=recompile, 
            use_simd=use_simd, 
            max_tol=max_tol
        )

        # init: 文字列の場合は VALID_INIT_METHODS、または callable で引数が ["X", "n_clusters"]
        if isinstance(init, str):
            if init not in VALID_INIT_METHODS:
                raise ValueError(f"init must be one of {VALID_INIT_METHODS}, but got '{init}'.")
        elif callable(init):
             # 引数のチェック
            sig = inspect.signature(init)
            if list(sig.parameters.keys()) != ["X", "n_clusters"]:
                raise ValueError(f"Custom init function must accept exactly two arguments: 'X' and 'n_clusters'. Got parameters: {list(sig.parameters.keys())}")
            # 戻り値のチェックはここでは実施しない。fit以降に実施する
        else:
            raise ValueError(f"init must be a string or a callable function, but got {init} (type: {type(init)}).")

    def __setstate__(self, state):
        self.is_loaded = True
        self.__dict__.update(state)

        self.arg_dist_vec, self.arg_dist_mat, self.arg_matrix_counter, \
            self.arg_sample_density, self.arg_dist_x_dens = return_cat_argtypes(self.input_cat_dtype)

        self._compile_lib(fn="common_funcs")
        self.lib = ctypes.CDLL("./src/common_funcs.so")
        self._generate_compile_load_get_nearest_cat_dist()
        self._generate_compile_load_dist_vec()
        self._generate_compile_load_dist_mat()

    def __getstate__(self):
        state = self.__dict__.copy()

        del state["arg_dist_mat"]
        del state["arg_dist_vec"]
        del state["arg_dist_x_dens"]
        del state["arg_matrix_counter"]
        del state["arg_sample_density"]
        del state["compute_cat_dist_mat"]
        del state["compute_dist_vec"]
        del state["get_nearest_cat_dist"]
        del state["lib"]
        del state["lib_dist_mat"]
        del state["lib_dist_vec"]
        del state["lib_get_nearest_cat_dist"]
        del state["matrix_counter"]
        del state["sample_density"]
        return state


    def __compute_distance_matrix(self, Xcat: np.ndarray):
        N = len(Xcat)
        dist_mat = np.zeros((N, self.n_clusters), dtype=np.int32, order="C")
        self.compute_cat_dist_mat(Xcat, N, self.n_cat_cols, self.C, self.n_clusters, dist_mat, self.n_jobs)
        return dist_mat

    def __predict_centroid_indices(self, Xcat: np.ndarray, return_distance: bool =False):
        dist_mat = self.__compute_distance_matrix(Xcat)
        nearest_centroid_indices = dist_mat.argmin(axis=1).astype(np.int32)
        if return_distance:
            return nearest_centroid_indices, dist_mat.min(axis=1)
        return nearest_centroid_indices

    def __select_initial_centroids(self, Xcat: np.ndarray, init_C: np.ndarray):
        N = len(Xcat)
        if init_C is not None:
            self.C = init_C
        elif self.init == "random":
            indices = list(range(N))
            shuffled_indices = np.random.permutation(indices)
            centroid_indices = shuffled_indices[:self.n_clusters]
            self.C = Xcat[centroid_indices,:]
        elif self.init == "k-means++":
            centroid_indices = []

            # Store Distances from X to Nearest Centroid
            dist_vec = np.array([np.iinfo(np.int32).max]*N, dtype=np.int32)

            # Random Selected Centroid
            rnd_idx = np.random.randint(0, len(Xcat))
            centroid_indices.append(rnd_idx)
            centroid_vec = Xcat[rnd_idx,:]
            self.get_nearest_cat_dist(Xcat, N, self.n_cat_cols, centroid_vec, dist_vec, self.n_jobs)

            # Select 2nd - N-th Centroids
            for c in range(1, self.n_clusters):
                weights = dist_vec / np.sum(dist_vec)
                rnd_idx = np.random.choice(N, p=weights)
                centroid_indices.append(rnd_idx)
                centroid_vec = Xcat[rnd_idx,:]
                self.get_nearest_cat_dist(Xcat, N, self.n_cat_cols, centroid_vec, dist_vec, self.n_jobs)
            self.C = Xcat[centroid_indices,:]
        elif self.init == "huang":
            self.C = np.zeros((self.n_clusters, self.n_cat_cols), dtype=Xcat.dtype)

            # ランダムに選択されたセントロイドを生成
            for k in range(self.n_cat_cols):
                X_k = Xcat[:,k]
                x_k = np.random.choice(X_k, size=self.n_clusters)
                self.C[:,k] = x_k

            # セントロイドを修正
            centroid_indices = []
            dist_mat = np.zeros((N, self.n_clusters), dtype=np.int32)
            self.compute_cat_dist_mat(Xcat, N, self.n_cat_cols, self.C, self.n_clusters, dist_mat, self.n_jobs)
            for k in range(self.n_clusters):
                dist_vec = dist_mat[:,k]
                ranking = np.argsort(dist_vec)
                for rnk in ranking:
                    if rnk not in centroid_indices:
                        centroid_indices.append(rnk)
                        break
            self.C = Xcat[centroid_indices,:]
        elif self.init == "cao":
            centroid_indices = []

            # 1番目のセントロイドを選択
            density_matrix = self._compute_density_matrix(Xcat)
            sample_densities = np.zeros((N,1), dtype=np.float32)
            self.sample_density(Xcat, N, self.n_cat_cols, density_matrix, sample_densities, self.offset)
            centroid_idx = np.argmax(sample_densities[:,0])
            centroid_indices.append(centroid_idx)
            distance_matrix = np.asfortranarray(np.zeros((N, self.n_clusters), dtype=np.int32))

            # 2個目のセントロイドを選択
            c_vector = Xcat[centroid_indices[0],:]
            dist_vec = self._compute_distance_vector(Xcat, c_vector)
            distance_matrix[:,0] = dist_vec
            centroid_idx = np.argmax(dist_vec * sample_densities[:,0])
            centroid_indices.append(centroid_idx)
            c_vector = Xcat[centroid_indices[1],:]
            dist_vec = self._compute_distance_vector(Xcat, c_vector)
            distance_matrix[:,1] = dist_vec

            # 3個目以降のセントロイドを選択
            tmp_dens_mat = np.asfortranarray(np.zeros((N, self.n_clusters), dtype=np.float32))
            tmp_dens_mat[:,0] = distance_matrix[:,0] * sample_densities[:,0]
            for c in range(2, self.n_clusters):
                self.lib.dist_x_dens(distance_matrix, N, self.n_clusters, c-1, sample_densities, tmp_dens_mat)
                centroid_idx = np.argmax(np.min(tmp_dens_mat[:,:c], axis=1))
                centroid_indices.append(centroid_idx)
                c_vector = Xcat[centroid_idx,:]
                dist_vec = self._compute_distance_vector(Xcat, c_vector)
                distance_matrix[:,c] = dist_vec
            self.C = Xcat[centroid_indices,:]
        elif callable(self.init):
            self.C = self.init(Xcat, self.n_clusters)
            self.__check_custom_init_method_output(Xcat)
        else:
            raise NotImplementedError
        
    def __check_custom_init_method_output(self, Xcat):
        # Validate self.C
        if not isinstance(self.C, np.ndarray):
            raise ValueError("self.C must be a numpy ndarray.")
        if self.C.ndim != 2:
            raise ValueError("self.C must be a 2-dimensional array.")
        if self.C.shape[1] != self.n_cat_cols:
            raise ValueError("self.C must have the same number of columns as the categorical features.")
        if self.C.dtype not in [np.uint8, np.uint16]:
            raise ValueError(f"self.C must be np.uint8 or np.uint16, not {self.C.dtype}.")
        if np.any(self.C > Xcat[:, :].max()):
            raise ValueError("init_Ccat contains invalid values for categorical features.")

    def __update_centroids(self, Xcat: np.ndarray, old_centroid_inds: Array1D, new_centroid_inds: Array1D):
        N = len(Xcat)
        max_val = self.max_vals.max()

        if old_centroid_inds[0] == -1:
            # 初回の処理
            self.count_matrix = np.zeros((self.n_clusters, self.n_cat_cols, (max_val+1)), dtype=np.int32) # Centroids x Category x Feature
        else:
            self.count_matrix = self.count_matrix * 0

        self.matrix_counter(Xcat, N, self.n_cat_cols, self.offset, new_centroid_inds, self.count_matrix, self.n_clusters, max_val, self.n_jobs)

        # Centroidの更新
        is_exist = [False] * self.n_clusters
        from collections import Counter
        # print(Counter(new_centroid_inds))
        for c in new_centroid_inds:
            is_exist[c] = True
        for c_ind in range(self.n_clusters):
            if is_exist[c_ind]:
                self.C[c_ind,:] = self.count_matrix[c_ind,:,:].argmax(axis=1)
            else: # Empty Cluster
                self.C[c_ind,:] = np.random.randint(0, max_val+1, self.n_cat_cols)

    def __validate_train_X(self, X: np.ndarray, init_C: np.ndarray):
        if (init_C is not None) and callable(self.init):
            raise ValueError("Cannot provide both a custom init function (init is callable) and init_C. Please specify one or the other.")
        
        # Check if X is a numpy array
        if not isinstance(X, np.ndarray):
            raise ValueError(f"Error: X must be a numpy array. Current input type:{type(X)}")

        # Check if X is a 2D array
        if X.ndim != 2:
            raise ValueError(f"Error: X must be a 2D array. Current ndim: {X.ndim}")

        # Check if X's dtype is uint8 or uint16
        if X.dtype not in [np.uint8, np.uint16]:
            raise ValueError(f"Error: X's dtype must be uint8 or uint16. Current dtype: {X.dtype}")

        # Check if X is in C order
        if not X.flags['C_CONTIGUOUS']:
            raise ValueError("X must have order='C'.")

        # Check the number of unique rows in X
        n_unique_rows = np.unique(X, axis=0).shape[0]
        if n_unique_rows < self.n_clusters:
            raise ValueError(
                f"The number of unique rows in X ({n_unique_rows}) must be greater than or equal to n_clusters ({self.n_clusters}).")

        # Validate init_C if provided
        if init_C is not None:
            # Check if init_C is a numpy array
            if not isinstance(init_C, np.ndarray):
                raise ValueError(f"Error: init_C must be a numpy array. Current input type: {type(init_C)}")

            # Check if init_C is a 2D array
            if init_C.ndim != 2:
                raise ValueError(f"Error: init_C must be a 2D array. Current ndim: {init_C.ndim}")

            # Check if init_C has the same number of columns as X
            if init_C.shape[1] != X.shape[1]:
                raise ValueError(f"Error: init_C must have the same number of columns as X. "
                        f"init_C columns: {init_C.shape[1]}, X columns: {X.shape[1]}")

            # Check if init_C's dtype matches X's dtype
            if init_C.dtype != X.dtype:
                raise ValueError(f"Error: init_C's dtype must match X's dtype. "
                        f"init_C dtype: {init_C.dtype}, X dtype: {X.dtype}")

            # Check if init_C's values are within the valid range of X
            if np.any(init_C > X.max()):
                raise ValueError("Error: init_C contains values outside the range of X.")

            # Check if init_C has the correct number of rows (clusters)
            if init_C.shape[0] != self.n_clusters:
                raise ValueError(f"Error: init_C must have the same number of rows as the number of clusters (n_clusters). "
                        f"init_C rows: {init_C.shape[0]}, n_clusters: {self.n_clusters}")

        self.input_cat_dtype = str(X.dtype)
        self.n_cat_cols = X.shape[1]
        self.simd_size = 32 if self.input_cat_dtype == "uint8" else 16
        self.n_cat_cols_simd = (self.n_cat_cols // self.simd_size) * self.simd_size
        self.n_cat_cols_remain = self.n_cat_cols % self.simd_size
        self.arg_dist_vec, self.arg_dist_mat, self.arg_matrix_counter, \
            self.arg_sample_density, self.arg_dist_x_dens = return_cat_argtypes(self.input_cat_dtype)

    def __validate_predict_X(self, X: np.ndarray):
        """
        Validate input data for prediction in KModes clustering.

        Parameters:
            X (np.ndarray): The input data array for prediction.

        Raises:
            ValueError: If any validation check fails.
        """

        # Ensure the model has been fitted
        if not self.is_fitted:
            raise ValueError("Error: Model has not been fitted. Please call 'fit' before using 'predict'.")

        # Check if X is a numpy array
        if not isinstance(X, np.ndarray):
            raise ValueError(f"Error: X must be a numpy array. Current input type: {type(X)}")

        # Check if X is a 2D array
        if X.ndim != 2:
            raise ValueError(f"Error: X must be a 2D array. Current ndim: {X.ndim}")

        # Check if X's dtype matches the model's input dtype
        if X.dtype.name != self.input_cat_dtype:
            raise ValueError(f"Error: X's dtype ({X.dtype.name}) does not match the model's trained input dtype ({self.input_cat_dtype}).")

        # Check if X is C-order
        if np.isfortran(X):
            raise ValueError(f"Error: X must be C-order (row-major memory layout). Use np.array(X, order='C') to convert.")

        # Check if X has the same number of columns as the training data
        if X.shape[1] != self.n_cat_cols:
            raise ValueError(f"Error: X must have the same number of columns as the training data. "
                    f"Expected {self.n_cat_cols}, got {X.shape[1]}.")

        if X.shape[0] < 1:
            raise ValueError(f"Error: X must have at least one row. Got {X.shape[0]} row(s).")
        
        # Check if X contains valid values for the trained model
        if self.input_cat_dtype == "uint8":
            max_value = np.iinfo(np.uint8).max
        elif self.input_cat_dtype == "uint16":
            max_value = np.iinfo(np.uint16).max
        else:
            raise ValueError(f"Error: Invalid input_cat_dtype ({self.input_cat_dtype}) in the model.")
        
    def __create_file_names(self):
        if callable(self.categorical_measure):
            return
        suffix = ""
        if (self.n_cat_cols >= 32) & (self.use_simd): suffix = "_SIMD"
        self.fn_get_nearest_cat_dist = f"get_nearest_{self.categorical_measure}_dist_{self.input_cat_dtype}_{self.n_cat_cols}{suffix}"
        self.fn_dist_mat = f"{self.categorical_measure}_dist_mat_{self.input_cat_dtype}_{self.n_cat_cols}{suffix}"
        self.fn_dist_vec = f"{self.categorical_measure}_dist_vec_{self.input_cat_dtype}_{self.n_cat_cols}{suffix}"

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

    def fit(self, X: np.ndarray, init_C: np.ndarray =None):
        np.random.seed(self.random_state)
        self.__validate_train_X(X, init_C)
        self.__select_common_funcs()
        self.__create_file_names()

        self._generate_compile_load_get_nearest_cat_dist()
        self._generate_compile_load_dist_vec()
        self._generate_compile_load_dist_mat()
        Xcat = X.copy()

        self.max_vals = Xcat.max(axis=0).astype(np.int32)
        self.offset = [0] + [self.max_vals.max()+1] * (self.n_cat_cols-1)
        self.offset = np.cumsum(self.offset).astype(np.int32)

        best_cost = np.finfo(np.float64).max
        best_cluster = None
        fast_break = init_C is not None
        N = len(Xcat)
        for _ in range(self.n_init):
            s = datetime.datetime.now()
            self.__select_initial_centroids(Xcat, init_C)
            e = datetime.datetime.now()
            if self.print_log:
                print(f"Selected {self.n_clusters:>5}: ", (e-s).total_seconds())
            old_centroid_inds = - np.ones(N)

            n_no_update = 0
            for iter in range(self.max_iter):
                s = datetime.datetime.now()
                new_centroid_inds = self.__predict_centroid_indices(Xcat, return_distance=False)
                e = datetime.datetime.now(); time_distance = (e-s).total_seconds()

                n_moves = np.sum(new_centroid_inds != old_centroid_inds)

                s = datetime.datetime.now()
                self.__update_centroids(Xcat, old_centroid_inds, new_centroid_inds)
                e = datetime.datetime.now(); time_update = (e-s).total_seconds()

                cost = self.__compute_score_in_train(Xcat)
                if self.print_log:
                    print(f"{iter+1:=4}/{self.max_iter:=4} : N_MOVES = {n_moves:=10}, Cost = {cost:=10}, Time-Distance={time_distance:10.5f}, Time-Update={time_update:10.5f}")

                old_centroid_inds = new_centroid_inds

                if cost < best_cost:
                    best_cost = cost
                    best_cluster = self.C
                    self.best_cost = best_cost
                    n_no_update = 0
                else:
                    n_no_update += 1

                if n_moves <= self.min_n_moves:
                    break

                if self.max_tol is not None:
                    if n_no_update>self.max_tol:
                        break

            if self.init == "cao": break
            if fast_break: break

        self.C = best_cluster
        np.random.seed(None)
        self.is_fitted = True

    def __compute_score_in_train(self, Xcat):
        return self.__compute_distance_matrix(Xcat).min(axis=1).sum()

    def compute_score(self, X: np.ndarray):
        Xcat = X.copy()
        self.__validate_predict_X(Xcat)
        return self.__compute_distance_matrix(Xcat).min(axis=1).sum()

    def predict(self, X: np.ndarray, return_distance=False):
        self.__validate_predict_X(X)
        Xcat = X.copy()
        return self.__predict_centroid_indices(Xcat, return_distance=return_distance)
