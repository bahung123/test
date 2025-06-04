from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import faiss
from pyod.models.base import BaseDetector


class ANNKNN(BaseDetector):


    def __init__(self, n_neighbors=5, method='largest', contamination=0.1):
        super(ANNKNN, self).__init__(contamination=contamination)
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if method not in ['mean', 'median', 'largest']:
            raise ValueError("method must be 'mean', 'median', or 'largest'")
        self.n_neighbors = n_neighbors
        self.method = method 
        

    def fit(self, X, y=None):
        self._set_n_classes(y)
        X = check_array(X, accept_sparse=False)
        self.X_train_ = np.asarray(X, dtype=np.float32)
        n_samples, n_features = self.X_train_.shape
    
        # Kiểm tra n_neighbors không vượt quá số samples
        if self.n_neighbors >= n_samples:
            self.n_neighbors = n_samples - 1
            warn(f"n_neighbors was larger than the number of samples, "
                 f"setting n_neighbors = {self.n_neighbors}")

        quantizer = faiss.IndexFlatL2(n_features)
        self.index = faiss.IndexIVFFlat(quantizer, n_features, min(100, n_samples), faiss.METRIC_L2)

        self.index.train(self.X_train_) 
        self.index.add(self.X_train_)

        # Tính outlier scores cho training data
        distances, _ = self.index.search(self.X_train_, self.n_neighbors + 1)
        distances = distances[:, 1:] 
    
        if self.method == 'mean':
            self.decision_scores_ = np.mean(distances, axis=1)
        elif self.method == 'median':
            self.decision_scores_ = np.median(distances, axis=1)
        elif self.method == 'largest':
            self.decision_scores_ = np.max(distances, axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._process_decision_scores()
        return self

    
    def decision_function(self, X):
        check_is_fitted(self, ['index', 'X_train_'])
        X_checked = check_array(X, accept_sparse=False)
        X_float32 = np.asarray(X_checked, dtype=np.float32)

        distances_searched, _ = self.index.search(X_float32, self.n_neighbors + 1)

        is_X_train = (X_float32.shape == self.X_train_.shape and np.array_equal(X_float32, self.X_train_))

        if is_X_train:
            # Bỏ qua khoảng cách đến chính nó
            knn_distances = distances_searched[:, 1:self.n_neighbors+1] 
        else:
            # X là dữ liệu mới, lấy k khoảng cách đầu tiên
            knn_distances = distances_searched[:, :self.n_neighbors]

        if self.method == 'mean':
            scores = np.mean(knn_distances, axis=1)
        elif self.method == 'median':
            scores = np.median(knn_distances, axis=1)
        elif self.method == 'largest': 
            scores = np.max(knn_distances, axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}") 

        return scores

class ANNLOF(BaseDetector):
    
    def __init__(self, n_neighbors=20, contamination=0.1):
        super(ANNLOF, self).__init__(contamination=contamination)
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = n_neighbors
        
        
    def fit(self, X, y=None):
        self._set_n_classes(y)
        X = check_array(X, accept_sparse=False)
        self.X_train_ = np.asarray(X, dtype=np.float32)
        n_samples, n_features = self.X_train_.shape

        # Kiểm tra n_neighbors 
        if self.n_neighbors >= n_samples:
            actual_n_neighbors = n_samples -1
            warn(f"n_neighbors ({self.n_neighbors}) >= n_samples ({n_samples}). "
                f"Setting n_neighbors to {actual_n_neighbors} for fit.")
        else:
            actual_n_neighbors = self.n_neighbors

        quantizer = faiss.IndexFlatL2(n_features)
    
        nlist = min(100, n_samples) if n_samples > 0 else 1
        if n_samples > 0 and actual_n_neighbors == 0 and n_samples == 1:  
            self.index = faiss.IndexFlatL2(n_features)  
        elif n_samples < nlist and n_samples > 0:  
            self.index = faiss.IndexIVFFlat(quantizer, n_features, n_samples, faiss.METRIC_L2)
        elif n_samples == 0:
            raise ValueError("Cannot fit on empty data.")
        else:
            self.index = faiss.IndexIVFFlat(quantizer, n_features, nlist, faiss.METRIC_L2)

        if n_samples > 0: 
            self.index.train(self.X_train_)
            self.index.add(self.X_train_)

            distances_train_searched, indices_train_searched = self.index.search(self.X_train_, actual_n_neighbors + 1)

            # Bỏ qua chính nó
            distances_to_neighbors_train = distances_train_searched[:, 1 : actual_n_neighbors + 1] 
            indices_of_neighbors_train = indices_train_searched[:, 1 : actual_n_neighbors + 1]

            # k-distance của mỗi điểm trong X_train_
            self._k_distances_train = distances_to_neighbors_train[:, -1] # Lưu lại

            reach_dist_train = np.zeros_like(distances_to_neighbors_train)
            for i in range(n_samples):
                for j in range(actual_n_neighbors): # Duyệt qua actual_n_neighbors
                    neighbor_idx = indices_of_neighbors_train[i, j]
                    reach_dist_train[i, j] = max(distances_to_neighbors_train[i, j], self._k_distances_train[neighbor_idx])

            self.lrd_ = np.zeros(n_samples)
            for i in range(n_samples):
                if actual_n_neighbors == 0: # Không có lân cận nào
                    self.lrd_[i] = np.inf # Hoặc 0, tùy định nghĩa cho trường hợp này
                else:
                    mean_reach_dist = np.mean(reach_dist_train[i, :])
                    if mean_reach_dist > 0:
                        self.lrd_[i] = 1.0 / mean_reach_dist
                    else:
                        self.lrd_[i] = np.inf # Điểm trùng lặp hoặc rất gần
        else: # Không có dữ liệu huấn luyện
            self.lrd_ = np.array([])
            self._k_distances_train = np.array([])


        # Tính LOF scores cho training data
        if n_samples > 0:
            self.decision_scores_ = self._compute_lof_scores(self.X_train_, is_training_data=True)
        else:
            self.decision_scores_ = np.array([])

        self._process_decision_scores() 
        return self

    def _compute_lof_scores(self, X, is_training_data=False):
        check_is_fitted(self, ['index', 'X_train_', 'lrd_', '_k_distances_train'])
        X_float32 = np.asarray(X, dtype=np.float32)

        if self.X_train_.shape[0] == 0: # Nếu không có dữ liệu huấn luyện
            return np.ones(X_float32.shape[0]) # Trả về điểm mặc định
        num_neighbors_to_use = min(self.n_neighbors, self.X_train_.shape[0] -1 if self.X_train_.shape[0] > 0 else 0)
        if num_neighbors_to_use <=0 and self.X_train_.shape[0] > 0 : num_neighbors_to_use = 1 # Cần ít nhất 1 lân cận nếu có thể
        if self.X_train_.shape[0] == 0 : num_neighbors_to_use = 0


        if num_neighbors_to_use == 0: # Không có lân cận nào để so sánh
            return np.ones(X_float32.shape[0])


        distances_searched, indices_searched = self.index.search(X_float32, num_neighbors_to_use + 1)

        if is_training_data:
            # Bỏ qua chính nó, lấy num_neighbors_to_use lân cận
            actual_distances = distances_searched[:, 1 : num_neighbors_to_use + 1]
            actual_indices = indices_searched[:, 1 : num_neighbors_to_use + 1]
        else:
            # Lấy num_neighbors_to_use lân cận đầu tiên
            actual_distances = distances_searched[:, :num_neighbors_to_use]
            actual_indices = indices_searched[:, :num_neighbors_to_use]

        # Tính reachability distance cho X
        reach_dist_X = np.zeros_like(actual_distances) # Shape (X.shape[0], num_neighbors_to_use)
        for i in range(X_float32.shape[0]):
            for j in range(num_neighbors_to_use):
                neighbor_original_idx = actual_indices[i, j]
                if neighbor_original_idx < len(self._k_distances_train):
                    k_dist_of_neighbor = self._k_distances_train[neighbor_original_idx]
                else:
                    k_dist_of_neighbor = np.inf

                dist_X_to_neighbor = actual_distances[i, j]
                reach_dist_X[i, j] = max(dist_X_to_neighbor, k_dist_of_neighbor)

        # Tính LRD cho X
        lrd_X = np.zeros(X_float32.shape[0])
        for i in range(X_float32.shape[0]):
            if num_neighbors_to_use == 0:
                lrd_X[i] = np.inf
            else:
                mean_reach_dist = np.mean(reach_dist_X[i, :])
                if mean_reach_dist > 0:
                    lrd_X[i] = 1.0 / mean_reach_dist
                else:
                    lrd_X[i] = np.inf

        # Tính LOF scores cho X
        lof_scores = np.zeros(X_float32.shape[0])
        for i in range(X_float32.shape[0]):
            if num_neighbors_to_use == 0:
                lof_scores[i] = 1.0 # Default
                continue

            
            neighbor_lrds_from_train = self.lrd_[actual_indices[i, :]]

            finite_neighbor_lrds = neighbor_lrds_from_train[np.isfinite(neighbor_lrds_from_train)]
            if not finite_neighbor_lrds.size: # Tất cả LRD của lân cận là inf hoặc không có lân cận hợp lệ
                mean_lrd_of_neighbors = np.inf 
            else:
                mean_lrd_of_neighbors = np.mean(finite_neighbor_lrds)

            current_lrd_X_i = lrd_X[i]
            if current_lrd_X_i > 0 and not np.isinf(current_lrd_X_i):
                if np.isinf(mean_lrd_of_neighbors): 
                    lof_scores[i] = np.inf
                elif mean_lrd_of_neighbors == 0: 
                    lof_scores[i] = 1.0
                else:
                    lof_scores[i] = mean_lrd_of_neighbors / current_lrd_X_i
            elif np.isinf(current_lrd_X_i): 
                if np.isinf(mean_lrd_of_neighbors): 
                    lof_scores[i] = 1.0 
                else: 
                    lof_scores[i] = 0.0 
            else: 
                lof_scores[i] = 1.0 

        return lof_scores

    def decision_function(self, X):
        # check_is_fitted đã có trong _compute_lof_scores
        X_checked = check_array(X, accept_sparse=False) 
        return self._compute_lof_scores(X_checked, is_training_data=False)