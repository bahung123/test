from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import faiss
from pyod.models.base import BaseDetector


class ANNKNN(BaseDetector):
    """KNN Outlier Detection với Approximate Nearest Neighbors sử dụng FAISS"""
    
    def __init__(self, n_neighbors=5, method='mean', contamination=0.1):
        super(ANNKNN, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.method = method  # 'mean', 'median', 'largest'
        
    def fit(self, X, y=None):
        self._set_n_classes(y)
        self.X_train_ = np.asarray(X, dtype=np.float32)
        n_samples, n_features = self.X_train_.shape
        
        # Tạo FAISS index
        self.index = faiss.IndexFlatL2(n_features)
        self.index.add(self.X_train_)
        
        # Tính outlier scores cho training data
        distances, _ = self.index.search(self.X_train_, self.n_neighbors + 1)
        distances = distances[:, 1:]  # Loại bỏ distance tới chính nó
        
        if self.method == 'mean':
            self.decision_scores_ = np.mean(distances, axis=1)
        elif self.method == 'median':
            self.decision_scores_ = np.median(distances, axis=1)
        elif self.method == 'largest':
            self.decision_scores_ = np.max(distances, axis=1)
            
        self._process_decision_scores()
        return self
    
    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        distances, _ = self.index.search(X, self.n_neighbors)
        
        if self.method == 'mean':
            scores = np.mean(distances, axis=1)
        elif self.method == 'median':
            scores = np.median(distances, axis=1)
        elif self.method == 'largest':
            scores = np.max(distances, axis=1)
            
        return scores

class ANNLOF(BaseDetector):
    """LOF Outlier Detection với Approximate Nearest Neighbors sử dụng FAISS"""
    
    def __init__(self, n_neighbors=20, contamination=0.1):
        super(ANNLOF, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y=None):
        self._set_n_classes(y)
        self.X_train_ = np.asarray(X, dtype=np.float32)
        n_samples, n_features = self.X_train_.shape
        
        # Tạo FAISS index
        self.index = faiss.IndexFlatL2(n_features)
        self.index.add(self.X_train_)
        
        # Tính LOF scores
        self.decision_scores_ = self._compute_lof_scores(self.X_train_)
        self._process_decision_scores()
        return self
    
    def _compute_lof_scores(self, X):
        # Tìm k-nearest neighbors
        distances, indices = self.index.search(X, self.n_neighbors + 1)
        distances = distances[:, 1:]  # Loại bỏ distance tới chính nó
        indices = indices[:, 1:]
        
        # Tính k-distance
        k_distances = distances[:, -1]
        
        # Tính reachability distance
        reach_dist = np.zeros_like(distances)
        for i in range(X.shape[0]):
            for j, neighbor_idx in enumerate(indices[i]):
                reach_dist[i, j] = max(distances[i, j], k_distances[neighbor_idx])
        
        # Tính local reachability density
        lrd = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if np.sum(reach_dist[i]) > 0:
                lrd[i] = 1.0 / (np.mean(reach_dist[i]) + 1e-10)
            else:
                lrd[i] = np.inf
        
        # Tính LOF scores
        lof_scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            neighbor_lrds = lrd[indices[i]]
            if lrd[i] > 0:
                lof_scores[i] = np.mean(neighbor_lrds) / lrd[i]
            else:
                lof_scores[i] = 1.0
                
        return lof_scores
    
    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self._compute_lof_scores(X)

# Version sử dụng sklearn NearestNeighbors với thuật toán nhanh hơn
class FastKNN(BaseDetector):
    """KNN nhanh sử dụng ball_tree hoặc kd_tree"""
    
    def __init__(self, n_neighbors=5, algorithm='ball_tree', contamination=0.1):
        super(FastKNN, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        
    def fit(self, X, y=None):
        self._set_n_classes(y)
        self.X_train_ = X
        
        # Sử dụng thuật toán nhanh
        self.neigh_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            algorithm=self.algorithm,
            metric='euclidean'
        )
        self.neigh_.fit(X)
        
        # Tính outlier scores
        distances, _ = self.neigh_.kneighbors(X)
        distances = distances[:, 1:]  # Loại bỏ distance tới chính nó
        self.decision_scores_ = np.mean(distances, axis=1)
        
        self._process_decision_scores()
        return self
    
    def decision_function(self, X):
        distances, _ = self.neigh_.kneighbors(X)
        return np.mean(distances, axis=1)
