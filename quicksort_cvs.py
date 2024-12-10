import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional
import warnings

class QPLO:
    """Performs Quick-sort Leave-Pair-Out cross-validation with enhanced caching."""
    
    def __init__(self, classifier, X, y, groups):
        """
        Initialize QPLO with improved memory management.
        
        Parameters:
        -----------
        classifier : estimator object
            The machine learning model to be used for classification
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target labels
        groups : numpy.ndarray
            Group labels for each sample
        """
        self.classifier = classifier
        self.X = X
        self.y = y
        self.groups = groups
        self.unique_groups = np.unique(groups)
        self.n_groups = len(self.unique_groups)
        self.comparisons: Dict[Tuple[int, int], float] = {}
        
        # Precompute and cache group indices
        self.group_indices = {
            group: np.where(groups == group)[0] 
            for group in self.unique_groups
        }

    def _compare_groups(self, i: int, j: int) -> float:
        """
        Evaluates performance difference between groups with enhanced caching.
        
        Parameters:
        -----------
        i : int
            Index of first group
        j : int
            Index of second group
            
        Returns:
        --------
        float
            Performance difference between groups
        """
        if (i, j) in self.comparisons:
            return self.comparisons[(i, j)]
        
        group_i, group_j = self.unique_groups[i], self.unique_groups[j]
        indices_i = self.group_indices[group_i]
        indices_j = self.group_indices[group_j]
        
        # Define training indices efficiently
        train_mask = ~np.isin(np.arange(len(self.X)), np.concatenate([indices_i, indices_j]))
        train_indices = np.where(train_mask)[0]
        
        X_train = self.X[train_indices]
        y_train = self.y[train_indices]
        X_test = np.vstack((self.X[indices_i], self.X[indices_j]))
        y_test = np.concatenate((self.y[indices_i], self.y[indices_j]))
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            result = 0
        else:
            try:
                self.classifier.fit(X_train, y_train)
                proba = self.classifier.predict_proba(X_test)[:, 1]
                score_i = np.mean(proba[:len(indices_i)])
                score_j = np.mean(proba[len(indices_i):])
                result = float(score_i - score_j)
            except Exception:
                result = 0
        
        self.comparisons[(i, j)] = result
        self.comparisons[(j, i)] = -result
        return result

    def sort(self) -> List:
        """
        Sorts groups using quicksort algorithm with optimized comparisons.
        
        Returns:
        --------
        list
            Ordered list of unique groups
        """
        def quicksort(arr: List) -> List:
            if len(arr) <= 1:
                return arr
            
            pivot_idx = len(arr) // 2
            pivot = arr[pivot_idx]
            left = []
            middle = [pivot]
            right = []
            
            for x in arr:
                if x == pivot:
                    continue
                cmp = self._compare_groups(x, pivot)
                if cmp < 0:
                    left.append(x)
                elif cmp > 0:
                    right.append(x)
                else:
                    middle.append(x)
            
            return quicksort(left) + middle + quicksort(right)
        
        indices = list(range(self.n_groups))
        sorted_indices = quicksort(indices)
        return [self.unique_groups[i] for i in sorted_indices]

class AQPLO:
    """Performs Adaptive Quick-sort Leave-Pair-Out cross-validation with optimized performance."""
    
    def __init__(self, classifier, X, y, groups):
        """
        Initialize AQPLO with enhanced memory management and adaptive parameters.
        
        Parameters:
        -----------
        classifier : estimator object
            The machine learning model to be used for classification
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target labels
        groups : numpy.ndarray
            Group labels for each sample
        """
        self.classifier = classifier
        self.X = X
        self.y = y
        self.groups = groups
        self.unique_groups = np.unique(groups)
        self.n_groups = len(self.unique_groups)
        self.comparisons: Dict[Tuple[int, int], float] = {}
        self.group_scores: Dict[int, float] = {}
        
        # Adaptive tolerance based on dataset characteristics
        self.tolerance = 0.01
        
        # Precompute group indices for efficiency
        self.group_indices = {
            group: np.where(groups == group)[0] 
            for group in self.unique_groups
        }
        
        # Optimize cross-validation folds based on group sizes
        self.cv_folds = {
            group: min(3, len(indices)) 
            for group, indices in self.group_indices.items()
        }

    def _get_group_score(self, group) -> float:
        """
        Computes performance score for a group with optimized cross-validation.
        
        Parameters:
        -----------
        group : int or str
            Group identifier
            
        Returns:
        --------
        float
            ROC AUC score for the group
        """
        if group in self.group_scores:
            return self.group_scores[group]
        
        indices = self.group_indices[group]
        X_group = self.X[indices]
        y_group = self.y[indices]
        
        if len(np.unique(y_group)) < 2:
            score = 0.5
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proba = cross_val_predict(
                        self.classifier,
                        X_group,
                        y_group,
                        method='predict_proba',
                        cv=self.cv_folds[group],
                        n_jobs=1
                    )[:, 1]
                score = float(roc_auc_score(y_group, proba))
            except Exception:
                score = 0.5
        
        self.group_scores[group] = score
        return score

    def _compare_groups(self, i: int, j: int) -> float:
        """
        Compares performance scores of groups with adaptive thresholding.
        
        Parameters:
        -----------
        i : int
            Index of first group
        j : int
            Index of second group
            
        Returns:
        --------
        float
            Adjusted performance difference between groups
        """
        if (i, j) in self.comparisons:
            return self.comparisons[(i, j)]
        
        score_i = self._get_group_score(self.unique_groups[i])
        score_j = self._get_group_score(self.unique_groups[j])
        
        diff = score_i - score_j
        if abs(diff) < self.tolerance:
            result = 0
        else:
            result = float(diff)
        
        self.comparisons[(i, j)] = result
        self.comparisons[(j, i)] = -result
        return result

    def sort(self) -> List:
        """
        Sorts groups using adaptive quicksort algorithm with optimized comparisons.
        
        Returns:
        --------
        list
            Ordered list of unique groups
        """
        def quicksort(arr: List) -> List:
            if len(arr) <= 1:
                return arr
            
            # Choose middle element as pivot for stability
            pivot = arr[len(arr) // 2]
            left = []
            middle = [pivot]
            right = []
            
            for x in arr:
                if x == pivot:
                    continue
                cmp = self._compare_groups(x, pivot)
                if cmp < 0:
                    left.append(x)
                elif cmp > 0:
                    right.append(x)
                else:
                    middle.append(x)
            
            return quicksort(left) + middle + quicksort(right)

        indices = list(range(self.n_groups))
        sorted_indices = quicksort(indices)
        return [self.unique_groups[i] for i in sorted_indices]

def quicksort_leave_pair_out(X, y, groups, classifier):
    """
    Generate training/testing splits using QPLO cross-validation.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target labels
    groups : numpy.ndarray
        Group labels for each sample
    classifier : estimator object
        The machine learning model
        
    Yields:
    -------
    tuple
        (train_indices, test_indices) for each split
    """
    qplo = QPLO(classifier, X, y, groups)
    sorted_groups = qplo.sort()
    
    for i in range(len(sorted_groups)):
        for j in range(i + 1, len(sorted_groups)):
            group_i, group_j = sorted_groups[i], sorted_groups[j]
            
            train_mask = ~np.isin(groups, [group_i, group_j])
            test_mask = ~train_mask
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if (len(np.unique(y[train_indices])) < 2 or 
                len(np.unique(y[test_indices])) < 2):
                continue
            
            yield train_indices, test_indices

def adaptive_quicksort_leave_pair_out(X, y, groups, classifier):
    """
    Generate training/testing splits using AQPLO cross-validation.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target labels
    groups : numpy.ndarray
        Group labels for each sample
    classifier : estimator object
        The machine learning model
        
    Yields:
    -------
    tuple
        (train_indices, test_indices) for each split
    """
    aqplo = AQPLO(classifier, X, y, groups)
    sorted_groups = aqplo.sort()
    
    # Adaptive number of pairs based on dataset size
    n_pairs = min(len(sorted_groups) * (len(sorted_groups) - 1) // 2,
                  max(20, len(X) // 50))
    
    pairs_yielded = 0
    for i in range(len(sorted_groups)):
        if pairs_yielded >= n_pairs:
            break
            
        for j in range(i + 1, len(sorted_groups)):
            if pairs_yielded >= n_pairs:
                break
                
            group_i, group_j = sorted_groups[i], sorted_groups[j]
            
            train_mask = ~np.isin(groups, [group_i, group_j])
            test_mask = ~train_mask
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if (len(np.unique(y[train_indices])) < 2 or 
                len(np.unique(y[test_indices])) < 2):
                continue
            
            yield train_indices, test_indices
            pairs_yielded += 1