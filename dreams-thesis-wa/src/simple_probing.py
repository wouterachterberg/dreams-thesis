"""
Simple probing utilities for final embeddings analysis.

Uses pre-split scaffold-based train/test sets to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class SimpleProbe:
    """
    Simple probing class for testing what's encoded in embeddings.
    
    Works with final embeddings only - no layer-by-layer complexity yet.
    Uses pre-split train/test data (scaffold-based) to prevent leakage.
    """
    
    def __init__(self, embeddings: np.ndarray, random_state: int = 42):
        """
        Args:
            embeddings: Final layer embeddings, shape (n_samples, embedding_dim)
            random_state: Random seed for reproducibility
        """
        self.embeddings = embeddings
        self.random_state = random_state
        self.results = {}
        
    def probe_binary(
        self,
        target: np.ndarray,
        target_name: str,
        probe_type: str = 'linear',
        train_mask: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Probe for binary classification task.
        
        Args:
            target: Binary labels (0/1) for all samples
            target_name: Name of the target (e.g., 'has_alcohol')
            probe_type: 'linear' or 'mlp'
            train_mask: Boolean mask for training samples (use pre-split data!)
            test_mask: Boolean mask for test samples (use pre-split data!)
            
        Returns:
            Dictionary with AUROC, Average Precision, Accuracy
        """
        # Use pre-defined train/test splits (scaffold-aware from dataset)
        if train_mask is not None and test_mask is not None:
            X_train, X_test = self.embeddings[train_mask], self.embeddings[test_mask]
            y_train, y_test = target[train_mask], target[test_mask]
        else:
            raise ValueError("Must provide train_mask and test_mask! Use scaffold-based splits from dataset.")
        
        # Train probe
        if probe_type == 'linear':
            probe = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:  # mlp
            probe = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            )
        
        probe.fit(X_train, y_train)
        
        # Evaluate
        y_proba = probe.predict_proba(X_test)[:, 1]
        y_pred = probe.predict(X_test)
        
        metrics = {
            'auroc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba),
            'accuracy': (y_pred == y_test).mean()
        }
        
        # Store results
        self.results[f"{target_name}_{probe_type}"] = metrics
        
        print(f"\n{target_name} ({probe_type} probe):")
        print(f"  AUROC: {metrics['auroc']:.3f}")
        print(f"  AP: {metrics['avg_precision']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        
        return metrics
    
    def probe_regression(
        self,
        target: np.ndarray,
        target_name: str,
        probe_type: str = 'linear',
        train_mask: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Probe for regression task.
        
        Args:
            target: Continuous values for all samples
            target_name: Name of the target (e.g., 'mol_weight')
            probe_type: 'linear' or 'mlp'
            train_mask: Boolean mask for training samples (use pre-split data!)
            test_mask: Boolean mask for test samples (use pre-split data!)
            
        Returns:
            Dictionary with R2, MSE, RMSE
        """
        # Standardize target
        scaler = StandardScaler()
        target_scaled = scaler.fit_transform(target.reshape(-1, 1)).squeeze()
        
        # Use pre-defined train/test splits (scaffold-aware from dataset)
        if train_mask is not None and test_mask is not None:
            X_train, X_test = self.embeddings[train_mask], self.embeddings[test_mask]
            y_train, y_test = target_scaled[train_mask], target_scaled[test_mask]
        else:
            raise ValueError("Must provide train_mask and test_mask! Use scaffold-based splits from dataset.")
        
        # Train probe
        if probe_type == 'linear':
            probe = Ridge(alpha=1.0, random_state=self.random_state)
        else:  # mlp
            probe = MLPRegressor(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            )
        
        probe.fit(X_train, y_train)
        
        # Evaluate
        y_pred = probe.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Store results
        self.results[f"{target_name}_{probe_type}"] = metrics
        
        print(f"\n{target_name} ({probe_type} probe):")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        
        return metrics
    
    def get_results_df(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame(self.results).T


def knn_validation(
    embeddings: np.ndarray,
    targets: np.ndarray,
    k: int = 10,
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None
) -> float:
    """
    Check if embeddings encode target via k-NN retrieval.
    
    For each test sample, find k nearest neighbors in train set.
    Measure how well neighbor labels predict the test label.
    
    Args:
        embeddings: Embedding vectors for all samples
        targets: Binary or continuous targets for all samples
        k: Number of neighbors
        train_mask: Boolean mask for training samples
        test_mask: Boolean mask for test samples
        
    Returns:
        Mean agreement between test label and neighbor labels
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Use pre-defined train/test splits
    if train_mask is not None and test_mask is not None:
        X_train, X_test = embeddings[train_mask], embeddings[test_mask]
        y_train, y_test = targets[train_mask], targets[test_mask]
    else:
        raise ValueError("Must provide train_mask and test_mask! Use scaffold-based splits from dataset.")
    
    # Fit k-NN
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X_train)
    
    # Find neighbors for test samples
    distances, indices = knn.kneighbors(X_test)
    
    # For binary targets: measure how often majority vote matches
    if len(np.unique(targets)) == 2:
        neighbor_labels = y_train[indices]
        predictions = (neighbor_labels.mean(axis=1) > 0.5).astype(int)
        agreement = (predictions == y_test).mean()
    else:
        # For continuous targets: use mean of neighbors
        neighbor_values = y_train[indices]
        predictions = neighbor_values.mean(axis=1)
        agreement = 1 - np.abs(predictions - y_test).mean()  # Inverse MAE
    
    print(f"\nk-NN validation (k={k}):")
    print(f"  Agreement: {agreement:.3f}")
    
    return agreement


def umap_visualization(
    embeddings: np.ndarray,
    targets: np.ndarray,
    target_name: str,
    scaffold_ids: Optional[np.ndarray] = None,
    n_components: int = 2,
    save_path: Optional[str] = None
):
    """
    Visualize embeddings in 2D with UMAP, colored by target.
    
    Args:
        embeddings: Embedding vectors
        targets: Target values to color by
        target_name: Name for the plot title
        scaffold_ids: Scaffold IDs for stratified sampling if needed
        n_components: UMAP dimensions (2 or 3)
        save_path: Where to save the plot
    """
    import umap
    
    # Reduce dimensionality
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if len(np.unique(targets)) <= 10:
        # Categorical coloring
        scatter = plt.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=targets, cmap='tab10', alpha=0.6, s=10
        )
        plt.colorbar(scatter, label=target_name)
    else:
        # Continuous coloring
        scatter = plt.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=targets, cmap='viridis', alpha=0.6, s=10
        )
        plt.colorbar(scatter, label=target_name)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'Embedding Space: {target_name}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
