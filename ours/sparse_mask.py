"""
sparse_mask.py
Structured sparse attention mask generation module
Used to build a top-k neighbor graph based on feature correlation and generate Transformer attention masks
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def compute_feature_graph(data, top_k=5, method='correlation'):
    """
    Calculate correlation graph between features and generate top-k neighbor masks
    
    Args:
        data: pandas DataFrame, complete dataset (including features and labels)
        top_k: int, number of neighbors to keep for each feature
        method: str, correlation calculation method (currently only 'correlation' supported)
        
    Returns:
        adjacency_matrix: numpy array, shape [n_features, n_features], adjacency matrix
        attention_mask: torch tensor, shape [n_features, n_features], Transformer attention mask
    """
    # Extract features (excluding the last column label)
    X = data.iloc[:, :-1]
    n_features = X.shape[1]
    
    # Calculate correlation matrix
    if method == 'correlation':
        # Use Pearson correlation coefficient
        corr_matrix = X.corr().abs().values  # Take absolute value
        # Replace NaN with 0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    else:
        raise ValueError(f"Unknown method: {method}. Currently only 'correlation' is supported.")
    
    # Build adjacency matrix: for each feature, keep the top-k most relevant neighbors
    adjacency_matrix = np.zeros((n_features, n_features), dtype=np.float32)
    
    for i in range(n_features):
        # Get correlation with feature i (exclude self)
        correlations = corr_matrix[i].copy()
        correlations[i] = -1  # Set self to minimum to exclude
        
        # Find top-k neighbors
        if top_k >= n_features - 1:
            # If top_k is greater than or equal to the number of features, connect all features
            top_k_indices = list(range(n_features))
        else:
            top_k_indices = np.argsort(correlations)[-top_k:].tolist()
        
        # Mark connections in adjacency matrix
        adjacency_matrix[i, top_k_indices] = 1.0
        adjacency_matrix[i, i] = 1.0  # Self-connection
    
    # Symmetrize adjacency matrix (if i connects to j, then j also connects to i)
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    
    # Generate Transformer attention mask
    # In Transformer: mask=True means masked, mask=False means visible
    # We need to convert the adjacency matrix to a boolean mask
    attention_mask = torch.from_numpy(adjacency_matrix == 0)  # Places without connections are True (masked)
    
    return adjacency_matrix, attention_mask


class FeatureMaskGenerator:
    """Feature mask generator, used to manage and apply masks in the model"""
    
    def __init__(self, data, top_k=5, method='correlation'):
        """
        Args:
            data: pandas DataFrame, complete data containing features and labels
            top_k: int, number of neighbors for each feature
            method: str, correlation calculation method
        """
        self.n_features = data.shape[1] - 1  # subtract label column
        self.top_k = top_k
        
        # Calculate adjacency matrix and mask
        self.adjacency_matrix, self.attention_mask = compute_feature_graph(
            data, top_k=top_k, method=method
        )
        
        print(f"✅ Feature mask generation complete: {self.n_features} features, each keeping top-{top_k} neighbors")
        print(f"   Connection density: {self.adjacency_matrix.sum() / (self.n_features ** 2):.2%}")
    
    def get_encoder_mask(self, device='cuda'):
        """
        Get mask used by Encoder (boolean mask, suitable for nn.TransformerEncoder)
        
        Returns:
            mask: torch tensor, shape [n_features, n_features]
        """
        return self.attention_mask.to(device)
    
    def get_diffusion_mask(self, device='cuda'):
        """
        Get mask used by Diffusion Model (additive form, suitable for scaled dot-product attention)
        
        Returns:
            mask: torch tensor, shape [n_features, n_features]
                 connections are 0, non-connections are -inf
        """
        # Convert boolean mask to additive mask
        mask = torch.zeros((self.n_features, self.n_features), dtype=torch.float32, device=device)
        mask.masked_fill_(self.attention_mask.to(device), float('-inf'))
        return mask

