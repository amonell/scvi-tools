import numpy as np
from scipy.spatial import KDTree
import torch
import scipy.stats as stats
import pandas as pd
import scipy.sparse


def make_neighborhood(adata_subset, adata, n_neighbors=30, spatial_key='X_spatial', no_guide='No guide'):
    """Create neighborhood expression profiles for each cell"""
    
    # Convert to dense array if sparse
    if scipy.sparse.issparse(adata.X):
        adata_arr = adata.X.toarray()
    else:
        adata_arr = adata.X
        
    # Get spatial coordinates
    coords = adata.obsm[spatial_key]
    coords_subset = adata_subset.obsm[spatial_key]
    
    # Build KD tree for efficient nearest neighbor search
    tree = KDTree(coords)
    
    # Find nearest neighbors for each cell in subset
    distances, neighbors = tree.query(coords_subset, k=n_neighbors)
    
    # Calculate mean expression in neighborhoods
    neighborhood_expressions = []
    for cell_neighbors in neighbors:
        # Calculate mean expression of neighbors
        gene_array = np.array(np.mean(adata_arr[cell_neighbors, :], axis=0)).squeeze()
        neighborhood_expressions.append(gene_array)
    
    return np.array(neighborhood_expressions)


def remove_no_guides(adata, key: str = 'guide_rnas', no_guide='NT'):
    """
    Remove cells that were not assigned a guide RNA. Or were assigned in a guide RNA but are in the wrong cluster.
    
    Parameters
    ----------
    key : str
        The key in adata.obs that contains the sgRNA assignment.
    ----------

    Returns
    -------
    adata
    """

    adata = adata[~adata.obs[key].isin([no_guide])]
    return adata

def subset_adata_to_sgRNA_positive(adata, key: str = 'guide_rnas', control_guide: str = 'NT', guide_for_comparison: str = None):
    # Check if guide_for_comparison is None, and throw an error if it is
    if guide_for_comparison is None:
        raise ValueError("guide_for_comparison must be specified.")
    
    # Subset the adata object to include only the control_guide and guide_for_comparison
    adata = adata[adata.obs[key].isin([control_guide, guide_for_comparison])]
    
    # Initialize a list to store the treatment/guide labels
    non_targets = []
    
    # Iterate through the guide_rnas values to categorize them as Control or the specific guide
    for i in adata.obs[key].values:
        if i == control_guide:
            non_targets.append('Control')
        else:
            non_targets.append(i)
    
    # Assign the new categorical treatment labels to a new column in adata.obs
    adata.obs['treatments_guides'] = pd.Categorical(non_targets)

    # Return the modified adata object
    return adata

def get_causal_inputs(adata, control, comparison, n_neighbors=30):
    """Get inputs for causal inference"""
    
    # Get cells for current comparison
    current_comparison = adata[adata.obs['guide_rnas'].isin([control, comparison])]
    
    # Create neighborhood expression profiles
    x = make_neighborhood(current_comparison, adata, n_neighbors=n_neighbors, 
                        spatial_key='X_spatial', no_guide='No guide')
    
    # Get treatment indicators (0 for control, 1 for treated)
    t = np.array(current_comparison.obs['guide_rnas'] == comparison).astype(int)
    
    # Get expression values (convert to dense if sparse)
    if scipy.sparse.issparse(current_comparison.X):
        y = current_comparison.X.toarray()
    else:
        y = current_comparison.X
    
    return x, y, t