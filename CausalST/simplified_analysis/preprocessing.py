import numpy as np
from scipy.spatial import KDTree
import torch
import scipy.stats as stats
import pandas as pd


def make_neighborhood(adata, unsubsetted_adata, n_neighbors:int = 30, spatial_key:str = 'X_spatial', no_guide = 'No guide'):
    adata = adata.copy()

    sgRNA_postive_spatial = np.array([adata.obsm[spatial_key][:, 0], adata.obsm[spatial_key][:, 1]]).T

    unsubsetted = unsubsetted_adata[unsubsetted_adata.obs['guide_rnas'].isin([no_guide])]

    adata_arr = np.array(unsubsetted.X)
    spatial_points = np.array(
        [unsubsetted.obsm[spatial_key][:, 0], unsubsetted.obsm[spatial_key][:, 1]]
    ).T

    list_of_arrays = []
    tree = KDTree(spatial_points)
    for i_bac in range(len(sgRNA_postive_spatial)):
        _, neighbors = tree.query(sgRNA_postive_spatial[i_bac], k=n_neighbors)
        neighbors = list(neighbors)
        gene_array = np.array(np.mean(adata_arr[np.array(neighbors), :], axis=0)).squeeze()
        list_of_arrays.append(stats.zscore(gene_array))
    
    return torch.from_numpy(np.array(list_of_arrays))


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

def get_causal_inputs(adata, control, g_rna, n_neighbors = 30):
    current_comparison = subset_adata_to_sgRNA_positive(adata, control_guide = control, guide_for_comparison=g_rna)
    x = make_neighborhood(current_comparison, adata, n_neighbors = n_neighbors, spatial_key = 'X_spatial', no_guide='Other cells')
    #y = torch.from_numpy(stats.zscore(current_comparison.X, axis = 1))
    y = torch.from_numpy(current_comparison.X)
    t = torch.from_numpy(np.array([0 if i=='Control' else 1 for i in current_comparison.obs['treatments_guides']])).view(-1, 1)
    # Print unique treatment values and their counts
    unique_vals, counts = np.unique(t.numpy(), return_counts=True)
    print("\nTreatment value counts:")
    for val, count in zip(unique_vals, counts):
        print(f"t={val}: {count} samples")
    return x, y, t