import numpy as np
import os
import scanpy as sc
import pandas as pd

def transformation(x, a=0.1, b=0.1, c=0.5, d=2.5, f=4, w=1):
    """Custom biexponential transformation for epithelial distance."""
    x = np.array(x)
    return a * np.exp(b * ((x - w))) - c * np.exp(-d * (x - w)) + f

def create_synthetic_dataset(background_adata_path, 
                           batch,
                           storage_dir,
                           perturbation_group,
                           number_of_cells=1000,
                           n_percent=100,
                           random_seed=0):
    """
    Create synthetic dataset with perturbation effects and save ground truth CATEs.
    
    Args:
        background_adata_path (str): Path to background AnnData file
        batch (str): Batch identifier to use
        storage_dir (str): Base directory for saving results
        perturbation_group (str): Gene to perturb
        number_of_cells (int): Number of cells in each group
        n_percent (float): Percent of genes to alter
        random_seed (int): Random seed for reproducibility
    """
    # Load and filter background data
    background_adata = sc.read(background_adata_path)
    background_adata = background_adata[background_adata.obs['batch'] == batch].copy()
    
    # Transform epithelial distance
    background_adata.obs['epithelial_distance_transformed'] = transformation(
        background_adata.obs['epithelial_distance'].values
    )
    
    # Create random matrices for control and perturbed conditions
    np.random.seed(random_seed)
    random_matrix_ctrl = np.random.normal(0, 1, (2, len(background_adata.var.index.values))).T
    random_matrix_perturb = random_matrix_ctrl.copy()
    
    # Modify perturbed matrix
    total_indices = random_matrix_ctrl.shape[0]
    num_rerandomize = int(n_percent / 100 * total_indices)
    
    np.random.seed(random_seed + 1)
    indices_to_rerandomize = np.random.choice(total_indices, num_rerandomize, replace=False)
    new_random_values = np.random.normal(0, 1, (num_rerandomize, 2))
    random_matrix_perturb[indices_to_rerandomize] = new_random_values
    
    # Set perturbation group values to 0
    perturbation_group_indices = background_adata.var.index == perturbation_group
    random_matrix_perturb[perturbation_group_indices] = -10
    
    # Sample cells
    random_indices = np.random.choice(
        range(len(background_adata.obs.index)),
        number_of_cells,
        replace=False
    )
    
    # Get spatial values
    epithelial_values = background_adata.obs['epithelial_distance_transformed'].values[random_indices]
    cv_values = background_adata.obs['crypt_villi_axis'].values[random_indices]
    
    # Generate synthetic expression data
    def generate_expression(random_matrix, epithelial_values, cv_values):
        synthetic_data = []
        for i in range(len(epithelial_values)):
            gene_expression = []
            for j in range(len(random_matrix)):
                epi_express = random_matrix[j][0] * epithelial_values[i]
                cv_express = random_matrix[j][1] * cv_values[i]
                gene_expression.append(epi_express + cv_express)
            synthetic_data.append(gene_expression)
        return np.array(synthetic_data)
    
    synthetic_cell_by_gene_ctrl = generate_expression(random_matrix_ctrl, epithelial_values, cv_values)
    synthetic_cell_by_gene_perturb = generate_expression(random_matrix_perturb, epithelial_values, cv_values)
    
    # Calculate ground truth CATEs
    cates = synthetic_cell_by_gene_perturb - synthetic_cell_by_gene_ctrl
    
    # Combine data
    full_matrix = np.concatenate([synthetic_cell_by_gene_ctrl, synthetic_cell_by_gene_perturb])
    
    # Create observation annotations
    background_adata.obs['guide_rnas'] = 'No guide'
    full_obs = pd.concat([background_adata.obs.iloc[random_indices], 
                         background_adata.obs.iloc[random_indices]])
    full_obs['guide_rnas'] = ['Ctrl'] * number_of_cells + [perturbation_group] * number_of_cells
    
    # Create synthetic AnnData object
    synthetic_adata = sc.AnnData(
        X=full_matrix,
        obs=full_obs,
        var=pd.DataFrame(index=background_adata.var.index.values)
    )
    
    # Add spatial coordinates
    synthetic_adata.obsm['X_spatial'] = np.append(
        background_adata.obsm['X_spatial'][random_indices],
        background_adata.obsm['X_spatial'][random_indices],
        axis=0
    )
    
    # Combine with background data
    synthetic_adata = sc.concat([synthetic_adata, background_adata])
    
    # Get next dataset number
    dataset_dir = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)
    
    all_files = [x.split('.h5ad')[0] for x in os.listdir(dataset_dir) if x.endswith('.h5ad')]
    all_files = [int(x.split('_')[-1]) for x in all_files if x.split('_')[-1].isdigit()]
    largest_dataset = max(all_files) + 1 if all_files else 0
    
    # Save data
    output_path = os.path.join(dataset_dir, f'synthetic_dataset_{largest_dataset}.h5ad')
    synthetic_adata.write(output_path)
    
    # Log parameters
    parameter_outfile = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 'dataset_log.txt')
    with open(parameter_outfile, 'a') as f:
        f.write(f'{largest_dataset}\t{batch}\t{perturbation_group}\t{number_of_cells}\t{n_percent}\n')
    
    # Save CATEs
    cates_dir = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 'ground_truth_cates')
    os.makedirs(cates_dir, exist_ok=True)
    np.save(os.path.join(cates_dir, f'cates_{largest_dataset}.npy'), cates)
    
    return synthetic_adata, largest_dataset, cates

def main():
    # Parameters
    background_adata_path = '/mnt/sata4/2023_Spatial_Paper_copy_12_4_2024/2023_Spatial_Paper/GitHub/data/adata/timecourse.h5ad'
    batch = 'day8_SI_Ctrl'
    storage_dir = '/mnt/sata2/alex_storage/Desktop/spatial_perturb/CausalST'
    perturbation_group = 'Itgae'
    
    # Create synthetic dataset
    synthetic_adata, dataset_number, cates = create_synthetic_dataset(
        background_adata_path=background_adata_path,
        batch=batch,
        storage_dir=storage_dir,
        perturbation_group=perturbation_group
    )
    
    print(f"Created synthetic dataset {dataset_number}")

if __name__ == "__main__":
    main()