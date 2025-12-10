import numpy as np
import scanpy as sc
import scvi
import scvi.external.resolvi as RESOLVI


def main():
    path_to_reference_adata = (
        "/mnt/sata4/Alex_Xenium_Data/Perturb8/SI/combined/guides_assigned.h5ad"
    )
    perturbation_key = "guide_rnas"
    background_label = "Other cells"
    spatial_embedding_key = "X_cellcharter"

    query_adata = sc.read(path_to_reference_adata)
    query_adata.obsm["X_cellcharter_0"] = np.zeros(
        np.shape(query_adata.obsm[spatial_embedding_key])
    )

    RESOLVI.RESOLVI.setup_anndata(
        query_adata,
        labels_key="resolvi_predicted",
        layer="counts",
        batch_key="batch",
        perturbation_key=perturbation_key,
        control_perturbation="sgCd19",
        background_key=perturbation_key,
        background_category=background_label,
        spatial_embedding_key=spatial_embedding_key,
    )

    model = RESOLVI.RESOLVI(
        query_adata,
        semisupervised=True,
        mixture_k=1,
        n_latent=10,
        override_mixture_k_in_semisupervised=False,
        control_penalty_weight=10000000,
    )

    priors = model.compute_dataset_dependent_priors()
    print(priors)

    model.module.guide.downsample_counts_mean = float(
        model.module.guide.downsample_counts_mean
    )
    model.module.guide.downsample_counts_std = float(
        model.module.guide.downsample_counts_std
    )
    model.train(
        max_epochs=300,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        lr=3e-4,
        train_on_perturbed_only=True,
    )

    non_background_mask = query_adata.obs[perturbation_key] != background_label

    model.store_denoised_layers(
        query_adata,
        control_layer="resolvi_expression_no_shift",
        perturbed_layer="resolvi_expression_with_shift",
        indices=np.where(non_background_mask)[0],
        n_samples=1000,
        batch_size=256,
    )

    model.compute_and_store_raw_shifts(
        query_adata,
        control_layer="resolvi_expression_no_shift",
        perturbed_layer="resolvi_expression_with_shift",
        shift_layer="resolvi_raw_shifts",
        mask=non_background_mask,
    )

    model.compute_and_store_umap(
        query_adata,
        layer="resolvi_expression_no_shift",
        basis_key="X_resolvi_no_shift_umap",
    )
    model.compute_and_store_umap(
        query_adata,
        layer="resolvi_expression_with_shift",
        basis_key="X_resolvi_with_shift_umap",
    )

    genes_for_view = ["guide_rnas", "Klf2", "Perturbation_10", "Perturbation_4"]
    model.plot_resolvi_umap(
        query_adata,
        basis_key="X_resolvi_no_shift_umap",
        color=genes_for_view,
        layer="resolvi_expression_no_shift",
        ncols=2,
    )
    model.plot_resolvi_umap(
        query_adata,
        basis_key="X_resolvi_with_shift_umap",
        color=genes_for_view,
        layer="resolvi_expression_with_shift",
        ncols=2,
    )

    target_gene = "Cxcr3"
    cluster_cellcharter = 4
    target_mask = (query_adata.obs[perturbation_key] == f"sg{target_gene}") & (
        query_adata.obs["cluster_cellcharter"] == cluster_cellcharter
    )

    model.compute_shift_outputs_obsm(
        query_adata,
        obsm_key="resolvi_shift_outputs",
        perturbation_value=f"sg{target_gene}",
        perturbation_key=perturbation_key,
        cluster_key="cluster_cellcharter",
        cluster_value=cluster_cellcharter,
        batch_size=256,
    )

    model.plot_shift_volcano(
        query_adata,
        shift_layer="resolvi_raw_shifts",
        count_layer="counts",
        genes_to_highlight=["Itgae", "Gzma", "Perturbation_4", "Klf2", "Ccl5"],
        mask=target_mask,
        title="Volcano Plot: Shifts vs Log Mean Counts",
    )


if __name__ == "__main__":
    main()
