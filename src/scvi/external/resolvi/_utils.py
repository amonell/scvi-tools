import logging
import warnings
from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pyro import infer

from scvi import settings
from scvi.model._utils import _get_batch_code_from_category, parse_device_args
from scvi.utils import track

logger = logging.getLogger(__name__)


def _safe_log_norm(x: torch.Tensor, dim: int = 1, keepdim: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    Safely compute log1p(x / mean(x)) with numerical stability checks.
    
    Parameters
    ----------
    x
        Input tensor
    dim
        Dimension along which to compute mean
    keepdim
        Whether to keep the dimension
    eps
        Small epsilon to prevent division by zero
        
    Returns
    -------
    Normalized and log-transformed tensor with invalid values replaced by zeros.
    """
    x_mean = torch.mean(x, dim=dim, keepdim=keepdim)
    x_mean_safe = torch.clamp(x_mean, min=eps)
    x_normalized = x / x_mean_safe
    x_log = torch.log1p(x_normalized)
    # Handle any remaining invalid values (NaN, inf)
    return torch.where(torch.isfinite(x_log), x_log, torch.zeros_like(x_log))


class ResolVIPredictiveMixin:
    """Mixin class for generating samples from posterior distribution using infer.predictive."""

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 1,  # consistency, noqa, pylint: disable=unused-argument
        batch_size: int | None = None,
        return_dist: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Return the latent representation for each cell.

        This is denoted as :math:`z` in RESOLVI.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For consistency with scVI, this parameter is ignored.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_dist
            Return the distribution parameters of the latent variables rather than their sampled
            values. If `True`, ignores `give_mean` and `mc_samples`.

        Returns
        -------
        Low-dimensional representation for each cell or a tuple containing its mean and variance.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latent = []
        latent_qzm = []
        latent_qzv = []

        _, _, device = parse_device_args(
            accelerator="auto",
            devices="auto",
            return_device="torch",
            validate_single_device=True,
        )

        for tensors in scdl:
            _, kwargs = self.module._get_fn_args_from_batch(tensors)
            # More robust device transfer that handles all tensor types
            for k, v in kwargs.items():
                if v is not None and hasattr(v, 'to') and hasattr(v, 'device'):
                    kwargs[k] = v.to(device)

            if kwargs["cat_covs"] is not None and self.module.encode_covariates:
                categorical_input = list(torch.split(kwargs["cat_covs"], 1, dim=1))
            else:
                categorical_input = ()

            # Force explicit device transfer right before encoder call
            # Add numerical stability
            x_input = _safe_log_norm(kwargs["x"])
            batch_input = kwargs["batch_index"]
            
            # Ensure all inputs are on the correct device
            x_input = x_input.to(device)
            batch_input = batch_input.to(device)
            categorical_input = [c.to(device) for c in categorical_input]

            # Check and ensure encoder is on correct device
            if next(self.module.z_encoder.parameters()).device != device:
                self.module.z_encoder = self.module.z_encoder.to(device)

            # Also ensure the entire module is on the correct device
            if next(self.module.parameters()).device != device:
                self.module = self.module.to(device)

            qz_m, qz_v, z = self.module.z_encoder(
                x_input,
                batch_input,
                *categorical_input,
            )
            qz = torch.distributions.Normal(qz_m, qz_v.sqrt())
            if give_mean:
                z = qz.loc

            latent += [z.cpu()]
            latent_qzm += [qz.loc.cpu()]
            latent_qzv += [qz.scale.square().cpu()]
        return (
            (torch.cat(latent_qzm).numpy(), torch.cat(latent_qzv).numpy())
            if return_dist
            else torch.cat(latent).numpy()
        )

    @torch.inference_mode()
    def get_normalized_expression_importance(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[int | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | None = 1,
        n_samples: int = 30,
        n_samples_overall: int = None,
        batch_size: int | None = None,
        weights: str | np.ndarray | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        exprs = []
        weighting = []

        _, _, device = parse_device_args(
            accelerator="auto",
            devices="auto",
            return_device="torch",
            validate_single_device=True,
        )

        for tensors in scdl:
            args, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}
            model_now = partial(self.module.model_simplified, corrected_rate=True)
            importance_dist = infer.Importance(
                model_now, guide=self.module.guide.guide_simplified, num_samples=10 * n_samples
            )
            posterior = importance_dist.run(*args, **kwargs)
            marginal = infer.EmpiricalMarginal(posterior, sites=["mean_poisson", "px_scale"])
            samples = torch.cat([marginal().unsqueeze(1) for i in range(n_samples)], 1)
            log_weights = (
                torch.distributions.Poisson(samples[0, ...] + 1e-3)
                .log_prob(kwargs["x"].to(samples.device))
                .sum(-1)
            )
            log_weights = log_weights / kwargs["x"].to(samples.device).sum(-1)
            weighting.append(log_weights.reshape(-1).cpu())
            exprs.append(samples[1, ...].cpu())
        exprs = torch.cat(exprs, axis=1).numpy()
        if return_mean:
            exprs = exprs.mean(0)
        weighting = torch.cat(weighting, axis=0).numpy()
        if library_size is not None:
            exprs = library_size * exprs

        if n_samples_overall is not None:
            # Converts the 3d tensor to a 2d tensor
            exprs = exprs.reshape(-1, exprs.shape[-1])
            n_samples_ = exprs.shape[0]
            if (weights is None) or weights == "uniform":
                p = None
            else:
                weighting -= weighting.max()
                weighting = np.exp(weighting)
                p = weighting / weighting.sum(axis=0, keepdims=True)

            ind_ = np.random.choice(n_samples_, n_samples_overall, p=p, replace=True)
            exprs = exprs[ind_]

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return exprs

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[int | str] | None = None,
        gene_list: Sequence[str] | None = None,
        library_size: float | None = 1,
        n_samples: int = 1,
        n_samples_overall: int = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        silent: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        r"""Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of posterior samples to use for estimation. Overrides `n_samples`.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame
            includes gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults
             to `False`. Otherwise, it defaults to `True`.
        %(de_silent)s

        Returns
        -------
        If `n_samples` is provided and `return_mean` is False,
        this method returns a 3d tensor of shape (n_samples, n_cells, n_genes).
        If `n_samples` is provided and `return_mean` is True, it returns a 2d tensor
        of shape (n_cells, n_genes).
        In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        Otherwise, the method expects `n_samples_overall` to be provided and returns a 2d tensor
        of shape (n_samples_overall, n_genes).
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        exprs = []

        _, _, device = parse_device_args(
            accelerator="auto",
            devices="auto",
            return_device="torch",
            validate_single_device=True,
        )

        for tensors in scdl:
            per_batch_exprs = []
            for batch in track(transform_batch, disable=silent):
                _, kwargs = self.module._get_fn_args_from_batch(tensors)
                kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}

                if kwargs["cat_covs"] is not None and self.module.encode_covariates:
                    categorical_input = list(torch.split(kwargs["cat_covs"], 1, dim=1))
                else:
                    categorical_input = ()

                # Add numerical stability
                x_log = _safe_log_norm(kwargs["x"])
                
                qz_m, qz_v, _ = self.module.z_encoder(
                    x_log,
                    kwargs["batch_index"],
                    *categorical_input,
                )
                z = torch.distributions.Normal(qz_m, qz_v.sqrt()).sample([n_samples])

                if kwargs["cat_covs"] is not None:
                    categorical_input = list(torch.split(kwargs["cat_covs"], 1, dim=1))
                else:
                    categorical_input = ()
                if batch is not None:
                    batch = torch.full_like(kwargs["batch_index"], batch)
                else:
                    batch = kwargs["batch_index"]

                px_scale, _, px_rate, _ = self.module.model.decoder(
                    self.module.model.dispersion, z, kwargs["library"], batch, *categorical_input
                )
                if library_size is not None:
                    exp_ = library_size * px_scale
                else:
                    exp_ = px_rate

                exp_ = exp_[..., gene_mask]
                per_batch_exprs.append(exp_[None].cpu())
            per_batch_exprs = torch.cat(per_batch_exprs, dim=0).mean(0).numpy()
            exprs.append(per_batch_exprs)

        exprs = np.concatenate(exprs, axis=1)
        if return_mean:
            exprs = exprs.mean(0)

        if n_samples_overall is not None:
            # Converts the 3d tensor to a 2d tensor
            exprs = exprs.reshape(-1, exprs.shape[-1])
            n_samples_ = exprs.shape[0]
            ind_ = np.random.choice(n_samples_, n_samples_overall, replace=True)
            exprs = exprs[ind_]

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return exprs

    @torch.inference_mode()
    def get_neighbor_abundance(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        neighbor_key: str | None = None,
        n_samples: int = 1,
        n_samples_overall: int = None,
        batch_size: int | None = None,
        summary_frequency: int = 2,
        weights: str | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        r"""Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        neighbor_key
            Obsm key containing the spatial neighbors of each cell.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of posterior samples to use for estimation. Overrides `n_samples`.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        summary_frequency
            Compute summary_fn after summary_frequency batches. Reduces memory footprint.
        weights
            Spatial weights for each neighbor.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame
            includes gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults
             to `False`. Otherwise, it defaults to `True`.
        kwargs
            Additional keyword arguments that have no effect and only serve for compatibility.

        Returns
        -------
        If `n_samples` is provided and `return_mean` is False,
        this method returns a 3d tensor of shape (n_samples, n_cells, n_celltypes).
        If `n_samples` is provided and `return_mean` is True, it returns a 2d tensor
        of shape (n_cells, n_celltypes).
        In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        Otherwise, the method expects `n_samples_overall` to be provided and returns a 2d tensor
        of shape (n_samples_overall, n_celltypes).
        """
        if adata:
            assert neighbor_key is not None, "Must provide `neighbor_key` if `adata` is provided."
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if neighbor_key is None:
            neighbor_key = self.adata_manager.registry["field_registries"]["index_neighbor"][
                "data_registry"
            ]["attr_key"]
            neighbor_obsm = adata.obsm[neighbor_key]
        else:
            neighbor_obsm = adata.obsm[neighbor_key]
        n_neighbors = neighbor_obsm.shape[-1]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is `False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        if batch_size is not None:
            if batch_size % n_neighbors != 0:
                raise ValueError("Batch size must be divisible by the number of neighbors.")
        batch_size = batch_size if batch_size is not None else n_neighbors * settings.batch_size
        indices_ = neighbor_obsm[indices].reshape(-1)
        dl = self._make_data_loader(
            adata=adata, indices=indices_, shuffle=False, batch_size=batch_size
        )

        sampled_prediction = self.sample_posterior(
            input_dl=dl,
            model=self.module.model_corrected,
            return_sites=["probs_prediction"],
            summary_frequency=summary_frequency,
            num_samples=n_samples,
            return_samples=True,
        )
        flat_neighbor_abundance_ = sampled_prediction["posterior_samples"]["probs_prediction"]
        neighbor_abundance_ = flat_neighbor_abundance_.reshape(
            n_samples, len(indices), n_neighbors, -1
        )
        neighbor_abundance = np.average(neighbor_abundance_, axis=-2, weights=weights)

        if return_mean:
            neighbor_abundance = np.mean(neighbor_abundance, axis=0)

        if n_samples_overall is not None:
            # Converts the 3d tensor to a 2d tensor
            neighbor_abundance = neighbor_abundance.reshape(-1, neighbor_abundance.shape[-1])
            n_samples_ = neighbor_abundance.shape[0]
            ind_ = np.random.choice(n_samples_, n_samples_overall, replace=True)
            neighbor_abundance = neighbor_abundance[ind_]

        if return_numpy is None or return_numpy is False:
            assert return_mean, "Only numpy output is supported when `return_mean` is False."
            n_labels = len(neighbor_abundance[-1])
            return pd.DataFrame(
                neighbor_abundance,
                columns=self._label_mapping[:n_labels],
                index=adata.obs_names[indices],
            )
        else:
            return neighbor_abundance

    @torch.inference_mode()
    def get_perturbation_effects(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        perturbation_list: Sequence[int] | None = None,
        gene_list: Sequence[str] | None = None,
        n_samples: int = 30,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        return_uncertainty: bool = False,
        uncertainty_metrics: Sequence[str] = ("std", "ci_95"),
        alpha: float = 0.05,
        cell_summary_fn: str = "median",
        preserve_cell_heterogeneity: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray | pd.DataFrame | dict:
        """
        Get perturbation effects using the full generative model counterfactual pipeline.
        
        This method:
        1. Uses fixed latent representations from get_latent_representation
        2. Generates counterfactual expressions for each perturbation via the full model
        3. Compares the resulting count distributions (not just rates)
        
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        perturbation_list
            List of perturbation indices to compare. If `None`, uses all non-control perturbations.
        gene_list
            Return effects for a subset of genes. If `None`, all genes are used.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a numpy array instead of a pandas DataFrame.
        return_uncertainty
            Whether to return uncertainty metrics alongside the effects.
        uncertainty_metrics
            Which uncertainty metrics to compute. Options:
            - "std": Standard deviation across samples
            - "var": Variance across samples  
            - "ci_95": 95% credible interval (2.5% and 97.5% percentiles)
            - "ci_90": 90% credible interval (5% and 95% percentiles)
            - "prob_positive": Probability that effect is positive
            - "prob_negative": Probability that effect is negative
            - "prob_significant": Probability that |effect| > threshold
        alpha
            Significance threshold for prob_significant (default 0.05 log2FC ≈ 1.4x fold change)
        cell_summary_fn
            Function to summarize per-cell effects within each posterior sample.
            Options: "median", "mean", "trimmed_mean". 
        preserve_cell_heterogeneity
            If True, computes cell-level uncertainty before pooling. For each posterior sample,
            first computes per-cell log2FC, then summarizes across cells, then computes
            uncertainty across samples. This preserves heterogeneity - genes expressed in
            few cells will have wider uncertainty intervals.
            If False, uses the original approach (average across cells first).
        show_progress
            If True, displays progress bars for batch processing and uncertainty computation.
            
        Returns
        -------
        If return_uncertainty is False:
            Effects as numpy array or DataFrame
        If return_uncertainty is True:
            Dictionary with keys:
            - "effects": Main effects (mean or all samples)
            - "uncertainty": Dictionary of uncertainty metrics
        """
        import pyro
        from pyro import distributions as dist
        from pyro.infer import Predictive
        from scvi import REGISTRY_KEYS
        
        adata = self._validate_anndata(adata)
        
        if indices is None:
            indices = np.arange(adata.n_obs)
            
        # Ensure the entire module is on the correct device before any operations
        _, _, device = parse_device_args(
            "auto", "auto", return_device="torch", validate_single_device=True
        )
        if next(self.module.parameters()).device != device:
            self.module = self.module.to(device)

        # Step 1: Fix latent representations using trained posterior
        z_fixed = self.get_latent_representation(
            adata=adata, indices=indices, give_mean=True, batch_size=batch_size
        )
        
        # Convert to tensor 
        _, _, device = parse_device_args(
            "auto", "auto", return_device="torch", validate_single_device=True
        )
        z_fixed = torch.from_numpy(z_fixed).to(device)
        
        # Check perturbation setup
        perturbation_idx = self.module.model.perturbation_idx
        if perturbation_idx is None:
            raise ValueError(
                "No perturbation found in categorical covariates. "
                "To enable perturbation analysis, you need to:\n"
                "1. Call RESOLVI.setup_anndata() with perturbation_key parameter\n"
                "2. Specify which column in adata.obs contains perturbation categories\n"
                "3. Recreate the RESOLVI model\n\n"
                "Example:\n"
                "RESOLVI.setup_anndata(\n"
                "    adata,\n"
                "    perturbation_key='your_perturbation_column',\n"
                "    control_perturbation='Control'\n"
                ")\n"
                "model = RESOLVI(adata)"
            )
            
        n_perturbs = self.module.model.n_perturbs
        if perturbation_list is None:
            if n_perturbs <= 1:
                raise ValueError(f"Model has {n_perturbs} perturbations, need at least 2 (including control)")
            perturbation_list = list(range(1, n_perturbs))
            
        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)
        
        if n_samples > 1 and return_mean is False and not return_uncertainty:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        # Check what the actual perturbation categories are in the data
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            cat_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            if hasattr(cat_state_registry, 'field_keys'):
                field_keys = cat_state_registry.field_keys
                if perturbation_idx < len(field_keys):
                    perturbation_key = field_keys[perturbation_idx]
                    
                    # Get the actual perturbation values from the original data
                    original_perturb_values = adata.obs[perturbation_key].cat.categories.tolist()
                    
                    # Map category names to indices
                    perturb_name_to_idx = {name: idx for idx, name in enumerate(original_perturb_values)}
                    
                    # The control should be index 0 (first category)
                    control_name = original_perturb_values[0]
                    
                    # Update perturbation_list to use actual meaningful perturbations
                    if perturbation_list == list(range(1, n_perturbs)):
                        # Default case - use all non-control perturbations
                        perturbation_list = list(range(1, len(original_perturb_values)))
                        treatment_names = [original_perturb_values[i] for i in perturbation_list]
        
        # Create data loader for getting other necessary tensors
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        
        all_counterfactual_counts = []
        baseline_perturbation = None  # Will be set from first batch
        
        # Calculate total operations for progress tracking
        n_batches = len(scdl)
        # Note: actual perturbations per batch may vary, but this gives a rough estimate
        estimated_perturbations_per_batch = len(perturbation_list) + 1  # +1 for baseline
        
        # Conditionally add progress tracking
        if show_progress:
            progress_iter = track(
                enumerate(scdl),
                total=n_batches,
                description=f"Processing {n_batches} batches (≈{estimated_perturbations_per_batch} perturbations × {n_samples} samples each)"
            )
        else:
            progress_iter = enumerate(scdl)
        
        for batch_idx, tensors in progress_iter:
            _, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}
            
            batch_size_actual = kwargs["x"].shape[0]
            z_batch = z_fixed[:batch_size_actual, :]  # Use fixed z for this batch
            
            # If all cells are from the same perturbation condition,
            # we should use THAT as our baseline, not force it to 0
            if kwargs["cat_covs"] is not None:
                original_perturbs = kwargs["cat_covs"][:, perturbation_idx]
                actual_baseline_perturb = original_perturbs[0].item()  # All cells should have same perturbation
            else:
                actual_baseline_perturb = 0
            
            # Generate counterfactual counts for each perturbation separately
            perturbation_counts = {}
            
            # Use the ACTUAL baseline perturbation of these cells, plus other perturbations to compare
            baseline_perturb = actual_baseline_perturb
            if baseline_perturbation is None:
                baseline_perturbation = baseline_perturb  # Store for later use
            
            all_perturb_vals = [baseline_perturb] + [p for p in perturbation_list if p != baseline_perturb]
            
            # Track progress within each batch for perturbation computation
            if show_progress:
                perturb_progress = track(
                    enumerate(all_perturb_vals),
                    total=len(all_perturb_vals),
                    description=f"Batch {batch_idx+1}/{n_batches}: Running {len(all_perturb_vals)} perturbations ({n_samples} samples each)",
                    leave=False
                )
            else:
                perturb_progress = enumerate(all_perturb_vals)
            
            for perturb_idx_local, perturb_val in perturb_progress:
                # Create modified cat_covs with this specific perturbation
                if kwargs["cat_covs"] is not None:
                    cat_covs_modified = kwargs["cat_covs"].clone()
                    cat_covs_modified[:, perturbation_idx] = perturb_val
                else:
                    # Create cat_covs if it doesn't exist
                    cat_covs_modified = torch.zeros(
                        (batch_size_actual, perturbation_idx + 1), 
                        device=device, dtype=torch.long
                    )
                    cat_covs_modified[:, perturbation_idx] = perturb_val
                
                # Create model kwargs with modified perturbations
                model_kwargs = kwargs.copy()
                model_kwargs["cat_covs"] = cat_covs_modified
                
                # Create a model that uses the perturbation shift network
                def single_perturbation_model():
                    # Fix the latent representation
                    with pyro.condition(data={"latent": z_batch}):
                        self.module.model_corrected(**model_kwargs)
                
                # Use Predictive to generate samples
                predictive = Predictive(
                    single_perturbation_model,
                    num_samples=n_samples,
                    return_sites=["mean_poisson"]
                )
                
                counterfactual_samples = predictive()
                counts = counterfactual_samples["mean_poisson"]  # [n_samples, batch_size, n_genes]
                
                perturbation_counts[perturb_val] = counts
                
            all_counterfactual_counts.append(perturbation_counts)
        
        # Concatenate results across batches - use the actual baseline perturbation
        baseline_key = baseline_perturbation if baseline_perturbation is not None else 0
        control_counts = torch.cat([batch[baseline_key] for batch in all_counterfactual_counts], dim=1)  # [n_samples, total_cells, n_genes]
        
        perturbation_effects = []
        for perturb_idx in perturbation_list:
            perturb_counts = torch.cat([batch[perturb_idx] for batch in all_counterfactual_counts], dim=1)  # [n_samples, total_cells, n_genes]
            
            # Compute log fold change on actual counts
            eps = 1e-8
            control_safe = torch.clamp(control_counts, min=eps)
            perturb_safe = torch.clamp(perturb_counts, min=eps)
            
            log_fc = torch.log2(perturb_safe / control_safe)  # [n_samples, total_cells, n_genes]
            
            log_fc = log_fc[..., gene_mask]
            
            perturbation_effects.append(log_fc)
        
        # Stack and process results
        perturbation_effects = torch.stack(perturbation_effects, dim=1)  # [n_samples, n_perturbs, n_cells, n_genes]
        
        # Choose how to summarize across cells
        if preserve_cell_heterogeneity:
            # For each posterior sample, compute robust summary across cells
            # This preserves information about cell-to-cell heterogeneity
            if cell_summary_fn == "median":
                perturbation_effects = torch.median(perturbation_effects, dim=2)[0]  # [n_samples, n_perturbs, n_genes]
            elif cell_summary_fn == "mean":
                perturbation_effects = perturbation_effects.mean(dim=2)  # [n_samples, n_perturbs, n_genes]
            elif cell_summary_fn == "trimmed_mean":
                # Remove top and bottom 10% of cells, then take mean
                sorted_effects, _ = torch.sort(perturbation_effects, dim=2)
                n_cells = perturbation_effects.shape[2]
                trim_size = max(1, int(0.1 * n_cells))
                trimmed_effects = sorted_effects[:, :, trim_size:n_cells-trim_size, :]
                perturbation_effects = trimmed_effects.mean(dim=2)  # [n_samples, n_perturbs, n_genes]
            else:
                raise ValueError(f"Unknown cell_summary_fn: {cell_summary_fn}")
        else:
            # Original approach: simple average across cells
            perturbation_effects = perturbation_effects.mean(dim=2)  # [n_samples, n_perturbs, n_genes]
        
        # Compute uncertainty metrics if requested
        uncertainty_results = {}
        if return_uncertainty:
            effects_np = perturbation_effects.cpu().numpy()  # [n_samples, n_perturbs, n_genes]
            
            # Add progress tracking for uncertainty computation
            if show_progress:
                uncertainty_progress = track(
                    uncertainty_metrics,
                    description=f"Computing uncertainty metrics ({len(uncertainty_metrics)} metrics for {effects_np.shape[1]} perturbations × {effects_np.shape[2]} genes)"
                )
            else:
                uncertainty_progress = uncertainty_metrics
            
            for metric in uncertainty_progress:
                if metric == "std":
                    uncertainty_results["std"] = np.std(effects_np, axis=0)  # [n_perturbs, n_genes]
                elif metric == "var":
                    uncertainty_results["var"] = np.var(effects_np, axis=0)  # [n_perturbs, n_genes]
                elif metric == "ci_95":
                    uncertainty_results["ci_95_lower"] = np.percentile(effects_np, 2.5, axis=0)
                    uncertainty_results["ci_95_upper"] = np.percentile(effects_np, 97.5, axis=0)
                elif metric == "ci_90":
                    uncertainty_results["ci_90_lower"] = np.percentile(effects_np, 5, axis=0)
                    uncertainty_results["ci_90_upper"] = np.percentile(effects_np, 95, axis=0)
                elif metric == "prob_positive":
                    uncertainty_results["prob_positive"] = np.mean(effects_np > 0, axis=0)
                elif metric == "prob_negative":
                    uncertainty_results["prob_negative"] = np.mean(effects_np < 0, axis=0)
                elif metric == "prob_significant":
                    # Probability that |effect| > alpha (default 0.05 log2FC ≈ 1.4x fold change)
                    uncertainty_results["prob_significant"] = np.mean(np.abs(effects_np) > alpha, axis=0)
                else:
                    warnings.warn(f"Unknown uncertainty metric: {metric}", UserWarning)
        
        # Convert to numpy for final processing
        perturbation_effects = perturbation_effects.cpu().numpy()
        
        if return_mean:
            perturbation_effects = perturbation_effects.mean(axis=0)  # [n_perturbs, n_genes]

        # Format results
        if return_numpy is None or return_numpy is False:
            gene_names = adata.var_names[gene_mask] if gene_list is not None else adata.var_names
            perturbation_names = [f"perturbation_{i}" for i in perturbation_list]
            
            if return_mean:
                effects_df = pd.DataFrame(
                    perturbation_effects, index=perturbation_names, columns=gene_names
                )
            else:
                effects_df = perturbation_effects
                
            if return_uncertainty:
                # Convert uncertainty metrics to DataFrames too
                uncertainty_dfs = {}
                for metric_name, metric_values in uncertainty_results.items():
                    if return_mean or metric_values.ndim == 2:  # [n_perturbs, n_genes]
                        uncertainty_dfs[metric_name] = pd.DataFrame(
                            metric_values, index=perturbation_names, columns=gene_names
                        )
                    else:
                        uncertainty_dfs[metric_name] = metric_values
                
                return {
                    "effects": effects_df,
                    "uncertainty": uncertainty_dfs
                }
            else:
                return effects_df
        else:
            if return_uncertainty:
                return {
                    "effects": perturbation_effects,
                    "uncertainty": uncertainty_results
                }
            else:
                return perturbation_effects

    @torch.inference_mode()
    def get_perturbation_pvalues(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        perturbation_list: Sequence[int] | None = None,
        gene_list: Sequence[str] | None = None,
        n_samples: int = 100,
        batch_size: int | None = None,
        test_type: str = "two_sided",
        return_numpy: bool | None = None,
        cell_summary_fn: str = "median",
        preserve_cell_heterogeneity: bool = True,
        show_progress: bool = True,
        correct_multiple_testing: bool = True,
        correction_method: str = "fdr_bh", 
        alpha: float = 0.05,
    ) -> pd.DataFrame | np.ndarray:
        """
        Compute empirical p-values for perturbation effects with optional multiple testing correction.
        
        This method computes p-values by treating posterior samples as replicates
        and testing whether the log fold change is significantly different from zero.
        
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        perturbation_list
            List of perturbation indices to compare. If `None`, uses all non-control perturbations.
        gene_list
            Return p-values for a subset of genes. If `None`, all genes are used.
        n_samples
            Number of posterior samples to use for p-value estimation. More samples = more precise p-values.
        batch_size
            Minibatch size for data loading into model.
        test_type
            Type of test to perform:
            - "two_sided": Test if effect != 0
            - "greater": Test if effect > 0 (upregulation)
            - "less": Test if effect < 0 (downregulation)
        return_numpy
            Return a numpy array instead of a pandas DataFrame.
        cell_summary_fn
            Function to summarize per-cell effects within each posterior sample.
            Options: "median", "mean", "trimmed_mean".
        preserve_cell_heterogeneity
            If True, computes cell-level uncertainty before pooling for more robust p-values.
        show_progress
            If True, displays progress bars for effect computation and p-value calculation.
        correct_multiple_testing
            If True, applies multiple testing correction. **Highly recommended for genomics data**.
        correction_method
            Multiple testing correction method. Options:
            - "fdr_bh": Benjamini-Hochberg FDR (recommended for genomics)
            - "fdr_by": Benjamini-Yekutieli FDR (for dependent tests)
            - "bonferroni": Bonferroni correction (very conservative)
            - "holm": Holm-Bonferroni (less conservative than Bonferroni)
        alpha
            Significance threshold for multiple testing correction.
            
        Returns
        -------
        If correct_multiple_testing=False:
            P-values for each perturbation-gene combination (raw, uncorrected).
        If correct_multiple_testing=True:
            Dictionary containing:
            - "pvalues_raw": Raw uncorrected p-values
            - "pvalues_corrected": Multiple testing corrected p-values  
            - "is_significant": Boolean mask of genes passing correction at alpha level
            - "correction_method": Method used for correction
            - "alpha": Significance threshold used
            - "n_significant": Number of significant genes after correction
        """
        from scipy import stats
        
        # Get all posterior samples (not just mean)
        results = self.get_perturbation_effects(
            adata=adata,
            indices=indices,
            perturbation_list=perturbation_list,
            gene_list=gene_list,
            n_samples=n_samples,
            batch_size=batch_size,
            return_mean=False,
            return_numpy=True,
            return_uncertainty=False,
            cell_summary_fn=cell_summary_fn,
            preserve_cell_heterogeneity=preserve_cell_heterogeneity,
            show_progress=show_progress,
        )
        
        # results shape: [n_samples, n_perturbs, n_genes]
        n_samples_actual, n_perturbs, n_genes = results.shape
        
        # Compute p-values for each perturbation-gene combination
        pvalues = np.zeros((n_perturbs, n_genes))
        
        # Create progress tracking for p-value computation
        total_tests = list(range(n_perturbs * n_genes))
        if show_progress:
            progress_iter = track(
                total_tests, 
                description=f"Computing p-values ({n_perturbs} perturbations × {n_genes} genes)"
            )
        else:
            progress_iter = total_tests
        
        for test_idx in progress_iter:
            perturb_idx = test_idx // n_genes
            gene_idx = test_idx % n_genes
            
            samples = results[:, perturb_idx, gene_idx]
            
            # One-sample t-test against null hypothesis that mean = 0
            if test_type == "two_sided":
                _, pval = stats.ttest_1samp(samples, 0, alternative='two-sided')
            elif test_type == "greater":
                _, pval = stats.ttest_1samp(samples, 0, alternative='greater')
            elif test_type == "less":
                _, pval = stats.ttest_1samp(samples, 0, alternative='less')
            else:
                raise ValueError(f"Unknown test_type: {test_type}")
            
            pvalues[perturb_idx, gene_idx] = pval
        
        # Format results
        adata = self._validate_anndata(adata)
        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)
        gene_names = adata.var_names[gene_mask] if gene_list is not None else adata.var_names
        
        # Get perturbation names
        if perturbation_list is None:
            n_perturbs_total = self.module.model.n_perturbs
            perturbation_list = list(range(1, n_perturbs_total))
        
        perturbation_names = [f"perturbation_{i}" for i in perturbation_list]
        
        if return_numpy is None or return_numpy is False:
            pvalues_df = pd.DataFrame(
                pvalues, index=perturbation_names, columns=gene_names
            )
        else:
            pvalues_df = pvalues
            
        # Apply multiple testing correction if requested
        if correct_multiple_testing:
            if show_progress:
                print(f"Applying multiple testing correction ({correction_method})...")
                
            corrected_results = self.correct_pvalues(
                pvalues_df, 
                method=correction_method, 
                alpha=alpha,
                return_numpy=return_numpy
            )
            
            if show_progress:
                n_significant = corrected_results.get('n_significant', 0)
                print(f"Found {n_significant} significant genes after {correction_method} correction at α={alpha}")
                
            return corrected_results
        else:
            if show_progress:
                print("⚠️  WARNING: No multiple testing correction applied!")
                print("   For genomics data, consider setting correct_multiple_testing=True")
                
            return pvalues_df

    def correct_pvalues(
        self,
        pvalues: pd.DataFrame | np.ndarray,
        method: str = "fdr_bh",
        alpha: float = 0.05,
        return_numpy: bool = False,
    ) -> dict:
        """
        Apply multiple testing correction to p-values.
        
        This is a utility function that can be used to correct p-values from any source,
        not just get_perturbation_pvalues().
        
        Parameters
        ----------
        pvalues
            P-values to correct. Can be DataFrame or numpy array.
        method
            Multiple testing correction method:
            - "fdr_bh": Benjamini-Hochberg FDR (recommended)
            - "fdr_by": Benjamini-Yekutieli FDR (for dependent tests)
            - "bonferroni": Bonferroni correction
            - "holm": Holm-Bonferroni correction
        alpha
            Significance threshold.
        return_numpy
            Return numpy arrays instead of DataFrames.
            
        Returns
        -------
        Dictionary containing:
        - "pvalues_raw": Original uncorrected p-values
        - "pvalues_corrected": Corrected p-values
        - "is_significant": Boolean mask of significant results
        - "correction_method": Method used
        - "alpha": Threshold used
        - "n_significant": Number of significant results
        """
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            raise ImportError(
                "statsmodels is required for multiple testing correction. "
                "Install with: pip install statsmodels"
            )
        
        # Handle input format
        if isinstance(pvalues, pd.DataFrame):
            pvals_flat = pvalues.values.flatten()
            original_shape = pvalues.shape
            index = pvalues.index
            columns = pvalues.columns
        else:
            pvals_flat = pvalues.flatten()
            original_shape = pvalues.shape
            index = None
            columns = None
        
        # Apply correction
        reject, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(
            pvals_flat, alpha=alpha, method=method
        )
        
        # Reshape back to original format
        pvals_corrected = pvals_corrected.reshape(original_shape)
        reject = reject.reshape(original_shape)
        
        # Format output
        if not return_numpy and index is not None and columns is not None:
            # DataFrame output
            results = {
                "pvalues_raw": pvalues,
                "pvalues_corrected": pd.DataFrame(
                    pvals_corrected, index=index, columns=columns
                ),
                "is_significant": pd.DataFrame(
                    reject, index=index, columns=columns
                ),
            }
        else:
            # Numpy output
            results = {
                "pvalues_raw": pvalues if isinstance(pvalues, np.ndarray) else pvalues.values,
                "pvalues_corrected": pvals_corrected,
                "is_significant": reject,
            }
        
        # Add metadata
        results.update({
            "correction_method": method,
            "alpha": alpha,
            "n_significant": int(reject.sum()),
            "alpha_sidak": alpha_sidak,
            "alpha_bonferroni": alpha_bonf,
        })
        
        return results

    @torch.inference_mode()
    def analyze_perturbation_heterogeneity(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        perturbation_list: Sequence[int] | None = None,
        gene_list: Sequence[str] | None = None,
        n_samples: int = 10,
        batch_size: int | None = None,
        return_numpy: bool | None = None,
    ) -> pd.DataFrame | dict:
        """
        Analyze cell-level heterogeneity in perturbation responses.
        
        This method helps understand why certain genes have wider uncertainty intervals
        by examining the distribution of per-cell effects across posterior samples.
        
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        perturbation_list
            List of perturbation indices to compare. If `None`, uses all non-control perturbations.
        gene_list
            Return analysis for a subset of genes. If `None`, all genes are used.
        n_samples
            Number of posterior samples to use for analysis.
        batch_size
            Minibatch size for data loading into model.
        return_numpy
            Return numpy arrays instead of pandas DataFrames.
            
        Returns
        -------
        Dictionary containing heterogeneity metrics:
        - "cell_response_variability": CV of per-cell effects within each sample
        - "fraction_responding_cells": Fraction of cells with |effect| > 0.1
        - "median_vs_mean_difference": Difference between median and mean per-cell effects
        - "outlier_fraction": Fraction of cells with extreme effects (>2 SD from median)
        """
        import pyro
        from pyro import distributions as dist
        from pyro.infer import Predictive
        from scvi import REGISTRY_KEYS
        
        adata = self._validate_anndata(adata)
        
        if indices is None:
            indices = np.arange(adata.n_obs)
            
        # Ensure the entire module is on the correct device before any operations
        _, _, device = parse_device_args(
            "auto", "auto", return_device="torch", validate_single_device=True
        )
        if next(self.module.parameters()).device != device:
            self.module = self.module.to(device)

        # Get per-cell effects without summarizing
        z_fixed = self.get_latent_representation(
            adata=adata, indices=indices, give_mean=True, batch_size=batch_size
        )
        z_fixed = torch.from_numpy(z_fixed).to(device)
        
        # Check perturbation setup
        perturbation_idx = self.module.model.perturbation_idx
        if perturbation_idx is None:
            raise ValueError("No perturbation found in categorical covariates.")
            
        n_perturbs = self.module.model.n_perturbs
        if perturbation_list is None:
            if n_perturbs <= 1:
                raise ValueError(f"Model has {n_perturbs} perturbations, need at least 2 (including control)")
            perturbation_list = list(range(1, n_perturbs))
            
        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)
        
        # Create data loader for getting other necessary tensors
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        
        all_counterfactual_counts = []
        baseline_perturbation = None
        
        for tensors in scdl:
            _, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}
            
            batch_size_actual = kwargs["x"].shape[0]
            z_batch = z_fixed[:batch_size_actual, :]
            
            if kwargs["cat_covs"] is not None:
                original_perturbs = kwargs["cat_covs"][:, perturbation_idx]
                actual_baseline_perturb = original_perturbs[0].item()
            else:
                actual_baseline_perturb = 0
            
            perturbation_counts = {}
            baseline_perturb = actual_baseline_perturb
            if baseline_perturbation is None:
                baseline_perturbation = baseline_perturb
            
            all_perturb_vals = [baseline_perturb] + [p for p in perturbation_list if p != baseline_perturb]
            
            for perturb_val in all_perturb_vals:
                if kwargs["cat_covs"] is not None:
                    cat_covs_modified = kwargs["cat_covs"].clone()
                    cat_covs_modified[:, perturbation_idx] = perturb_val
                else:
                    cat_covs_modified = torch.zeros(
                        (batch_size_actual, perturbation_idx + 1), 
                        device=device, dtype=torch.long
                    )
                    cat_covs_modified[:, perturbation_idx] = perturb_val
                
                model_kwargs = kwargs.copy()
                model_kwargs["cat_covs"] = cat_covs_modified
                
                def single_perturbation_model():
                    with pyro.condition(data={"latent": z_batch}):
                        self.module.model_corrected(**model_kwargs)
                
                predictive = Predictive(
                    single_perturbation_model,
                    num_samples=n_samples,
                    return_sites=["mean_poisson"]
                )
                
                counterfactual_samples = predictive()
                counts = counterfactual_samples["mean_poisson"]  # [n_samples, batch_size, n_genes]
                
                perturbation_counts[perturb_val] = counts
                
            all_counterfactual_counts.append(perturbation_counts)
        
        # Concatenate results across batches
        baseline_key = baseline_perturbation if baseline_perturbation is not None else 0
        control_counts = torch.cat([batch[baseline_key] for batch in all_counterfactual_counts], dim=1)
        
        heterogeneity_metrics = {}
        
        for perturb_idx in perturbation_list:
            perturb_counts = torch.cat([batch[perturb_idx] for batch in all_counterfactual_counts], dim=1)
            
            # Compute log fold change per cell per sample
            eps = 1e-8
            control_safe = torch.clamp(control_counts, min=eps)
            perturb_safe = torch.clamp(perturb_counts, min=eps)
            
            log_fc = torch.log2(perturb_safe / control_safe)  # [n_samples, n_cells, n_genes]
            log_fc = log_fc[..., gene_mask]
            log_fc_np = log_fc.cpu().numpy()
            
            # Compute heterogeneity metrics for each gene
            n_genes = log_fc_np.shape[-1]
            
            # 1. Coefficient of variation of per-cell effects within each sample
            cv_per_sample = []
            for sample_idx in range(n_samples):
                sample_effects = log_fc_np[sample_idx, :, :]  # [n_cells, n_genes]
                cv = np.abs(np.std(sample_effects, axis=0) / (np.mean(sample_effects, axis=0) + eps))
                cv_per_sample.append(cv)
            cv_per_sample = np.array(cv_per_sample)  # [n_samples, n_genes]
            mean_cv = np.mean(cv_per_sample, axis=0)  # [n_genes]
            
            # 2. Fraction of cells responding (|effect| > 0.1 log2FC)
            responding_cells = []
            for sample_idx in range(n_samples):
                sample_effects = log_fc_np[sample_idx, :, :]  # [n_cells, n_genes]
                frac_responding = np.mean(np.abs(sample_effects) > 0.1, axis=0)  # [n_genes]
                responding_cells.append(frac_responding)
            responding_cells = np.array(responding_cells)  # [n_samples, n_genes]
            mean_responding = np.mean(responding_cells, axis=0)  # [n_genes]
            
            # 3. Difference between median and mean (measure of skewness)
            median_vs_mean_diff = []
            for sample_idx in range(n_samples):
                sample_effects = log_fc_np[sample_idx, :, :]  # [n_cells, n_genes]
                medians = np.median(sample_effects, axis=0)  # [n_genes]
                means = np.mean(sample_effects, axis=0)  # [n_genes]
                diff = np.abs(medians - means)  # [n_genes]
                median_vs_mean_diff.append(diff)
            median_vs_mean_diff = np.array(median_vs_mean_diff)  # [n_samples, n_genes]
            mean_median_mean_diff = np.mean(median_vs_mean_diff, axis=0)  # [n_genes]
            
            # 4. Fraction of outlier cells (>2 SD from median)
            outlier_fractions = []
            for sample_idx in range(n_samples):
                sample_effects = log_fc_np[sample_idx, :, :]  # [n_cells, n_genes]
                medians = np.median(sample_effects, axis=0)  # [n_genes]
                mads = np.median(np.abs(sample_effects - medians[None, :]), axis=0)  # [n_genes]
                # Use MAD-based outlier detection (more robust than SD)
                outliers = np.abs(sample_effects - medians[None, :]) > (2.5 * mads[None, :])  # [n_cells, n_genes]
                frac_outliers = np.mean(outliers, axis=0)  # [n_genes]
                outlier_fractions.append(frac_outliers)
            outlier_fractions = np.array(outlier_fractions)  # [n_samples, n_genes]
            mean_outlier_frac = np.mean(outlier_fractions, axis=0)  # [n_genes]
            
            heterogeneity_metrics[f"perturbation_{perturb_idx}"] = {
                "cell_response_variability": mean_cv,
                "fraction_responding_cells": mean_responding,
                "median_vs_mean_difference": mean_median_mean_diff,
                "outlier_fraction": mean_outlier_frac,
            }
        
        # Format results
        if return_numpy is None or return_numpy is False:
            gene_names = adata.var_names[gene_mask] if gene_list is not None else adata.var_names
            
            formatted_results = {}
            for perturb_name, metrics in heterogeneity_metrics.items():
                formatted_results[perturb_name] = {}
                for metric_name, metric_values in metrics.items():
                    formatted_results[perturb_name][metric_name] = pd.Series(
                        metric_values, index=gene_names, name=metric_name
                    )
            
            return formatted_results
        else:
            return heterogeneity_metrics

    @torch.inference_mode()
    def get_perturbation_de(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        perturbation_list: Sequence[int] | None = None,
        gene_list: Sequence[str] | None = None,
        mode: str = "change",
        delta: float = 0.25,
        fdr_target: float = 0.05,
        n_samples: int = 200,
        batch_size: int | None = None,
        cell_summary_fn: str = "median",
        preserve_cell_heterogeneity: bool = True,
        use_mixture_posterior: bool = True,
        show_progress: bool = True,
        return_numpy: bool = False,
    ) -> pd.DataFrame:
        """
        Perform Bayesian differential expression testing for perturbations.
        
        This method implements the same statistical framework as scVI's differential_expression()
        but adapted for perturbation analysis using counterfactual inference.
        
        Implements two testing modes:
        1. "vanilla": H0: βg = 0 vs H1: βg ≠ 0 (point-null hypothesis)
        2. "change": H0: |βg| ≤ δ vs H1: |βg| > δ (thresholded-null hypothesis)
        
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        perturbation_list
            List of perturbation indices to compare. If `None`, uses all non-control perturbations.
        gene_list
            Return DE results for a subset of genes. If `None`, all genes are used.
        mode
            Testing mode:
            - "vanilla": Point-null hypothesis testing using Bayes factors
            - "change": Thresholded-null hypothesis using probability of significant change
        delta
            Effect size threshold for change mode (log2 fold change units).
        fdr_target
            Target false discovery rate for Bayesian FDR control.
        n_samples
            Number of posterior samples for Bayesian testing. More samples = more precise estimates.
        batch_size
            Minibatch size for data loading into model.
        cell_summary_fn
            Function to summarize per-cell effects within each posterior sample.
            Options: "median", "mean", "trimmed_mean".
        preserve_cell_heterogeneity
            If True, computes cell-level effects before pooling for more robust testing.
        use_mixture_posterior
            If True, uses mixture posterior sampling like scVI. If False, uses fixed latent representations.
        show_progress
            If True, displays progress bars for effect computation and testing.
        return_numpy
            Return numpy arrays instead of pandas DataFrames.
            
        Returns
        -------
        DataFrame with differential expression results containing:
        - perturbation: Perturbation identifier
        - gene: Gene identifier  
        - effect_size: Posterior mean log2 fold change
        - effect_std: Posterior standard deviation of log2 fold change
        - effect_median: Posterior median log2 fold change
        - bayes_factor: Log Bayes factor (vanilla mode only)
        - proba_de: Probability of differential expression (change mode only)
        - proba_positive: Probability that effect > 0
        - proba_negative: Probability that effect < 0
        - is_de_fdr: Whether gene passes Bayesian FDR control at target level
        """
        if mode not in ["vanilla", "change"]:
            raise ValueError(f"mode must be 'vanilla' or 'change', got {mode}")
            
        adata = self._validate_anndata(adata)
        
        if indices is None:
            indices = np.arange(adata.n_obs)
            
        # Check perturbation setup
        perturbation_idx = self.module.model.perturbation_idx
        if perturbation_idx is None:
            raise ValueError(
                "No perturbation found in categorical covariates. "
                "To enable perturbation analysis, you need to:\n"
                "1. Call RESOLVI.setup_anndata() with perturbation_key parameter\n"
                "2. Specify which column in adata.obs contains perturbation categories\n"
                "3. Recreate the RESOLVI model"
            )
            
        n_perturbs = self.module.model.n_perturbs
        if perturbation_list is None:
            if n_perturbs <= 1:
                raise ValueError(f"Model has {n_perturbs} perturbations, need at least 2 (including control)")
            perturbation_list = list(range(1, n_perturbs))
            
        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)
        
        # Get posterior samples of log fold changes
        if show_progress:
            print(f"Computing posterior samples for Bayesian testing (mode={mode})...")
            
        if use_mixture_posterior:
            # Use mixture posterior sampling like scVI (more rigorous)
            effects = self._get_perturbation_effects_mixture_posterior(
                adata=adata,
                indices=indices,
                perturbation_list=perturbation_list,
                gene_list=gene_list,
                n_samples=n_samples,
                batch_size=batch_size,
                cell_summary_fn=cell_summary_fn,
                preserve_cell_heterogeneity=preserve_cell_heterogeneity,
                show_progress=show_progress,
            )
        else:
            # Use existing fixed latent approach (faster but less rigorous)
            effects = self.get_perturbation_effects(
                adata=adata,
                indices=indices,
                perturbation_list=perturbation_list,
                gene_list=gene_list,
                n_samples=n_samples,
                batch_size=batch_size,
                return_mean=False,
                return_numpy=True,
                return_uncertainty=False,
                cell_summary_fn=cell_summary_fn,
                preserve_cell_heterogeneity=preserve_cell_heterogeneity,
                show_progress=show_progress,
            )
        
        # effects shape: [n_samples, n_perturbs, n_genes]
        if show_progress:
            print(f"Running Bayesian hypothesis testing ({mode} mode)...")
            
        results = []
        n_samples_actual, n_perturbs_actual, n_genes = effects.shape
        
        # Get gene and perturbation names
        if gene_list is not None:
            gene_names = adata.var_names[adata.var_names.isin(gene_list)].tolist()
        else:
            gene_names = adata.var_names.tolist()
            
        perturbation_names = [f"perturbation_{i}" for i in perturbation_list]
        
        # Progress tracking for testing
        total_tests = n_perturbs_actual * n_genes
        if show_progress:
            progress_iter = track(
                range(total_tests),
                description=f"Bayesian testing ({mode} mode): {n_perturbs_actual} perturbations × {n_genes} genes"
            )
        else:
            progress_iter = range(total_tests)
            
        for test_idx in progress_iter:
            perturb_idx = test_idx // n_genes
            gene_idx = test_idx % n_genes
            
            # Get posterior samples for this perturbation-gene combination
            samples = effects[:, perturb_idx, gene_idx]  # [n_samples]
            
            # Basic effect statistics
            effect_mean = np.mean(samples)
            effect_std = np.std(samples)
            effect_median = np.median(samples)
            
            # Probability computations
            prob_positive = np.mean(samples > 0)
            prob_negative = np.mean(samples < 0)
            
            result = {
                'perturbation': perturbation_names[perturb_idx],
                'gene': gene_names[gene_idx],
                'effect_size': effect_mean,
                'effect_std': effect_std,
                'effect_median': effect_median,
                'proba_positive': prob_positive,
                'proba_negative': prob_negative,
            }
            
            if mode == "vanilla":
                # Vanilla mode: H0: βg = 0 vs H1: βg ≠ 0
                # Compute log Bayes factor: log(P(βg > 0) / P(βg ≤ 0))
                if prob_negative > 1e-10:  # Avoid division by zero
                    bayes_factor = np.log(prob_positive / prob_negative)
                else:
                    bayes_factor = np.log(prob_positive / 1e-10)  # Clip for numerical stability
                    
                result['bayes_factor'] = bayes_factor
                
            elif mode == "change":
                # Change mode: H0: |βg| ≤ δ vs H1: |βg| > δ  
                # Compute P(|βg| > δ | data)
                prob_de = np.mean(np.abs(samples) > delta)
                result['proba_de'] = prob_de
                result['delta'] = delta
            
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Apply Bayesian FDR control
        if show_progress:
            print(f"Applying Bayesian FDR control (target={fdr_target})...")
            
        if mode == "change":
            # Bayesian FDR control for change mode
            df = df.sort_values('proba_de', ascending=False).reset_index(drop=True)
            
            # Bayesian FDR: find largest k such that Σ(1-pf)/k ≤ α
            proba_de_vals = df['proba_de'].values
            cumsum_fdr = np.cumsum(1 - proba_de_vals) / np.arange(1, len(df) + 1)
            
            # Find the largest k where FDR condition is satisfied
            passing_indices = np.where(cumsum_fdr <= fdr_target)[0]
            if len(passing_indices) > 0:
                num_de = passing_indices[-1] + 1  # +1 because indices are 0-based
            else:
                num_de = 0
            
            # Mark significant genes
            df['is_de_fdr'] = False
            if num_de > 0:
                df.iloc[:num_de, df.columns.get_loc('is_de_fdr')] = True
                
            df['fdr_target'] = fdr_target
            df['n_de_fdr'] = num_de
            
        elif mode == "vanilla":
            # For vanilla mode, sort by absolute Bayes factor
            df['abs_bayes_factor'] = np.abs(df['bayes_factor'])
            df = df.sort_values('abs_bayes_factor', ascending=False).reset_index(drop=True)
            
            # Simple threshold-based approach for vanilla mode
            # (Could implement more sophisticated Bayesian FDR for Bayes factors)
            bf_threshold = np.log(3)  # log(3) ≈ 1.1 corresponds to "substantial evidence"
            df['is_de_bf'] = np.abs(df['bayes_factor']) > bf_threshold
            df['bf_threshold'] = bf_threshold
            
        # Add summary statistics
        if show_progress:
            if mode == "change":
                n_significant = df['is_de_fdr'].sum()
                print(f"Found {n_significant} DE genes at FDR {fdr_target}")
            else:
                n_significant = df['is_de_bf'].sum()
                print(f"Found {n_significant} DE genes with |BF| > {bf_threshold:.2f}")
                
        # Add metadata
        df['mode'] = mode
        df['n_samples'] = n_samples_actual
        df['method'] = 'bayesian_perturbation_de'
        
        if return_numpy:
            return df.values
        else:
            return df

    def _get_perturbation_effects_mixture_posterior(
        self,
        adata: AnnData,
        indices: Sequence[int],
        perturbation_list: Sequence[int],
        gene_list: Sequence[str] | None = None,
        n_samples: int = 200,
        batch_size: int | None = None,
        cell_summary_fn: str = "median",
        preserve_cell_heterogeneity: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Get perturbation effects using mixture posterior sampling like scVI.
        
        Instead of using fixed latent representations, this method:
        1. Creates mixture posteriors for each condition like scVI: P̂_A(Z) = (1/|N_A|) Σ q_θ(z|x_n)
        2. Samples paired latent representations from control vs perturbation conditions
        3. Uses these for counterfactual inference
        
        This provides more rigorous uncertainty quantification by properly accounting
        for variability in the latent representations.
        
        Returns
        -------
        Array of shape [n_samples, n_perturbs, n_genes] with log fold change samples
        """
        import pyro
        from pyro.infer import Predictive
        from scvi import REGISTRY_KEYS
        
        # Ensure device compatibility
        _, _, device = parse_device_args(
            "auto", "auto", return_device="torch", validate_single_device=True
        )
        if next(self.module.parameters()).device != device:
            self.module = self.module.to(device)
            
        gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)
        perturbation_idx = self.module.model.perturbation_idx
        
        # Get variational posteriors for all cells 
        qz_means, qz_vars = self.get_latent_representation(
            adata=adata, indices=indices, return_dist=True, batch_size=batch_size
        )
        qz_means = torch.from_numpy(qz_means).to(device)
        qz_vars = torch.from_numpy(qz_vars).to(device)
        
        # Identify which cells belong to which perturbation condition
        control_indices = np.arange(len(indices))  # Default: use all cells
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            cat_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            if hasattr(cat_state_registry, 'field_keys'):
                field_keys = cat_state_registry.field_keys
                if perturbation_idx < len(field_keys):
                    perturbation_key = field_keys[perturbation_idx]
                    cell_perturbations = adata.obs.iloc[indices][perturbation_key].values
                    
                    # Find control condition (should be index 0)
                    control_mask = cell_perturbations == 0
                    control_indices = np.where(control_mask)[0]
                    
                    if len(control_indices) == 0:
                        # If no explicit control cells, use the most common condition as baseline
                        unique_vals, counts = np.unique(cell_perturbations, return_counts=True)
                        control_val = unique_vals[np.argmax(counts)]
                        control_mask = cell_perturbations == control_val
                        control_indices = np.where(control_mask)[0]
        
        # Create data loader for getting model tensors
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        
        all_effects = []
        
        # Progress tracking
        if show_progress:
            batch_progress = track(
                enumerate(scdl),
                total=len(scdl),
                description=f"Mixture posterior sampling: {len(scdl)} batches × {n_samples} samples"
            )
        else:
            batch_progress = enumerate(scdl)
            
        for batch_idx, tensors in batch_progress:
            _, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}
            
            batch_size_actual = kwargs["x"].shape[0]
            
            # For each perturbation, generate counterfactual expressions
            batch_effects = []
            
            for perturb_val in perturbation_list:
                # Collect log fold changes across samples for this perturbation
                perturbation_log_fcs = []
                
                for sample_idx in range(n_samples):
                    # Sample latent representations from mixture posteriors for this batch
                    control_latents = []
                    perturb_latents = []
                    
                    for cell_idx in range(batch_size_actual):
                        # Sample from control mixture posterior
                        if len(control_indices) > 0:
                            # Randomly select a control cell from the full dataset
                            control_cell_idx = np.random.choice(control_indices)
                            qz_m_control = qz_means[control_cell_idx]
                            qz_v_control = qz_vars[control_cell_idx]
                            z_control = torch.distributions.Normal(qz_m_control, qz_v_control.sqrt()).sample()
                        else:
                            # Fallback: sample from random cell
                            random_cell_idx = np.random.choice(len(qz_means))
                            qz_m = qz_means[random_cell_idx]
                            qz_v = qz_vars[random_cell_idx]
                            z_control = torch.distributions.Normal(qz_m, qz_v.sqrt()).sample()
                        
                        # For perturbation condition, use same latent (will be modified by perturbation network)
                        z_perturb = z_control.clone()
                        
                        control_latents.append(z_control)
                        perturb_latents.append(z_perturb)
                    
                    control_latents = torch.stack(control_latents)  # [batch_size, n_latent]
                    perturb_latents = torch.stack(perturb_latents)  # [batch_size, n_latent]
                    
                    # Generate counterfactual expressions for control condition
                    control_kwargs = kwargs.copy()
                    if control_kwargs["cat_covs"] is not None:
                        control_kwargs["cat_covs"] = control_kwargs["cat_covs"].clone()
                        control_kwargs["cat_covs"][:, perturbation_idx] = 0  # Control condition
                    else:
                        control_kwargs["cat_covs"] = torch.zeros(
                            (batch_size_actual, perturbation_idx + 1), 
                            device=device, dtype=torch.long
                        )
                    
                    def control_model():
                        with pyro.condition(data={"latent": control_latents}):
                            self.module.model_corrected(**control_kwargs)
                    
                    control_predictive = Predictive(
                        control_model, num_samples=1, return_sites=["mean_poisson"]
                    )
                    control_samples = control_predictive()
                    control_counts = control_samples["mean_poisson"][0]  # [1, batch_size, n_genes]
                    control_counts = control_counts.squeeze(0)  # [batch_size, n_genes]
                    
                    # Generate counterfactual expressions for perturbation condition  
                    perturb_kwargs = kwargs.copy()
                    if perturb_kwargs["cat_covs"] is not None:
                        perturb_kwargs["cat_covs"] = perturb_kwargs["cat_covs"].clone()
                        perturb_kwargs["cat_covs"][:, perturbation_idx] = perturb_val
                    else:
                        perturb_kwargs["cat_covs"] = torch.zeros(
                            (batch_size_actual, perturbation_idx + 1), 
                            device=device, dtype=torch.long
                        )
                        perturb_kwargs["cat_covs"][:, perturbation_idx] = perturb_val
                    
                    def perturb_model():
                        with pyro.condition(data={"latent": perturb_latents}):
                            self.module.model_corrected(**perturb_kwargs)
                    
                    perturb_predictive = Predictive(
                        perturb_model, num_samples=1, return_sites=["mean_poisson"]
                    )
                    perturb_samples = perturb_predictive()
                    perturb_counts = perturb_samples["mean_poisson"][0]  # [1, batch_size, n_genes]
                    perturb_counts = perturb_counts.squeeze(0)  # [batch_size, n_genes]
                    
                    # Compute log fold changes
                    eps = 1e-8
                    control_safe = torch.clamp(control_counts, min=eps)
                    perturb_safe = torch.clamp(perturb_counts, min=eps)
                    
                    log_fc = torch.log2(perturb_safe / control_safe)  # [batch_size, n_genes]
                    log_fc = log_fc[..., gene_mask]
                    
                    # Summarize across cells based on cell_summary_fn
                    if preserve_cell_heterogeneity:
                        if cell_summary_fn == "median":
                            log_fc_summary = torch.median(log_fc, dim=0)[0]  # [n_genes]
                        elif cell_summary_fn == "mean":
                            log_fc_summary = log_fc.mean(dim=0)  # [n_genes]
                        elif cell_summary_fn == "trimmed_mean":
                            # Remove top and bottom 10% of cells, then take mean
                            sorted_effects, _ = torch.sort(log_fc, dim=0)
                            n_cells = log_fc.shape[0]
                            trim_size = max(1, int(0.1 * n_cells))
                            trimmed_effects = sorted_effects[trim_size:n_cells-trim_size, :]
                            log_fc_summary = trimmed_effects.mean(dim=0)  # [n_genes]
                        else:
                            raise ValueError(f"Unknown cell_summary_fn: {cell_summary_fn}")
                    else:
                        log_fc_summary = log_fc.mean(dim=0)  # [n_genes]
                    
                    perturbation_log_fcs.append(log_fc_summary.cpu().numpy())
                
                # Stack samples for this perturbation
                batch_effects.append(np.stack(perturbation_log_fcs, axis=0))  # [n_samples, n_genes]
            
            all_effects.append(np.stack(batch_effects, axis=1))  # [n_samples, n_perturbs, n_genes]
        
        # Average across batches (since we're sampling independently for each batch)
        effects = np.stack(all_effects, axis=0).mean(axis=0)  # [n_samples, n_perturbs, n_genes]
        
        return effects

def bayesian_perturbation_analysis_example():
    """
    Comprehensive example of using ResolVI's Bayesian differential expression testing.
    
    This example demonstrates how to use the new get_perturbation_de() method that
    implements the same statistical rigor as scVI's differential expression analysis.
    
    Examples
    --------
    # Basic setup
    >>> import scanpy as sc
    >>> from scvi.external import RESOLVI
    
    # Setup ResolVI with perturbation data
    >>> RESOLVI.setup_anndata(
    ...     adata,
    ...     perturbation_key='condition',  # Column with perturbation categories
    ...     control_perturbation='Control'  # Control condition name
    ... )
    >>> model = RESOLVI(adata)
    >>> model.train()
    
    # 1. Bayesian DE testing (recommended approach)
    # Change mode: tests H0: |effect| ≤ δ vs H1: |effect| > δ
    >>> de_results = model.get_perturbation_de(
    ...     mode="change",           # Bayesian thresholded-null testing
    ...     delta=0.25,             # Effect size threshold (log2 FC)
    ...     fdr_target=0.05,        # Target false discovery rate
    ...     n_samples=200,          # Posterior samples for testing
    ...     use_mixture_posterior=True,  # Most rigorous (like scVI)
    ... )
    
    # View significant genes
    >>> significant_genes = de_results[de_results['is_de_fdr']]
    >>> print(f"Found {len(significant_genes)} DE genes")
    
    # Vanilla mode: tests H0: βg = 0 vs H1: βg ≠ 0  
    >>> de_results_vanilla = model.get_perturbation_de(
    ...     mode="vanilla",         # Point-null hypothesis testing
    ...     fdr_target=0.05,
    ...     n_samples=200,
    ... )
    
    # View genes with strong evidence (high Bayes factors)
    >>> strong_evidence = de_results_vanilla[de_results_vanilla['is_de_bf']]
    
    # 2. Enhanced p-value testing (with automatic correction)
    >>> pval_results = model.get_perturbation_pvalues(
    ...     correct_multiple_testing=True,  # Now enabled by default!
    ...     correction_method="fdr_bh",     # Benjamini-Hochberg FDR
    ...     alpha=0.05,
    ... )
    
    # Access corrected results
    >>> significant_pvals = pval_results['is_significant']
    >>> corrected_pvals = pval_results['pvalues_corrected']
    
    # 3. Compare different approaches
    >>> # Bayesian change mode (most principled)
    >>> bayesian_de = model.get_perturbation_de(mode="change")
    >>> bayesian_genes = set(bayesian_de[bayesian_de['is_de_fdr']]['gene'])
    
    >>> # Enhanced p-values with correction
    >>> pval_corrected = model.get_perturbation_pvalues(correct_multiple_testing=True)
    >>> pval_genes = set(pval_corrected['is_significant'].stack().index[
    ...     pval_corrected['is_significant'].stack()
    ... ].get_level_values(1))
    
    >>> # Compare overlap
    >>> overlap = bayesian_genes & pval_genes
    >>> print(f"Overlap: {len(overlap)} genes")
    
    # 4. Analyze uncertainty and heterogeneity  
    >>> uncertainty_results = model.get_perturbation_effects(
    ...     return_uncertainty=True,
    ...     uncertainty_metrics=["std", "ci_95", "prob_positive"]
    ... )
    
    >>> heterogeneity = model.analyze_perturbation_heterogeneity()
    
    # 5. Manual correction of existing p-values (utility function)
    >>> raw_pvals = model.get_perturbation_pvalues(correct_multiple_testing=False)
    >>> corrected = model.correct_pvalues(raw_pvals, method="fdr_bh")
    
    Key Advantages of Bayesian Approach
    -----------------------------------
    1. **Proper uncertainty quantification**: Uses full posterior distribution
    2. **Effect size thresholds**: Can test meaningful biological effects (δ > 0.25 log2FC)
    3. **Bayesian FDR control**: More principled than classical FDR
    4. **Mixture posterior sampling**: Accounts for latent variable uncertainty (like scVI)
    5. **Rich output**: Provides probabilities, not just binary decisions
    
    Comparison with scVI DE
    -----------------------
    ResolVI's get_perturbation_de() implements the same statistical framework as
    scVI's differential_expression(), but adapted for perturbation analysis:
    
    - **scVI**: Compares populations A vs B directly
    - **ResolVI**: Uses counterfactual inference (control vs perturbed same cells)
    - **Both**: Bayesian testing with mixture posteriors and proper FDR control
    
    Method Selection Guidelines
    ---------------------------
    - **get_perturbation_de()**: Most rigorous, recommended for publications
    - **get_perturbation_pvalues()**: Faster, good for exploration (now with correction!)
    - **get_perturbation_effects()**: For effect size estimation and uncertainty
    
    See Also
    --------
    scvi.model.SCVI.differential_expression : scVI's DE testing framework
    """
    pass

def perturbation_focused_training_example():
    """
    Example of using ResolVI's perturbation-focused training mode.
    
    This demonstrates the new train_on_perturbed_only functionality which trains
    the model only on perturbed cells while keeping background and neighbor
    computations from all cells.
    
    Examples
    --------
    # Setup ResolVI with perturbation data
    >>> import scanpy as sc
    >>> from scvi.external import RESOLVI
    
    >>> # Load your spatial + perturbation data
    >>> adata = sc.read_h5ad("your_perturbation_data.h5ad")
    
    >>> # Setup with perturbation information
    >>> RESOLVI.setup_anndata(
    ...     adata,
    ...     spatial_rep="X_spatial",        # Spatial coordinates
    ...     perturbation_key="condition",    # Column with perturbation info
    ...     control_perturbation="Control",  # Name of control condition
    ...     batch_key="batch",              # Optional batch info
    ... )
    
    >>> # Create model
    >>> model = RESOLVI(adata)
    
    >>> # Check perturbation distribution before training
    >>> summary = model.get_perturbation_summary()
    >>> print(summary)
    #     perturbation_category  perturbation_code  n_cells  percentage  is_control  used_for_training
    # 0               Control                  0     5000        50.0        True              False
    # 1             Treatment1                  1     2500        25.0       False               True
    # 2             Treatment2                  2     1500        15.0       False               True
    # 3             Treatment3                  3     1000        10.0       False               True
    
    >>> # Training Option 1: Standard training (all cells)
    >>> model_standard = RESOLVI(adata)
    >>> model_standard.train(max_epochs=50)
    >>> print("Standard training: learned from all 10,000 cells")
    
    >>> # Training Option 2: Perturbation-focused training (perturbed cells only)
    >>> model_focused = RESOLVI(adata)
    >>> model_focused.train(
    ...     max_epochs=50,
    ...     train_on_perturbed_only=True,  # KEY: Only train on perturbed cells
    ...     lr=1e-3,                       # May need different learning rate
    ... )
    >>> print("Focused training: learned from 5,000 perturbed cells")
    >>> print("Background/neighbors still computed from all 10,000 cells")
    
    >>> # Compare perturbation effects between training modes
    >>> effects_standard = model_standard.get_perturbation_effects()
    >>> effects_focused = model_focused.get_perturbation_effects()
    
    >>> # The focused model often shows:
    >>> # - More sensitive detection of perturbation effects
    >>> # - Better separation of perturbation-specific patterns
    >>> # - Reduced influence of control cell variability
    
    >>> # Analyze with Bayesian testing
    >>> de_standard = model_standard.get_perturbation_de(mode="change")
    >>> de_focused = model_focused.get_perturbation_de(mode="change")
    
    >>> # Compare number of significant genes
    >>> n_sig_standard = de_standard['is_de_fdr'].sum()
    >>> n_sig_focused = de_focused['is_de_fdr'].sum()
    >>> print(f"Standard training: {n_sig_standard} significant genes")
    >>> print(f"Focused training: {n_sig_focused} significant genes")
    
    When to Use Perturbation-Focused Training
    ------------------------------------------
    **Use train_on_perturbed_only=True when:**
    - You want maximum sensitivity for detecting perturbation effects
    - Control cells are very heterogeneous and might dilute perturbation signals
    - You have many control cells but fewer perturbed cells
    - Your perturbations create strong, specific effects you want to capture
    
    **Use standard training when:**
    - You want a general model of the cellular state space
    - Control and perturbed cells are well-balanced
    - You need the model to understand the full range of cellular variation
    - You're doing exploratory analysis before focusing on perturbations
    
    Technical Details
    -----------------
    When train_on_perturbed_only=True:
    
    1. **Training data**: Only perturbed cells (condition != control)
    2. **Spatial neighbors**: Computed from ALL cells (maintains spatial context)
    3. **Background estimation**: Computed from ALL cells (proper background model)
    4. **Model parameters**: Same architecture, different training data
    5. **Inference**: Works on any cell (control or perturbed)
    
    This approach maintains the spatial and background modeling benefits while
    focusing the learning objective on perturbation-specific patterns.
    
    See Also
    --------
    RESOLVI.get_perturbation_summary : Check perturbation distribution
    RESOLVI.get_perturbation_effects : Estimate perturbation effects
    RESOLVI.get_perturbation_de : Bayesian differential expression testing
    """
    pass