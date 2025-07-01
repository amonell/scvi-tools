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

            # Debug: print tensor devices
            print(f"Debug - device target: {device}")
            for k, v in kwargs.items():
                if v is not None and hasattr(v, 'device'):
                    print(f"Debug - {k} device: {v.device}")

            if kwargs["cat_covs"] is not None and self.module.encode_covariates:
                categorical_input = list(torch.split(kwargs["cat_covs"], 1, dim=1))
            else:
                categorical_input = ()

            # Force explicit device transfer right before encoder call
            x_input = torch.log1p(kwargs["x"] / torch.mean(kwargs["x"], dim=1, keepdim=True))
            batch_input = kwargs["batch_index"]
            
            # Ensure all inputs are on the correct device
            x_input = x_input.to(device)
            batch_input = batch_input.to(device)
            categorical_input = [c.to(device) for c in categorical_input]
            
            print(f"Debug - x_input device: {x_input.device}")
            print(f"Debug - batch_input device: {batch_input.device}")
            print(f"Debug - categorical_input devices: {[c.device for c in categorical_input]}")

            # Check and ensure encoder is on correct device
            print(f"Debug - z_encoder device: {next(self.module.z_encoder.parameters()).device}")
            if next(self.module.z_encoder.parameters()).device != device:
                print(f"Debug - Moving z_encoder to {device}")
                self.module.z_encoder = self.module.z_encoder.to(device)
                print(f"Debug - z_encoder device after move: {next(self.module.z_encoder.parameters()).device}")

            # Also ensure the entire module is on the correct device
            if next(self.module.parameters()).device != device:
                print(f"Debug - Moving entire module to {device}")
                self.module = self.module.to(device)
                print(f"Debug - Module device after move: {next(self.module.parameters()).device}")

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

                qz_m, qz_v, _ = self.module.z_encoder(
                    torch.log1p(kwargs["x"] / torch.mean(kwargs["x"], dim=1, keepdim=True)),
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
    ) -> np.ndarray | pd.DataFrame:
        """
        Get perturbation effects using the full generative model counterfactual pipeline.
        
        This method:
        1. Uses fixed latent representations from get_latent_representation
        2. Generates counterfactual expressions for each perturbation via the full model
        3. Compares the resulting count distributions (not just rates)
        """
        import pyro
        from pyro import distributions as dist
        from pyro.infer import Predictive
        from scvi import REGISTRY_KEYS
        
        adata = self._validate_anndata(adata)
        
        if indices is None:
            indices = np.arange(adata.n_obs)
            
        # Get the perturbation index from categorical covariates
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
        
        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        # Ensure the entire module is on the correct device before any operations
        _, _, device = parse_device_args(
            "auto", "auto", return_device="torch", validate_single_device=True
        )
        if next(self.module.parameters()).device != device:
            print(f"Debug - Moving entire module to {device} at start of get_perturbation_effects")
            self.module = self.module.to(device)
            print(f"Debug - Module device after move: {next(self.module.parameters()).device}")

        # Step 1: Fix latent representations using trained posterior
        z_fixed = self.get_latent_representation(
            adata=adata, indices=indices, give_mean=True, batch_size=batch_size
        )
        
        # Convert to tensor 
        _, _, device = parse_device_args(
            "auto", "auto", return_device="torch", validate_single_device=True
        )
        z_fixed = torch.from_numpy(z_fixed).to(device)
        
        # Debug: Check perturbation setup
        print(f"Debug: perturbation_idx = {perturbation_idx}, n_perturbs = {n_perturbs}")
        print(f"Debug: perturbation_list = {perturbation_list}")
        
        # CRITICAL: Check what the actual perturbation categories are in the data
        # We need to use the ACTUAL perturbation values, not just indices 0,1,2
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            cat_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            if hasattr(cat_state_registry, 'field_keys'):
                field_keys = cat_state_registry.field_keys
                if perturbation_idx < len(field_keys):
                    perturbation_key = field_keys[perturbation_idx]
                    
                    # Get the actual perturbation values from the original data
                    original_perturb_values = adata.obs[perturbation_key].cat.categories.tolist()
                    print(f"Debug: Actual perturbation categories: {original_perturb_values}")
                    
                    # Get the current perturbation values in our selected cells
                    current_perturb_values = adata[indices].obs[perturbation_key].unique()
                    print(f"Debug: Perturbation values in selected cells: {current_perturb_values}")
                    
                    # Map category names to indices
                    perturb_name_to_idx = {name: idx for idx, name in enumerate(original_perturb_values)}
                    print(f"Debug: Perturbation name to index mapping: {perturb_name_to_idx}")
                    
                    # The control should be index 0 (first category)
                    control_name = original_perturb_values[0]
                    print(f"Debug: Control condition: '{control_name}' (index 0)")
                    
                    # Update perturbation_list to use actual meaningful perturbations
                    if perturbation_list == list(range(1, n_perturbs)):
                        # Default case - use all non-control perturbations
                        perturbation_list = list(range(1, len(original_perturb_values)))
                        print(f"Debug: Updated perturbation_list to: {perturbation_list}")
                        treatment_names = [original_perturb_values[i] for i in perturbation_list]
                        print(f"Debug: Treatment conditions: {treatment_names}")
        
        # Create data loader for getting other necessary tensors
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        
        all_counterfactual_counts = []
        baseline_perturbation = None  # Will be set from first batch
        
        for tensors in scdl:
            _, kwargs = self.module._get_fn_args_from_batch(tensors)
            kwargs = {k: v.to(device) if v is not None else v for k, v in kwargs.items()}
            
            batch_size_actual = kwargs["x"].shape[0]
            z_batch = z_fixed[:batch_size_actual, :]  # Use fixed z for this batch
            
            # Debug: Check original perturbation values in this batch
            if kwargs["cat_covs"] is not None:
                original_perturbs = kwargs["cat_covs"][:, perturbation_idx]
                print(f"Debug: Original perturbation values in batch: {original_perturbs.unique()}")
                
                # IMPORTANT: If all cells are from the same perturbation condition,
                # we should use THAT as our baseline, not force it to 0
                actual_baseline_perturb = original_perturbs[0].item()  # All cells should have same perturbation
                print(f"Debug: Actual baseline perturbation for these cells: {actual_baseline_perturb}")
            else:
                print("Debug: No cat_covs found in batch")
                actual_baseline_perturb = 0
            
            # Generate counterfactual counts for each perturbation separately (easier to debug)
            perturbation_counts = {}
            
            # Use the ACTUAL baseline perturbation of these cells, plus other perturbations to compare
            baseline_perturb = actual_baseline_perturb
            if baseline_perturbation is None:
                baseline_perturbation = baseline_perturb  # Store for later use
            
            all_perturb_vals = [baseline_perturb] + [p for p in perturbation_list if p != baseline_perturb]
            print(f"Debug: Will generate counterfactuals for perturbations: {all_perturb_vals}")
            print(f"Debug: Baseline (control): {baseline_perturb}, Treatments: {[p for p in perturbation_list if p != baseline_perturb]}")
            
            for perturb_val in all_perturb_vals:
                print(f"Debug: Generating counterfactuals for perturbation {perturb_val}")
                
                # Create modified cat_covs with this specific perturbation
                if kwargs["cat_covs"] is not None:
                    cat_covs_modified = kwargs["cat_covs"].clone()
                    cat_covs_modified[:, perturbation_idx] = perturb_val
                    print(f"Debug: Modified cat_covs perturbation column to {perturb_val}")
                    print(f"Debug: cat_covs_modified[:, {perturbation_idx}] = {cat_covs_modified[:, perturbation_idx].unique()}")
                else:
                    # Create cat_covs if it doesn't exist
                    cat_covs_modified = torch.zeros(
                        (batch_size_actual, perturbation_idx + 1), 
                        device=device, dtype=torch.long
                    )
                    cat_covs_modified[:, perturbation_idx] = perturb_val
                    print(f"Debug: Created new cat_covs with perturbation {perturb_val}")
                
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
                
                print(f"Debug: Generated counts shape for perturbation {perturb_val}: {counts.shape}")
                print(f"Debug: Sample mean counts: {counts.mean().item():.3f}")
                
                perturbation_counts[perturb_val] = counts
                
            all_counterfactual_counts.append(perturbation_counts)
        
        # Concatenate results across batches - use the actual baseline perturbation
        baseline_key = baseline_perturbation if baseline_perturbation is not None else 0
        print(f"Debug: Using baseline perturbation {baseline_key} as control")
        control_counts = torch.cat([batch[baseline_key] for batch in all_counterfactual_counts], dim=1)  # [n_samples, total_cells, n_genes]
        
        print(f"Debug: Control counts shape: {control_counts.shape}")
        print(f"Debug: Control counts mean: {control_counts.mean().item():.3f}")
        print(f"Debug: Control counts range: {control_counts.min().item():.3f} - {control_counts.max().item():.3f}")
        
        perturbation_effects = []
        for perturb_idx in perturbation_list:
            perturb_counts = torch.cat([batch[perturb_idx] for batch in all_counterfactual_counts], dim=1)  # [n_samples, total_cells, n_genes]
            
            print(f"Debug: Perturbation {perturb_idx} counts shape: {perturb_counts.shape}")
            print(f"Debug: Perturbation {perturb_idx} counts mean: {perturb_counts.mean().item():.3f}")
            print(f"Debug: Perturbation {perturb_idx} counts range: {perturb_counts.min().item():.3f} - {perturb_counts.max().item():.3f}")
            
            # Compute log fold change on actual counts
            eps = 1e-8
            control_safe = torch.clamp(control_counts, min=eps)
            perturb_safe = torch.clamp(perturb_counts, min=eps)
            
            log_fc = torch.log2(perturb_safe / control_safe)  # [n_samples, total_cells, n_genes]
            
            print(f"Debug: Log fold change for perturbation {perturb_idx}:")
            print(f"  - Mean: {log_fc.mean().item():.3f}")
            print(f"  - Range: {log_fc.min().item():.3f} - {log_fc.max().item():.3f}")
            print(f"  - Std: {log_fc.std().item():.3f}")
            
            log_fc = log_fc[..., gene_mask]
            
            perturbation_effects.append(log_fc)
        
        # Stack and process results
        perturbation_effects = torch.stack(perturbation_effects, dim=1)  # [n_samples, n_perturbs, n_cells, n_genes]
        
        # Average across cells
        perturbation_effects = perturbation_effects.mean(dim=2).cpu().numpy()  # [n_samples, n_perturbs, n_genes]
        
        if return_mean:
            perturbation_effects = perturbation_effects.mean(axis=0)  # [n_perturbs, n_genes]

        if return_numpy is None or return_numpy is False:
            gene_names = adata.var_names[gene_mask] if gene_list is not None else adata.var_names
            perturbation_names = [f"perturbation_{i}" for i in perturbation_list]
            
            if return_mean:
                return pd.DataFrame(
                    perturbation_effects, index=perturbation_names, columns=gene_names
                )
            else:
                return perturbation_effects
        else:
            return perturbation_effects