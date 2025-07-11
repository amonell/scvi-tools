from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyro
from pyro.infer import Trace_ELBO

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    ObsmField,
)
from scvi.dataloaders import AnnTorchDataset
from scvi.model._utils import (
    scrna_raw_counts_properties,
)
from scvi.model.base import ArchesMixin, BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.model.base._de_core import _de_core
from scvi.utils import de_dsp, setup_anndata_dsp

from ._module import RESOLVAE
from ._utils import ResolVIPredictiveMixin

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class RESOLVI(
    PyroSviTrainMixin, PyroSampleMixin, ResolVIPredictiveMixin, BaseModelClass, ArchesMixin
):
    """
    ResolVI addresses noise and bias in single-cell resolved spatial transcriptomics data.

    This model also supports perturbation analysis through counterfactual inference,
    allowing estimation of gene expression changes due to genetic or chemical perturbations.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    override_mixture_k_in_semisupervised
        If True (default), automatically sets mixture_k to the number of cell type labels
        when semisupervised=True. If False, respects the user-provided mixture_k value
        even in semisupervised mode. This is useful when you have homogeneous cell types
        in your perturbation data and want to set mixture_k=1.
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    Basic spatial analysis:
    
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.external.RESOLVI.setup_anndata(adata, batch_key="batch")
    >>> model = scvi.external.RESOLVI(adata)
    >>> model.train()
    >>> adata.obsm["X_ResolVI"] = model.get_latent_representation()
    >>> adata.obsm["X_normalized_ResolVI"] = model.get_normalized_expression()
    
    Perturbation analysis:
    
    >>> # Setup with perturbation data
    >>> scvi.external.RESOLVI.setup_anndata(
    ...     adata, 
    ...     perturbation_key="condition",
    ...     control_perturbation="Control"
    ... )
    >>> model = scvi.external.RESOLVI(adata)
    
    >>> # Setup with perturbation + background data (separate keys)
    >>> scvi.external.RESOLVI.setup_anndata(
    ...     adata, 
    ...     perturbation_key="condition", 
    ...     control_perturbation="Control",
    ...     background_key="cell_type",
    ...     background_category="Background"
    ... )
    
    >>> # Setup with perturbation + background data (same key, different categories)
    >>> scvi.external.RESOLVI.setup_anndata(
    ...     adata, 
    ...     perturbation_key="guide_rnas", 
    ...     control_perturbation="sgCd19",        # Control guide RNA
    ...     background_key="guide_rnas",          # Same key as perturbation
    ...     background_category="Other cells"     # Background category
    ... )
    
    >>> # Check perturbation distribution
    >>> summary = model.get_perturbation_summary()
    
    >>> # Option 1: Train on all cells (standard)
    >>> model.train()
    
    >>> # Option 2: Train only on perturbation-relevant cells (control + perturbed, excluding background)
    >>> model.train(train_on_perturbed_only=True)
    
    >>> # Analyze perturbation effects
    >>> effects = model.get_perturbation_effects()
    >>> de_results = model.get_perturbation_de(mode="change")
    
    Using custom mixture_k in semisupervised mode:
    
    >>> # When your perturbed cells are homogeneous (same cell type)
    >>> # and you want to override the automatic mixture_k setting
    >>> model = scvi.external.RESOLVI(
    ...     adata, 
    ...     semisupervised=True,
    ...     mixture_k=1,  # Custom value instead of n_labels
    ...     override_mixture_k_in_semisupervised=False  # Respect custom mixture_k
    ... )

    Notes
    -----
    For perturbation studies, ResolVI can be trained in two modes:
    
    1. **Standard training**: Uses all cells (control + perturbed + background) for training. 
       This learns a general representation of the cellular state space.
       
    2. **Perturbation-focused training**: Uses only perturbation-relevant cells 
       (both control and perturbed) for training via `train(train_on_perturbed_only=True)`. 
       Background cells are excluded from training but still used for spatial context.
       This helps the shift network learn both baseline (control) and perturbation effects
       while improving sensitivity for detecting perturbation-specific patterns.

    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/api_overview`
    2. :doc:`/tutorials/notebooks/harmonization`
    3. :doc:`/tutorials/notebooks/scarches_scvi_tools`
    4. :doc:`/tutorials/notebooks/scvi_in_R`
    """

    _module_cls = RESOLVAE
    _block_parameters = []

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 32,
        n_hidden_encoder: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.05,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        background_ratio=None,
        median_distance=None,
        semisupervised=False,
        mixture_k=50,
        downsample_counts=True,
        perturbation_embed_dim: int = 16,
        perturbation_hidden_dim: int = 64,
        override_mixture_k_in_semisupervised: bool = True,
        **model_kwargs,
    ):
        pyro.clear_param_store()

        super().__init__(adata)
        if semisupervised:
            self._set_indices_and_labels()

        results = self.compute_dataset_dependent_priors()

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_labels = self.summary_stats.n_labels - 1
        
        # Get number of perturbation categories by inspecting registered fields
        n_perturbs = 1  # default (no perturbations)
        perturbation_idx = None
        
        # Check if we have categorical covariates and extract perturbation info
        try:
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
                cat_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
                
                # Get the field keys (which are the categorical covariate names)
                if hasattr(cat_state_registry, 'field_keys'):
                    field_keys = cat_state_registry.field_keys
                    
                    # Look for perturbation key in setup args - try multiple approaches
                    perturbation_key = None
                    
                    # Method 1: Try registry.setup_method_args
                    if hasattr(self.adata_manager, 'registry') and hasattr(self.adata_manager.registry, 'setup_method_args'):
                        setup_method_args = self.adata_manager.registry.setup_method_args
                        perturbation_key = setup_method_args.get('perturbation_key')
                    
                    # Method 2: Try direct access to setup_method_args
                    if not perturbation_key and hasattr(self.adata_manager, 'setup_method_args'):
                        setup_method_args = self.adata_manager.setup_method_args
                        perturbation_key = setup_method_args.get('perturbation_key')
                    
                    # Method 3: Check the _registry itself
                    if not perturbation_key and hasattr(self.adata_manager, '_registry'):
                        registry = self.adata_manager._registry
                        if hasattr(registry, 'setup_method_args'):
                            setup_method_args = registry.setup_method_args
                            perturbation_key = setup_method_args.get('perturbation_key')
                    
                    # Method 4: Check AnnDataManager's own setup_method_args
                    if not perturbation_key:
                        for attr_name in ['setup_method_args', '_setup_method_args']:
                            if hasattr(self.adata_manager, attr_name):
                                setup_method_args = getattr(self.adata_manager, attr_name)
                                if isinstance(setup_method_args, dict):
                                    perturbation_key = setup_method_args.get('perturbation_key')
                                    if perturbation_key:
                                        break
                    
                    # Method 5: Check if field_keys contains what looks like perturbation data
                    if not perturbation_key:
                        for key in field_keys:
                            # This might be the perturbation key we're looking for
                            if key in ['mapped_batch', 'perturbation', 'condition', 'treatment', 'guide_rnas', 'guide_rna', 'grna', 'crispr']:
                                perturbation_key = key
                                break
                    
                    if perturbation_key and perturbation_key in field_keys:
                        # Found the perturbation! Get its index and number of categories
                        perturbation_idx = field_keys.index(perturbation_key)
                        if hasattr(cat_state_registry, 'n_cats_per_key'):
                            n_perturbs = cat_state_registry.n_cats_per_key[perturbation_idx]
                        
        except Exception as e:
            # Just use defaults
            pass

        if background_ratio is None:
            background_ratio = results["background_ratio"]
        if median_distance is None:
            median_distance = results["median_distance"]
        if downsample_counts:
            downsample_counts_mean = results["mean_log_counts"]
            downsample_counts_std = results["std_log_counts"]
        else:
            downsample_counts_mean = None
            downsample_counts_std = 1.0

        expression_anntorchdata = AnnTorchDataset(
            self.adata_manager,
            getitem_tensors=["X"],
            load_sparse_tensor=True,
        )
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_cats_per_cov=n_cats_per_cov,
            n_labels=n_labels,
            mixture_k=mixture_k,
            expression_anntorchdata=expression_anntorchdata,
            n_neighbors=self.summary_stats.n_distance_neighbor,
            n_obs=self.summary_stats["n_ind_x"],
            n_hidden=n_hidden,
            n_hidden_encoder=n_hidden_encoder,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            background_ratio=background_ratio,
            median_distance=median_distance,
            semisupervised=semisupervised,
            downsample_counts_mean=downsample_counts_mean,
            downsample_counts_std=downsample_counts_std,
            n_perturbs=n_perturbs,
            perturbation_embed_dim=perturbation_embed_dim,
            perturbation_hidden_dim=perturbation_hidden_dim,
            perturbation_idx=perturbation_idx,
            override_mixture_k_in_semisupervised=override_mixture_k_in_semisupervised,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"RESOLVI Model with the following params: \nn_hidden: {n_hidden} "
            f"n_latent: {n_latent}, n_layers: {n_layers}, dropout_rate: "
            f"{dropout_rate}, dispersion: {dispersion}, gene_likelihood: {gene_likelihood} "
            f"n_neighbors: {self.summary_stats.n_distance_neighbor}"
        )
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 50,
        lr: float = 3e-3,
        lr_extra: float = 1e-2,
        extra_lr_parameters: tuple = ("per_neighbor_diffusion_map", "u_prior_means"),
        batch_size: int = 512,
        weight_decay: float = 0.0,
        eps: float = 1e-4,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 20,
        plan_kwargs: dict | None = None,
        expose_params: list = (),
        train_on_perturbed_only: bool = False,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        lr_extra
            Learning rate for parameters (non-amortized and custom ones)
        extra_lr_parameters
            List of parameters to train with `lr_extra` learning rate.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        plan_kwargs
            Keyword args for :class:`~resolvi.train.PyroTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        expose_params
            List of parameters to train if running model in Arches mode.
        train_on_perturbed_only
            If True, trains the model only on perturbation-relevant cells (both control 
            and perturbed cells). This excludes background cells that are not part of 
            the perturbation experiment. Background and neighbor computations still use 
            all cells, but the training objective focuses on control and perturbed cells.
            This helps the shift network learn both the baseline (control) and the 
            perturbation effects. Background cells are used for spatial context but 
            not for training the perturbation model.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        # Handle perturbation-focused training
        if train_on_perturbed_only:
            perturbation_idx = self.module.model.perturbation_idx
            if perturbation_idx is None:
                raise ValueError(
                    "train_on_perturbed_only=True requires perturbation_key to be set during setup_anndata. "
                    "Please call RESOLVI.setup_anndata() with perturbation_key parameter."
                )
            
            # Get perturbation and background categories from the data
            from scvi import REGISTRY_KEYS
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
                cat_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
                if hasattr(cat_state_registry, 'field_keys'):
                    field_keys = cat_state_registry.field_keys
                    if perturbation_idx < len(field_keys):
                        perturbation_key = field_keys[perturbation_idx]
                        
                        # Get perturbation values for all cells
                        cell_perturbations = self.adata.obs[perturbation_key].cat.codes.values
                        
                        # Find background cells if background_key was specified
                        background_indices = np.array([], dtype=int)
                        background_key = None
                        
                        # Look for background key in the setup args
                        setup_args_dict = self.adata_manager._get_setup_method_args()
                        from scvi.data._constants import _SETUP_ARGS_KEY
                        setup_args = setup_args_dict.get(_SETUP_ARGS_KEY, {})
                        if 'background_key' in setup_args and setup_args['background_key'] is not None:
                            background_key = setup_args['background_key']
                            
                            # Find the background_key index in categorical covariates
                            background_idx = None
                            for idx, key in enumerate(field_keys):
                                if key == background_key:
                                    background_idx = idx
                                    break
                            
                            if background_idx is not None:
                                # Get background values for all cells
                                cell_backgrounds = self.adata.obs[background_key].cat.codes.values
                                background_indices = np.where(cell_backgrounds == 0)[0]  # First category is background
                        
                        # Identify perturbation-relevant cells (control + perturbed, excluding background)
                        all_indices = np.arange(self.adata.n_obs)
                        perturbation_relevant_indices = np.setdiff1d(all_indices, background_indices)
                        
                        # Split perturbation-relevant cells into control and perturbed
                        control_cell_indices = np.intersect1d(
                            perturbation_relevant_indices,
                            np.where(cell_perturbations == 0)[0]
                        )
                        perturbed_cell_indices = np.intersect1d(
                            perturbation_relevant_indices,
                            np.where(cell_perturbations != 0)[0]
                        )
                        
                        if len(perturbation_relevant_indices) == 0:
                            raise ValueError(
                                "No perturbation-relevant cells found. All cells are background cells. "
                                "Cannot train on perturbation cells only."
                            )
                        
                        if len(perturbed_cell_indices) == 0:
                            raise ValueError(
                                f"No perturbed cells found among non-background cells. "
                                f"All non-background cells have control perturbation "
                                f"(category 0 in {perturbation_key}). Cannot train on perturbation experiment."
                            )
                        
                        print(f"Training configuration with train_on_perturbed_only=True:")
                        print(f"  Training set: {len(perturbation_relevant_indices)} perturbation-relevant cells "
                              f"({len(perturbation_relevant_indices)/self.adata.n_obs*100:.1f}%)")
                        print(f"    - Control cells: {len(control_cell_indices)} "
                              f"({len(control_cell_indices)/self.adata.n_obs*100:.1f}%)")
                        print(f"    - Perturbed cells: {len(perturbed_cell_indices)} "
                              f"({len(perturbed_cell_indices)/self.adata.n_obs*100:.1f}%)")
                        if len(background_indices) > 0:
                            print(f"  Excluded from training: {len(background_indices)} background cells "
                                  f"({len(background_indices)/self.adata.n_obs*100:.1f}%)")
                        print(f"  Background/neighbor computations: all {self.adata.n_obs} cells")
                        
                        # Use external_indexing to provide custom train/val/test splits
                        if 'datasplitter_kwargs' not in kwargs:
                            kwargs['datasplitter_kwargs'] = {}
                        
                        # Train on all perturbation-relevant cells (control + perturbed)
                        # Use background cells for validation (if any), otherwise split perturbation cells
                        if len(background_indices) > 0:
                            # Use background cells for validation
                            kwargs['datasplitter_kwargs']['external_indexing'] = [
                                perturbation_relevant_indices,  # train on control + perturbed cells
                                background_indices,            # validation on background cells
                                np.array([], dtype=int),       # empty test set
                            ]
                        else:
                            # No background cells, so split perturbation-relevant cells
                            # Use 80% for training, 20% for validation
                            np.random.seed(42)  # For reproducibility
                            shuffled_indices = np.random.permutation(perturbation_relevant_indices)
                            split_point = int(0.8 * len(shuffled_indices))
                            
                            train_indices = shuffled_indices[:split_point]
                            val_indices = shuffled_indices[split_point:]
                            
                            kwargs['datasplitter_kwargs']['external_indexing'] = [
                                train_indices,               # 80% of perturbation-relevant cells
                                val_indices,                 # 20% of perturbation-relevant cells  
                                np.array([], dtype=int),     # empty test set
                            ]
                        
                    else:
                        raise ValueError(f"Perturbation index {perturbation_idx} out of range for categorical covariates")
                else:
                    raise ValueError("Could not find field_keys in categorical covariates state registry")
            else:
                raise ValueError("No categorical covariates found in data registry")

        blocked = self._block_parameters.copy()
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                blocked.append(name)
                param.requires_grad = True
        blocked = set(blocked) - set(expose_params)

        if blocked:
            print("Running scArches. Set lr to 0 and blocking variables.")

        def per_param_callable(module_name, param_name):
            store_name = f"{module_name}$$${param_name}" if "." in param_name else param_name
            if store_name in blocked:
                return {"lr": 0.0, "weight_decay": 0, "eps": eps}
            if store_name in extra_lr_parameters:
                return {"lr": lr_extra, "weight_decay": weight_decay, "eps": eps}
            else:
                return {"lr": lr, "weight_decay": weight_decay, "eps": eps}

        optim = pyro.optim.Adam(per_param_callable)

        if plan_kwargs is None:
            plan_kwargs = {}
        plan_kwargs.update(
            {
                "optim_kwargs": {"lr": lr, "weight_decay": weight_decay, "eps": eps},
                "optim": optim,
                "blocked": blocked,
                "n_epochs_kl_warmup": n_epochs_kl_warmup
                if n_epochs_kl_warmup is not None
                else max_epochs,
                "n_steps_kl_warmup": n_steps_kl_warmup,
                "loss_fn": Trace_ELBO(
                    num_particles=5, vectorize_particles=True, retain_graph=True
                ),
            }
        )

        super().train(
            max_epochs=max_epochs,
            train_size=1.0,
            plan_kwargs=plan_kwargs,
            batch_size=batch_size,
            **kwargs,
        )

    def get_perturbation_summary(self) -> pd.DataFrame:
        """
        Get a summary of perturbation conditions in the dataset.
        
        This method helps users understand which cells have which perturbations
        and what will happen when train_on_perturbed_only=True is used.
        
        Returns
        -------
        DataFrame with perturbation summary containing:
        - perturbation_category: Name of the perturbation condition
        - perturbation_code: Numeric code (0=control, >0=perturbed)
        - n_cells: Number of cells with this condition
        - percentage: Percentage of total cells
        - is_control: Whether this is the control condition
        - used_for_training: Whether cells with this condition will be used
          when train_on_perturbed_only=True (both control and perturbed are used)
        - is_background: Whether cells are background (excluded from perturbation training)
        """
        perturbation_idx = self.module.model.perturbation_idx
        if perturbation_idx is None:
            raise ValueError(
                "No perturbation found in categorical covariates. "
                "Please set perturbation_key during setup_anndata."
            )
        
        # Get perturbation information
        from scvi import REGISTRY_KEYS
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            cat_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            if hasattr(cat_state_registry, 'field_keys'):
                field_keys = cat_state_registry.field_keys
                if perturbation_idx < len(field_keys):
                    perturbation_key = field_keys[perturbation_idx]
                    
                    # Get perturbation data
                    perturbation_series = self.adata.obs[perturbation_key]
                    category_names = perturbation_series.cat.categories.tolist()
                    category_codes = perturbation_series.cat.codes.values
                    
                    # Check for background cells
                    background_indices = np.array([], dtype=int)
                    setup_args_dict = self.adata_manager._get_setup_method_args()
                    from scvi.data._constants import _SETUP_ARGS_KEY
                    setup_args = setup_args_dict.get(_SETUP_ARGS_KEY, {})
                    background_key = None
                    
                    if 'background_key' in setup_args and setup_args['background_key'] is not None:
                        background_key = setup_args['background_key']
                        
                        # Find the background_key index in categorical covariates
                        background_idx = None
                        for idx, key in enumerate(field_keys):
                            if key == background_key:
                                background_idx = idx
                                break
                        
                        if background_idx is not None:
                            # Get background values for all cells
                            cell_backgrounds = self.adata.obs[background_key].cat.codes.values
                            background_indices = np.where(cell_backgrounds == 0)[0]  # First category is background

                    # Create summary
                    summary_data = []
                    total_cells = len(category_codes)
                    
                    for code, category_name in enumerate(category_names):
                        n_cells = np.sum(category_codes == code)
                        percentage = (n_cells / total_cells) * 100
                        is_control = (code == 0)  # First category is control
                        
                        # Both control and perturbed cells are used for training in train_on_perturbed_only mode
                        # Background cells are excluded from training
                        used_for_training = True  # Both control and perturbed used for training
                        is_background = False
                        
                        summary_data.append({
                            'perturbation_category': category_name,
                            'perturbation_code': code,
                            'n_cells': n_cells,
                            'percentage': round(percentage, 2),
                            'is_control': is_control,
                            'used_for_training': used_for_training,
                            'is_background': is_background
                        })
                    
                    # Add background information if available
                    if len(background_indices) > 0 and background_key:
                        background_categories = self.adata.obs[background_key].cat.categories.tolist()
                        background_codes = self.adata.obs[background_key].cat.codes.values
                        
                        for code, category_name in enumerate(background_categories):
                            n_cells = np.sum(background_codes == code)
                            if n_cells > 0:
                                percentage = (n_cells / total_cells) * 100
                                is_background = (code == 0)  # First category is background
                                
                                summary_data.append({
                                    'perturbation_category': f"[Background] {category_name}",
                                    'perturbation_code': f"bg_{code}",
                                    'n_cells': n_cells,
                                    'percentage': round(percentage, 2),
                                    'is_control': False,
                                    'used_for_training': not is_background,  # Background cells excluded
                                    'is_background': is_background
                                })
                    
                    df = pd.DataFrame(summary_data)
                    
                    # Add summary statistics
                    n_control = df[(df['is_control']) & (~df['is_background'])]['n_cells'].sum()
                    n_perturbed = df[(~df['is_control']) & (~df['is_background'])]['n_cells'].sum()
                    n_background = df[df['is_background']]['n_cells'].sum()
                    n_perturbation_relevant = n_control + n_perturbed
                    
                    print(f"Perturbation Summary for key '{perturbation_key}':")
                    print(f"  Total cells: {total_cells}")
                    print(f"  Control cells: {n_control} ({n_control/total_cells*100:.1f}%)")
                    print(f"  Perturbed cells: {n_perturbed} ({n_perturbed/total_cells*100:.1f}%)")
                    if n_background > 0:
                        print(f"  Background cells: {n_background} ({n_background/total_cells*100:.1f}%)")
                    print(f"\nWhen train_on_perturbed_only=True:")
                    print(f"  - Training will use {n_perturbation_relevant} perturbation-relevant cells")
                    print(f"    (both {n_control} control + {n_perturbed} perturbed cells)")
                    if n_background > 0:
                        print(f"  - {n_background} background cells will be excluded from training")
                    print(f"  - Background/neighbor computations will still use all {total_cells} cells")
                    
                    return df
                else:
                    raise ValueError(f"Perturbation index {perturbation_idx} out of range")
            else:
                raise ValueError("Could not find field_keys in categorical covariates")
        else:
            raise ValueError("No categorical covariates found in data registry")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        perturbation_key: str | None = None,
        control_perturbation: str | None = None,
        background_key: str | None = None,
        background_category: str | None = None,
        prepare_data: bool | None = True,
        prepare_data_kwargs: dict = None,
        unlabeled_category: str = "unknown",
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_cat_cov_keys)s
        perturbation_key
            Key in adata.obs for perturbation categories (e.g., knockout conditions).
        control_perturbation
            Value in perturbation_key column that represents the control condition.
            If None, the first category alphabetically will be used as control.
        background_key
            Key in adata.obs for background cell categories. Background cells are those
            that should not be used for perturbation training (neither control nor perturbed).
            When train_on_perturbed_only=True, only control and perturbed cells will be used
            for training, while background cells will be excluded from training but still
            used for neighbor and background computations. Can be the same as perturbation_key
            if different categories within the same column represent different cell roles.
        background_category
            Value in background_key column that represents background cells.
            If None and background_key is provided, the first category will be used.
            Should be different from control_perturbation if using the same key.
        prepare_data
            If True, prepares AnnData for training. Computes spatial neighbors and distances.
        prepare_data_kwargs
            Keyword args for :meth:`scvi.external.RESOLVI._prepare_data`
        %(param_unlabeled_category)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        if layer is None:
            x = adata.X
        else:
            x = adata.layers[layer]
        assert np.min(x.sum(axis=1)) > 0, (
            "Please filter cells with less than 5 counts prior to running resolVI."
        )
        if batch_key is not None:
            adata.obs["_indices"] = (
                adata.obs[batch_key].astype(str) + "_" + adata.obs_names.astype(str)
            )
        else:
            adata.obs["_indices"] = adata.obs_names.astype(str)
        adata.obs["_indices"] = adata.obs["_indices"].astype("category")
        assert not adata.obs["_indices"].duplicated(keep="first").any(), (
            "obs_names need to be unique prior to running resolVI."
        )
        if labels_key is None:
            label_field = CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key)
        else:
            label_field = LabelsWithUnlabeledObsField(
                REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category
            )

        if prepare_data:
            if prepare_data_kwargs is None:
                prepare_data_kwargs = {}
            RESOLVI._prepare_data(adata, batch_key=batch_key, **prepare_data_kwargs)

        # Initialize categorical_covariate_keys as a set to avoid duplicates
        if categorical_covariate_keys is None:
            categorical_covariate_keys = []
        else:
            categorical_covariate_keys = list(categorical_covariate_keys)
        
        # Validate usage when same key is used for perturbation and background
        if (perturbation_key is not None and background_key is not None and 
            perturbation_key == background_key):
            if (control_perturbation is not None and background_category is not None and 
                control_perturbation == background_category):
                raise ValueError(
                    f"When using the same key ('{perturbation_key}') for both perturbation_key and "
                    f"background_key, control_perturbation ('{control_perturbation}') and "
                    f"background_category ('{background_category}') must be different values."
                )

        # Handle control perturbation mapping and include in categorical covariates
        if perturbation_key is not None:
            if control_perturbation is not None:
                # Ensure control perturbation is at index 0
                if control_perturbation not in adata.obs[perturbation_key].unique():
                    raise ValueError(f"Control perturbation '{control_perturbation}' not found in {perturbation_key}")
                
                # Reorder categories to put control first
                current_categories = adata.obs[perturbation_key].cat.categories.tolist()
                if control_perturbation in current_categories:
                    current_categories.remove(control_perturbation)
                    new_categories = [control_perturbation] + current_categories
                    adata.obs[perturbation_key] = adata.obs[perturbation_key].cat.reorder_categories(new_categories)
            
            # Include perturbation in categorical covariates (avoid duplicates)
            if perturbation_key not in categorical_covariate_keys:
                categorical_covariate_keys.append(perturbation_key)

        # Handle background key if provided
        if background_key is not None:
            if background_category is not None:
                if background_category not in adata.obs[background_key].unique():
                    raise ValueError(f"Background category '{background_category}' not found in {background_key}")
                
                # Reorder categories to put background first
                current_categories = adata.obs[background_key].cat.categories.tolist()
                if background_category in current_categories:
                    current_categories.remove(background_category)
                    new_categories = [background_category] + current_categories
                    adata.obs[background_key] = adata.obs[background_key].cat.reorder_categories(new_categories)
            
            # Include background in categorical covariates (avoid duplicates)
            if background_key not in categorical_covariate_keys:
                categorical_covariate_keys.append(background_key)
        
        # Convert back to None if empty to maintain compatibility
        if len(categorical_covariate_keys) == 0:
            categorical_covariate_keys = None

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            ObsmField("index_neighbor", "index_neighbor"),
            ObsmField("distance_neighbor", "distance_neighbor"),
            CategoricalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
            label_field,
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @staticmethod
    def _prepare_data(
        adata, n_neighbors=10, spatial_rep="X_spatial", batch_key=None, slice_key=None, **kwargs
    ):
        if slice_key is not None:
            batch_key = slice_key
        try:
            import scanpy
            from sklearn.neighbors._base import _kneighbors_from_graph
        except ImportError as err:
            raise ImportError(
                "Please install scanpy and scikit-learn -- `pip install scanpy`"
            ) from err

        # Spatial neighbors are batch dependent.
        if batch_key is None:
            indices = [np.arange(adata.n_obs)]
        else:
            indices = [
                np.where(adata.obs[batch_key] == i)[0] for i in adata.obs[batch_key].unique()
            ]

        distance_neighbor = 1e6 * np.ones([adata.n_obs, n_neighbors])
        index_neighbor = np.zeros([adata.n_obs, n_neighbors], dtype=int)

        for index in indices:
            sub_data = adata[index].copy()
            try:
                import rapids_singlecell

                print("RAPIDS SingleCell is installed and can be imported")
                rapids_singlecell.pp.neighbors(
                    sub_data, n_neighbors=n_neighbors + 5, use_rep=spatial_rep
                )
            except ImportError:
                scanpy.pp.neighbors(sub_data, n_neighbors=n_neighbors + 5, use_rep=spatial_rep)
            distances = sub_data.obsp["distances"] ** 2

            distance_neighbor[index, :], index_neighbor_batch = _kneighbors_from_graph(
                distances, n_neighbors, return_distance=True
            )
            index_neighbor[index, :] = index[index_neighbor_batch]

        adata.obsm["X_spatial"] = adata.obsm[spatial_rep]
        adata.obsm["index_neighbor"] = index_neighbor
        adata.obsm["distance_neighbor"] = distance_neighbor

    def compute_dataset_dependent_priors(self, n_small_genes=None):
        x = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        n_small_genes = x.shape[1] // 50 if n_small_genes is None else int(n_small_genes)
        # Computing library size over low expressed genes (expectation for background).
        # Handles sparse and dense counts.
        smallest_means = x[:, np.array(x.sum(0)).flatten().argsort()[:n_small_genes]].mean(
            1
        ) / np.array(x.mean(1))
        background_ratio = np.mean(np.array(smallest_means))

        # Median distance for empiric expectation of kernel size in diffusion
        distance = self.adata_manager.get_from_registry("distance_neighbor")
        median_distance = np.median(np.partition(distance, 5)[:, 5])
        log_library_size = np.log1p(np.array(x.sum(1)))
        mean_log_counts = np.median(log_library_size)
        std_log_counts = np.std(log_library_size)

        return {
            "background_ratio": background_ratio,
            "median_distance": median_distance,
            "mean_log_counts": mean_log_counts,
            "std_log_counts": std_log_counts,
        }

    @de_dsp.dedent
    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        subset_idx: Sequence[int] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        weights: Literal["uniform", "importance"] | None = "uniform",
        filter_outlier_cells: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for differential expression analysis.

        Implements `"vanilla"` DE :cite:p:`Lopez18` and `"change"` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_subset_idx)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        weights
        filter_outlier_cells
            Whether to filter outlier cells with
            :meth:`~scvi.model.base.DifferentialComputation.filter_outlier_cells`
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        model_fn = partial(
            self.get_normalized_expression_importance,
            return_numpy=True,
            n_samples=5,
            batch_size=batch_size,
            weights=weights,
            return_mean=False,
        )

        representation_fn = self.get_latent_representation if filter_outlier_cells else None

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=representation_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            subset_idx=subset_idx,
            all_stats=all_stats,
            all_stats_fn=scrna_raw_counts_properties,
            col_names=adata.var_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )

        return result

    @de_dsp.dedent
    def differential_niche_abundance(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        neighbor_key: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        subset_idx: Sequence[int] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        filter_outlier_cells: bool = False,
        pseudocounts: float = 1e-3,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for niche differential abundance analysis.

        Implements `"vanilla"` DE :cite:p:`Lopez18` and `"change"` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        neighbor_key
            Obsm key containing the spatial neighbors of each cell.
        %(de_idx1)s
        %(de_idx2)s
        %(de_subset_idx)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_fdr_target)s
        %(de_silent)s
        filter_outlier_cells
            Whether to filter outlier cells with
            :meth:`~scvi.model.base.DifferentialComputation.filter_outlier_cells`
        pseudocounts
            pseudocount offset used for the mode `change`.
            When None, observations from non-expressed genes are used to estimate its value.
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        model_fn = partial(
            self.get_neighbor_abundance,
            return_numpy=True,
            n_samples=5,
            batch_size=batch_size,
            return_mean=False,
            neighbor_key=neighbor_key,
        )

        representation_fn = self.get_latent_representation if filter_outlier_cells else None

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=representation_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            subset_idx=subset_idx,
            all_stats=False,
            all_stats_fn=scrna_raw_counts_properties,
            col_names=self._label_mapping[:-1],
            mode=mode,
            batchid1=None,
            batchid2=None,
            delta=delta,
            batch_correction=False,
            fdr=fdr_target,
            silent=silent,
            pseudocounts=pseudocounts,
            **kwargs,
        )

        return result

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        soft: bool = False,
        batch_size: int | None = 500,
        num_samples: int | None = 30,
    ) -> np.ndarray | pd.DataFrame:
        """
        Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        indices
            Subsample AnnData to these indices.
        soft
            If True, returns per class probabilities
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        num_samples
            Samples to draw from the posterior for cell-type prediction.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        sampled_prediction = self.sample_posterior(
            adata=adata,
            indices=indices,
            model=self.module.model_corrected,
            return_sites=["probs_prediction"],
            num_samples=num_samples,
            return_samples=False,
            batch_size=batch_size,
            summary_frequency=10,
            return_observed=True,
        )
        y_pred = sampled_prediction["post_sample_means"]["probs_prediction"]

        if not soft:
            y_pred = y_pred.argmax(axis=1)
            predictions = [self._code_to_label[p] for p in y_pred]
            return np.array(predictions)
        else:
            n_labels = len(y_pred[0])
            predictions = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_labels],
                index=adata.obs_names[indices],
            )
            return predictions

    def _set_indices_and_labels(self):
        """Set indices for labeled and unlabeled cells."""
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping

        # set unlabeled and labeled indices
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category_).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = dict(enumerate(self._label_mapping))
