"""Main module."""

from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import pyro
import torch
import torch.nn.functional as F
from pyro.distributions import (
    Categorical,
    Delta,
    Dirichlet,
    Exponential,
    Gamma,
    Independent,
    LogNormal,
    Multinomial,
    Normal,
    Poisson,
    constraints,
)
from pyro.nn import PyroModule
from pyro.infer import Trace_ELBO

from scvi import REGISTRY_KEYS
from scvi.dataloaders import AnnTorchDataset
from scvi.module._classifier import Classifier
from scvi.module.base import PyroBaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder

_RESOLVAE_PYRO_MODULE_NAME = "resolvae"


from pyro.infer import Trace_ELBO

class ControlPenaltyELBO(Trace_ELBO):
    def __init__(self, control_penalty_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.control_penalty_weight = control_penalty_weight

    def loss(self, model, guide, *args, **kwargs):
        """
        Computes the ELBO loss with control penalty.
        """
        elbo_loss = 0.0
        control_penalty = 0.0

        # Use parent class's _get_traces method
        for guide_trace, model_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = guide_trace.log_prob_sum() - model_trace.log_prob_sum()
            elbo_loss += elbo_particle / self.num_particles

            if "control_penalty" in model_trace.nodes:
                penalty_value = model_trace.nodes["control_penalty"]["value"]
                control_penalty += penalty_value / self.num_particles

        # Return total loss (ELBO + control penalty)
        total_loss = -elbo_loss + control_penalty

        return total_loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        Computes the loss and gradients for the total loss (ELBO + control penalty).
        """
        # Compute total loss
        total_loss = self.loss(model, guide, *args, **kwargs)
        
        # Compute gradients on the total loss
        total_loss.backward()
        
        return total_loss

def _safe_log_norm(x: torch.Tensor, dim: int = 1, keepdim: bool = True, eps: float = 1e-12) -> torch.Tensor:
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


class ShiftNetGeneScale(torch.nn.Module):
    """
    Shift network with per-gene scaling based on expression levels.
    
    Parameters
    ----------
    input_dim
        Input dimension (typically n_latent + perturbation_embed_dim)
    n_genes
        Number of genes
    expression_data
        Expression data to compute gene means for initialization
    hidden_dim
        Hidden layer dimension
    global_k
        Global scaling factor for initialization
    min_scale
        Minimum scale value for any gene
    """
    
    def __init__(
        self, 
        input_dim: int, 
        n_genes: int, 
        expression_data: torch.Tensor,
        hidden_dim: int = 64,
        global_k: float = 4.0, 
        min_scale: float = 0.1
    ):
        super().__init__()
        
        # The shift network that produces raw deltas
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_genes),
            torch.nn.Tanh(),  # Bound outputs to [-1, 1]
        )
        
        # Compute per-gene scaling based on expression means
        with torch.no_grad():
            if expression_data.layout in [torch.sparse_csr, torch.sparse_csc]:
                expression_data = expression_data.to_dense()
            means = torch.mean(expression_data, dim=0).float()
            
            # Build per-gene scale: proportional to mean + floor
            init_scale = means * global_k
            init_scale = torch.clamp(init_scale, min=min_scale)
        
        # Make it learnable so the model can adjust per gene
        self.shift_scale = torch.nn.Parameter(init_scale)
    
    def forward(self, x):
        """
        Forward pass with per-gene scaling.
        
        Parameters
        ----------
        x
            Input tensor [batch_size, input_dim] or [n_particles, batch_size, input_dim]
            
        Returns
        -------
        Per-gene scaled shifts [batch_size, n_genes] or [n_particles, batch_size, n_genes]
        """
        raw_delta = self.net(x)  # shape [batch, n_genes] or [n_particles, batch, n_genes]
        delta = raw_delta * self.shift_scale  # per-gene scaling
        return delta


class RESOLVAEModel(PyroModule):
    """A PyroModule that serves as the model for the RESOLVAE class.

    Parameters
    ----------
    n_input
        Number of input genes
    n_obs
        Number of total input cells
    n_neighbors
        Number of spatial neighbors to consider for diffusion.
    z_encoder
        Shared encoder between model (neighboring cells) and guide.
    expression_anntorchdata
        AnnTorchDataset containing expression data.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    mixture_k
        Number of components in the Mixture-of-Gaussian prior
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    n_labels
        Number of cell-type labels in the dataset
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    semisupervised
        Whether to use a semi-supervised model
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder.
        This option only applies when `n_layers` > 1.
        The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    classifier_parameters
        Parameters for the cell-type classifier
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    prior_true_amount
        Prior for true_proportion.
        Equals Gamma(prior_proportions_rate, prior_proportions_rate/prior_true_amount)
        Default is 1.0
    prior_diffusion_amount
        Prior for diffusion_proportion.
        Equals Gamma(prior_proportions_rate, prior_proportions_rate/prior_diffusion_amount)
        Default is 0.3
    sparsity_diffusion
        Prior for sparsity_diffusion. Controls the concentration of the Dirichlet distribution.
        Equals Gamma(prior_proportions_rate, prior_proportions_rate/sparsity_diffusion)
        Default is 3.0
    background_ratio:
        Prior for background_proportion
        Equals Gamma(prior_proportions_rate,
                     prior_proportions_rate/(10*background_ratio*prior_true_amount))
        Default is 0.1
    prior_proportions_rate:
        Rate parameter for the prior proportions.
    median_distance:
        Kernel size in the RBF kernel to estimate distances between cells and neighbors.
    encode_covariates:
        Whether to concatenate covariates to expression in encoder
    override_mixture_k_in_semisupervised:
        If True (default), automatically sets mixture_k to the number of cell type labels
        when semisupervised=True. If False, respects the user-provided mixture_k value
        even in semisupervised mode. When mixture_k < n_labels in semisupervised mode,
        label information is mapped to available mixture components:
        - If mixture_k=1: all cells use the same mixture component
        - If mixture_k>1: labels are mapped modulo mixture_k to available components
    shift_global_k
        Global scaling factor for per-gene shift initialization. The initial per-gene 
        scale is computed as gene_mean * shift_global_k. Default is 2.0.
    shift_min_scale
        Minimum scale value for any gene in the shift network. Ensures that even 
        low-expressed genes can have meaningful perturbation effects. Default is 0.05.
    
    Notes
    -----
    When using custom mixture_k in semisupervised mode (override_mixture_k_in_semisupervised=False),
    the model will adapt the label conditioning to work with the available mixture components.
    This is useful for perturbation studies where perturbed cells are homogeneous (same cell type)
    and you want to use mixture_k=1 for computational efficiency.
    """

    def __init__(
        self,
        n_input: int,
        n_obs: int,
        n_neighbors: int,
        z_encoder: Encoder,
        expression_anntorchdata: AnnTorchDataset,
        n_batch: int = 0,
        n_hidden: int = 32,
        n_latent: int = 10,
        mixture_k: int = 100,
        n_layers: int = 2,
        n_cats_per_cov: Iterable[int] | None = None,
        n_labels: Iterable[int] | None = None,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        semisupervised: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        classifier_parameters: dict | None = None,
        prior_true_amount: float = 1.0,
        prior_diffusion_amount: float = 0.3,
        sparsity_diffusion: float = 3.0,
        background_ratio: float = 0.1,
        prior_proportions_rate: float = 10.0,
        median_distance: float = 1.0,
        encode_covariates: bool = False,
        n_perturbs: int = 1,
        perturbation_embed_dim: int = 16,
        perturbation_hidden_dim: int = 64,
        perturbation_idx: int | None = None,
        override_mixture_k_in_semisupervised: bool = True,
        control_penalty_weight: float = 10.0,
        shift_global_k: float = 2.0,
        shift_min_scale: float = 0.05,
    ):
        super().__init__(_RESOLVAE_PYRO_MODULE_NAME)
        self.z_encoder = z_encoder
        self.expression_anntorchdata = expression_anntorchdata
        self.override_mixture_k_in_semisupervised = override_mixture_k_in_semisupervised
        self.register_buffer("gene_dummy", torch.ones([n_batch, n_input]))

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.mixture_k = mixture_k
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_obs = n_obs
        self.n_neighbors = n_neighbors
        self.expression_anntorchdata = expression_anntorchdata
        self.semisupervised = semisupervised
        self.eps = torch.tensor(1e-6)
        self.encode_covariates = encode_covariates
        self.n_perturbs = n_perturbs
        self.perturbation_idx = perturbation_idx  # Index of perturbation in cat_covs
        self.control_penalty_weight = control_penalty_weight
        
        # Store key information for background handling
        self.background_key = None  # Will be set by setup_anndata
        self.perturbation_key = None  # Will be set by setup_anndata
        
        # Perturbation embedding and shift network
        self.perturb_emb = torch.nn.Embedding(n_perturbs, perturbation_embed_dim)
        
        # Get expression data for computing gene means
        with torch.no_grad():
            # Sample a few batches to estimate means (more memory efficient for large datasets)
            n_samples = min(1000, len(expression_anntorchdata))
            sample_indices = torch.randperm(len(expression_anntorchdata))[:n_samples]
            sample_data = []
            for idx in sample_indices:
                x_sample = expression_anntorchdata[idx.item()]["X"]
                if isinstance(x_sample, np.ndarray):
                    x_sample = torch.from_numpy(x_sample)
                # Convert sparse tensors to dense for stacking
                if x_sample.layout in [torch.sparse_csr, torch.sparse_csc]:
                    x_sample = x_sample.to_dense()
                sample_data.append(x_sample)
            expression_sample = torch.stack(sample_data)
        
        # Initialize ShiftNetGeneScale with expression data
        self.shift_net_gene_scale = ShiftNetGeneScale(
            input_dim=n_latent + perturbation_embed_dim,
            n_genes=n_input,
            expression_data=expression_sample,
            hidden_dim=perturbation_hidden_dim,
            global_k=shift_global_k,
            min_scale=shift_min_scale
        )

        if self.dispersion == "gene":
            init_px_r = torch.full([n_input], 0.01)
        elif self.dispersion == "gene-batch":
            init_px_r = torch.full([n_input, n_batch], 0.01)
        else:
            raise ValueError(
                f"dispersion must be one of ['gene', 'gene-batch'], but input was {dispersion}."
            )
        self.register_buffer("px_r", init_px_r)

        self.register_buffer("median_distance", torch.tensor(median_distance))
        self.register_buffer("sparsity_diffusion", torch.tensor(sparsity_diffusion))
        self.register_buffer("gene_dummy", torch.ones([n_batch, n_input]))

        if self.semisupervised and override_mixture_k_in_semisupervised:
            mixture_k = n_labels

        self.register_buffer("u_prior_logits", torch.ones([mixture_k]))
        if self.semisupervised:
            self.register_buffer("u_prior_means", torch.zeros([mixture_k, n_latent]))
            self.register_buffer("u_prior_scales", torch.zeros([mixture_k, n_latent]) - 1.0)
        else:
            self.register_buffer("u_prior_means", torch.randn([mixture_k, n_latent]))
            self.register_buffer("u_prior_scales", torch.zeros([mixture_k, n_latent]) - 1.0)

        self.register_buffer("diffusion_scale", torch.tensor([1]))
        self.register_buffer(
            "prior_proportions",
            torch.tensor(
                [
                    prior_true_amount,
                    prior_diffusion_amount,
                    10 * background_ratio * prior_true_amount + 1e-3,
                ]
            ),
        )
        self.register_buffer("prior_proportions_rate", torch.tensor([prior_proportions_rate]))

        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        if self.semisupervised:
            classifier_parameters = classifier_parameters or {}
            self.n_labels = n_labels
            # Classifier takes n_latent as input
            cls_parameters = {
                "n_layers": 0,
                "n_hidden": 128,
                "dropout_rate": 0.0,
            }

            cls_parameters.update(classifier_parameters)
            self.classifier = Classifier(
                n_latent,
                n_labels=n_labels,
                use_batch_norm=False,
                use_layer_norm=True,
                **cls_parameters,
            )

    def _get_fn_args_from_batch(self, tensor_dict: dict[str, torch.Tensor]) -> Iterable | dict:
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        y = tensor_dict[REGISTRY_KEYS.LABELS_KEY].long().ravel()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensor_dict[cat_key] if cat_key in tensor_dict.keys() else None

        # Extract perturbation data separately (not from cat_covs)
        perturbation_data = tensor_dict.get("perturbation", None)
        if perturbation_data is not None:
            perturbation_data = perturbation_data.long().ravel()
            
            # Handle the case where background_key is the same as perturbation_key
            # In this case, the background category (index 1) should be excluded from perturbation logic
            # We need to remap the perturbation codes to exclude background
            if hasattr(self, 'background_key') and hasattr(self, 'perturbation_key'):
                if self.background_key == self.perturbation_key:
                    # Create a mapping that excludes background (index 1)
                    # Original: [control=0, background=1, perturb1=2, perturb2=3, ...]
                    # New: [control=0, perturb1=1, perturb2=2, ...] (background becomes -1 or special value)
                    background_mask = (perturbation_data == 1)  # Background is at index 1
                    
                    # For background cells, set perturbation to a special value (-1) to exclude from shift network
                    # For non-background cells, remap: 0->0 (control), 2->1, 3->2, etc.
                    remapped_perturbation = perturbation_data.clone()
                    remapped_perturbation[background_mask] = -1  # Mark background cells
                    
                    # Remap non-background, non-control cells
                    non_background_mask = ~background_mask & (perturbation_data > 1)
                    if non_background_mask.any():
                        remapped_perturbation[non_background_mask] = perturbation_data[non_background_mask] - 1
                    
                    perturbation_data = remapped_perturbation
        else:
            # If no perturbation data, treat all cells as control (index 0)
            perturbation_data = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        ind_x = tensor_dict[REGISTRY_KEYS.INDICES_KEY].long().ravel()
        distances_n = tensor_dict["distance_neighbor"]
        ind_neighbors = tensor_dict["index_neighbor"].long()

        x_n = self.expression_anntorchdata[ind_neighbors.cpu().numpy().flatten(), :]["X"]
        if isinstance(x_n, np.ndarray):
            x_n = torch.from_numpy(x_n)
        x_n = x_n.to(x.device)

        if x.layout is torch.sparse_csr or x.layout is torch.sparse_csc:
            x = x.to_dense()
        if x_n.layout is torch.sparse_csr or x_n.layout is torch.sparse_csc:
            x_n = x_n.to_dense()
        x_n = x_n.reshape(x.shape[0], -1)
        library = torch.log(torch.sum(x, dim=1, keepdim=True))

        return (), {
            "x": x,
            "ind_x": ind_x,
            "library": library,
            "y": y,
            "batch_index": batch_index,
            "cat_covs": cat_covs,
            "perturbation_data": perturbation_data,
            "x_n": x_n,
            "distances_n": distances_n,
        }

    @auto_move_data
    def model_unconditioned(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        perturbation_data: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        """Full model."""
        sparsity_diffusion = pyro.sample(
            "sparsity_diffusion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / self.sparsity_diffusion,
            ),
        ).mean()

        # Per gene background.
        per_gene_background = pyro.sample(
            "per_gene_background",
            Dirichlet(
                concentration=5.0 * self.gene_dummy,
                validate_args=False,  # Softmax has rounding errors
            ).to_event(1),
        )

        prior_proportions = sparsity_diffusion * self.prior_proportions

        # Proportion of true counts
        true_proportion = pyro.sample(
            "true_proportion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / prior_proportions[0],
            ),
        ).mean()

        # Background proportion
        background_proportion = pyro.sample(
            "background_proportion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / prior_proportions[2],
            ),
        ).mean()

        # Diffusion proportion
        diffusion_proportion = pyro.sample(
            "diffusion_proportion",
            Gamma(
                concentration=self.prior_proportions_rate,
                rate=self.prior_proportions_rate / prior_proportions[1],
            ),
        ).mean()

        # Weights on which range diffusion happens compared to median distance.
        diffusion_scale = pyro.sample("diffuse_scale", Exponential(x.new_ones([1])).to_event(1))

        u_prior_logits = pyro.param("u_prior_logits", self.u_prior_logits)
        u_prior_means = pyro.param("u_prior_means", self.u_prior_means)
        u_prior_scales = pyro.param("u_prior_scales", self.u_prior_scales)

        with pyro.plate("obs_plate", size=n_obs or self.n_obs, subsample_size=x.shape[0], dim=-1):
            # Expected dispersion given distance between cells
            distances = 30.0 * pyro.deterministic(
                "distances",
                torch.exp(
                    -torch.clamp(diffusion_scale * distances_n / self.median_distance, max=20.0)
                )
                + 1e-3,
                event_dim=1,
            )  # clamping here as otherwise gradient not defined
            px_r = 1 / pyro.sample("px_r_inv", Exponential(torch.ones_like(x)).to_event(1))

            per_neighbor_diffusion = pyro.sample(
                "per_neighbor_diffusion",
                Dirichlet(concentration=distances, validate_args=False),  # rounding errors
            )
            with pyro.poutine.scale(scale=5.0):
                mixture_proportions = pyro.sample(
                    "mixture_proportions",
                    Dirichlet(
                        concentration=torch.tensor(
                            [true_proportion, diffusion_proportion, background_proportion],
                            device=x.device,
                        ),
                        validate_args=False,  # Softmax has rounding errors
                    ),
                )

            true_mixture_proportion = pyro.deterministic(
                "true_mixture_proportion", mixture_proportions[..., 0]
            )

            diffusion_mixture_proportion = pyro.deterministic(
                "diffusion_mixture_proportion", mixture_proportions[..., 1]
            )

            background_mixture_proportion = pyro.deterministic(
                "background_mixture_proportion", mixture_proportions[..., 2]
            )

            v = pyro.deterministic(
                "diffusion_proportion_per_neighbor",
                per_neighbor_diffusion * diffusion_mixture_proportion.unsqueeze(-1),
                event_dim=1,
            )

            background = pyro.deterministic(
                "background",
                background_mixture_proportion.unsqueeze(-1)
                * torch.exp(library)
                * torch.matmul(
                    torch.nn.functional.one_hot(batch_index.flatten(), self.n_batch).float(),
                    per_gene_background,
                ),
                event_dim=1,
            )

            if self.semisupervised:
                # Get the actual mixture_k from the registered buffers
                current_mixture_k = u_prior_logits.shape[0]
                
                if current_mixture_k == self.n_labels:
                    # Standard case: mixture_k equals n_labels
                    logits_input = (
                        torch.stack(
                            [
                                torch.nn.functional.one_hot(y_i, self.n_labels)
                                if y_i < self.n_labels
                                else torch.zeros(self.n_labels).to(x.device)
                                for y_i in y
                            ]
                        )
                        .to(x.device)
                        .float()
                    )
                    u_prior_logits = u_prior_logits + 10 * logits_input
                else:
                    # Custom mixture_k case: map labels to available mixture components
                    if current_mixture_k == 1:
                        # Single mixture component: all cells use the same mixture component
                        logits_input = torch.ones(x.shape[0], 1, device=x.device)
                    else:
                        # Multiple mixture components: map labels modulo mixture_k
                        mapped_labels = torch.clamp(y % current_mixture_k, 0, current_mixture_k - 1)
                        logits_input = torch.nn.functional.one_hot(mapped_labels, current_mixture_k).float()
                    
                    u_prior_logits = u_prior_logits + 10 * logits_input
                
                u_prior_means = u_prior_means.expand(x.shape[0], -1, -1)
                u_prior_scales = u_prior_scales.expand(x.shape[0], -1, -1)
            cats = Categorical(logits=u_prior_logits)

            normal_dists = Independent(
                Normal(u_prior_means, torch.exp(self.u_prior_scales) + 1e-4),
                reinterpreted_batch_ndims=1,
            )

            # sample from prior (value will be sampled by guide when computing the ELBO)
            with pyro.poutine.scale(scale=kl_weight):
                z = pyro.sample("latent", pyro.distributions.MixtureSameFamily(cats, normal_dists))
            
            # get the "normalized" mean of the negative binomial
            # Note: perturbations do NOT go through encoder/decoder, only through shift network
            if cat_covs is not None:
                categorical_input = list(torch.split(cat_covs, 1, dim=1))
            else:
                categorical_input = ()
            px_scale, _, px_rate, _ = self.decoder(
                self.dispersion,
                z,
                library,
                batch_index,
                *categorical_input,
            )

            # Initialize delta (shifts) - all zeros by default
            if z.ndim == 2:
                delta = torch.zeros(z.shape[0], self.n_input, device=z.device)
            else:
                delta = torch.zeros(z.shape[0], z.shape[1], self.n_input, device=z.device)
            
            if perturbation_data is not None:
                px_rate_pre_shift = px_rate.clone()
                
                # Only process perturbed cells (not control, not background) through shift network
                # Control cells: perturbation_data == 0 (skip shift network entirely)
                # Background cells: perturbation_data == -1 (skip shift network entirely)
                # Perturbed cells: perturbation_data > 0 (process through shift network)
                perturbed_mask = (perturbation_data > 0)
                
                if perturbed_mask.any():
                    # Only process perturbed cells through shift network
                    perturbed_perturbation_data = perturbation_data[perturbed_mask]
                    perturbed_z = z[perturbed_mask] if z.ndim == 2 else z[:, perturbed_mask, :]
                    
                    u_k = self.perturb_emb(perturbed_perturbation_data.long())  # → (n_perturbed_cells, D_s)
                    
                    # Handle dimension mismatch when z has particle dimension (vectorized ELBO)
                    if perturbed_z.ndim == 3 and u_k.ndim == 2:
                        # perturbed_z: [n_particles, n_perturbed_cells, n_latent], u_k: [n_perturbed_cells, embedding_dim]
                        # Expand u_k to match perturbed_z's dimensions
                        u_k = u_k.unsqueeze(0).expand(perturbed_z.shape[0], -1, -1)  # → [n_particles, n_perturbed_cells, embedding_dim]
                    
                    perturbed_delta = self.shift_net_gene_scale(torch.cat([perturbed_z, u_k], dim=-1))  # Scale constrained output
                    
                    # Apply shifts only to perturbed cells, leave control and background cells unchanged
                    if z.ndim == 2:
                        delta[perturbed_mask] = perturbed_delta
                    else:
                        delta[:, perturbed_mask, :] = perturbed_delta
                
                # Debug: Check what's happening
                control_mask = (perturbation_data == 0)
                background_mask = (perturbation_data == -1)
                
                # Apply shifts to ALL cells (control cells get zero shifts, perturbed cells get computed shifts)
                # Work in log-space: log(λ) = log(λ_base) + δ, then λ = exp(log(λ_base) + δ)
                # This ensures λ > 0 even with negative δ
                log_px_rate_base = torch.log(px_rate + 1e-12)  # Add small epsilon to avoid log(0)
                log_px_rate = log_px_rate_base + delta  # Apply shifts (zero for control/background, computed for perturbed)
                px_rate = torch.exp(log_px_rate)
            else:
                # No perturbation data - delta remains all zeros
                pass
            
            # For control penalty, we want to penalize any non-zero shifts for control cells
            # Since control cells should never have shifts, any non-zero shifts for them should be penalized
            # But with the new logic, control cells will always have zero shifts by design
            control_mask_for_penalty = (perturbation_data == 0).float().unsqueeze(-1) if perturbation_data is not None else torch.zeros(z.shape[0], 1, device=z.device)
            if z.ndim == 3:  # particles dimension
                control_mask_for_penalty = control_mask_for_penalty.unsqueeze(0)
            control_shifts = delta * control_mask_for_penalty  # Should always be zero now
            
            # Use sample with Delta distribution and observe it to ensure it appears in trace
            pyro.sample("control_shifts", Delta(control_shifts).to_event(1), obs=control_shifts)
            
            # Add control penalty as a deterministic node for ELBO computation
            control_penalty = torch.mean(control_shifts ** 2) * self.control_penalty_weight

            pyro.deterministic("control_penalty", control_penalty)

            if self.semisupervised:
                probs_prediction_ = self.classifier(z)

            # Stored for use in residual model.
            px_rate = pyro.deterministic("px_rate", px_rate, event_dim=1)
            pyro.deterministic("px_scale", px_scale, event_dim=1)

            # Set model to eval mode. Best estimate of neighbor cells.
            # Autoencoder for all neighboring cells. Autoencoder is learned above.

            # sample from prior for neighboring cells (mode collapse when gradient used)
            with torch.no_grad():
                if cat_covs is not None:
                    categorical_input = [
                        i.repeat_interleave(self.n_neighbors).unsqueeze(1)
                        for i in torch.split(cat_covs, 1, dim=1)
                    ]
                else:
                    categorical_input = ()
                if cat_covs is not None and self.encode_covariates:
                    categorical_encoder = categorical_input
                else:
                    categorical_encoder = ()

                # Add numerical stability for neighboring cell processing
                x_n_log = _safe_log_norm(x_n)
                x_n_input = torch.reshape(x_n_log, (x.shape[0] * self.n_neighbors, x.shape[1]))
                
                qz_m_n, qz_v_n, _ = self.z_encoder(
                    x_n_input,
                    batch_index.repeat_interleave(self.n_neighbors).unsqueeze(1),
                    *categorical_encoder,
                )

                if z.ndim == 2:
                    zn = Normal(
                        qz_m_n.reshape(x.shape[0], self.n_neighbors, self.n_latent),
                        torch.sqrt(qz_v_n.reshape(x.shape[0], self.n_neighbors, self.n_latent)),
                    ).sample()
                    _, _, px_rate_n, _ = self.decoder(
                        self.dispersion,
                        zn.reshape([x.shape[0] * self.n_neighbors, self.n_latent]),
                        library.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        batch_index.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        *categorical_input,
                    )
                    px_rate_n = px_rate_n.reshape([x.shape[0], self.n_neighbors, self.n_input])
                else:
                    zn = Normal(
                        qz_m_n.reshape(x.shape[0], self.n_neighbors, self.n_latent),
                        torch.sqrt(qz_v_n.reshape(x.shape[0], self.n_neighbors, self.n_latent)),
                    ).sample([z.shape[0]])
                    _, _, px_rate_n, _ = self.decoder(
                        self.dispersion,
                        zn.reshape([z.shape[0], x.shape[0] * self.n_neighbors, self.n_latent]),
                        library.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        batch_index.repeat_interleave(self.n_neighbors).unsqueeze(1),
                        *categorical_input,
                    )
                    px_rate_n = px_rate_n.reshape(
                        [z.shape[0], x.shape[0], self.n_neighbors, self.n_input]
                    )

                px_rate_n = pyro.deterministic("px_rate_n", px_rate_n, event_dim=2)

            # Collecting all means. Sample by v from neighboring cells.
            px_rate_sum = torch.sum(
                torch.cat(
                    [
                        (true_mixture_proportion.unsqueeze(-1) * px_rate).unsqueeze(-2),
                        v.unsqueeze(-1) * px_rate_n,
                    ],
                    dim=-2,
                ),
                dim=-2,
            )
            if self.gene_likelihood == "poisson":
                mean_nb = Delta(px_rate_sum, event_dim=1).rsample()
            else:
                # Fix the rate parameter to avoid constraint violations
                rate_param = torch.clamp(px_r / (px_rate_sum + self.eps), min=self.eps, max=1e6)
                mean_nb = (
                    Gamma(concentration=px_r, rate=rate_param)
                    .to_event(1)
                    .rsample()
                )

            mean_poisson = pyro.deterministic(
                "mean_poisson",
                mean_nb + background,
                event_dim=1,  # batch_size, n_genes
            )

            # Sample count distribution
            pyro.sample("obs", Poisson(mean_poisson + 1e-9).to_event(1))

            if self.semisupervised:
                probs_prediction = pyro.deterministic(
                    "probs_prediction",
                    probs_prediction_,
                    event_dim=1,  # batch_size, n_labels
                )

                # Last label is unknown class.
                is_observed = y != self.n_labels
                valid_data = y.clone()
                valid_data[~is_observed] = 0

                with pyro.poutine.scale(scale=50.0):
                    with pyro.poutine.mask(mask=is_observed):
                        pyro.sample(
                            "prediction", Categorical(probs=probs_prediction), obs=valid_data
                        )

    @auto_move_data
    def forward(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        perturbation_data: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        """Forward pass."""
        # Using condition handle for training, this is the reconstruction loss.
        pyro.condition(self.model_unconditioned, data={"obs": x})(
            x, ind_x, library, y, batch_index, cat_covs, perturbation_data, x_n, distances_n, n_obs, kl_weight
        )

    @auto_move_data
    def model_corrected(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        perturbation_data: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        pyro.condition(
            self.model_unconditioned,
            data={
                "background_mixture_proportion": torch.zeros(x.shape[0], device=x.device),
                "diffusion_mixture_proportion": torch.zeros(x.shape[0], device=x.device),
                "true_mixture_proportion": torch.ones(x.shape[0], device=x.device),
            },
        )(x, ind_x, library, y, batch_index, cat_covs, perturbation_data, x_n, distances_n, n_obs, kl_weight)

    @auto_move_data
    def model_residuals(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        perturbation_data: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        pyro.condition(
            self.model_unconditioned,
            data={
                "true_mixture_proportion": torch.zeros(x.shape[0], device=x.device),
            },
        )(x, ind_x, library, y, batch_index, cat_covs, perturbation_data, x_n, distances_n, n_obs, kl_weight)

    @auto_move_data
    def model_simplified(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        perturbation_data: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
        corrected_rate: bool = False,
        observed_rate: torch.Tensor = None,
    ):
        if observed_rate is not None:
            x = observed_rate

        hide = [
            "per_gene_background",
            "diffuse_scale",
            "sparsity_diffusion",
            "per_neighbor_diffusion",
            "mixture_proportions",
            "distances",
            "px_r_inv",
            "prediction",
            "true_proportion",
            "background_proportion",
            "diffusion_proportion",
        ]
        simplified_model = pyro.poutine.block(self.model_unconditioned, hide=hide)

        with pyro.poutine.scale(scale=0.01 * x.shape[0] / self.n_obs):
            if corrected_rate:
                pyro.condition(
                    simplified_model,
                    data={
                        "background_mixture_proportion": torch.zeros(x.shape[0], device=x.device),
                        "diffusion_mixture_proportion": torch.zeros(x.shape[0], device=x.device),
                        "true_mixture_proportion": torch.ones(x.shape[0], device=x.device),
                    },
                )(x, ind_x, library, y, batch_index, cat_covs, perturbation_data, x_n, distances_n, n_obs, kl_weight)
            else:
                simplified_model(
                    x, ind_x, library, y, batch_index, cat_covs, perturbation_data, x_n, distances_n, n_obs, kl_weight
                )


class RESOLVAEGuide(PyroModule):
    """A PyroModule that serves as the guide for the RESOLVAE class.

    Parameters
    ----------
    n_input
        Number of input genes
    n_obs
        Number of total input cells
    n_neighbors
        Number of spatial neighbors to consider for diffusion.
    z_encoder
        Shared encoder between model (neighboring cells) and guide.
    n_latent
        Dimensionality of the latent space.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    n_hidden_encoder
        Number of nodes per hidden layer in the encoder.
    n_cats_per_cov
        Number of categories for each extra categorical covariate.
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    downsample_counts_mean
        Mean of the log-normal distribution used to downsample counts.
    downsample_counts_std
        Standard deviation of the log-normal distribution used to downsample counts.
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder.
        This option only applies when `n_layers` > 1.
        The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    median_distance:
        Kernel size in the RBF kernel to estimate distances between cells and neighbors.
    diffusion_eps:
        Epsilon value for diffusion.
    """

    def __init__(
        self,
        n_input: int,
        n_obs: int,
        n_neighbors: int,
        z_encoder: Encoder,
        n_batch: int = 0,
        n_latent: int = 10,
        n_layers: int = 2,
        n_hidden_encoder: int = 128,
        n_cats_per_cov: Iterable[int] | None = None,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        downsample_counts_mean: int | None = None,
        downsample_counts_std: float = 1.0,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        median_distance: float = 1.0,
        diffusion_eps: float = 0.01,
    ):
        super().__init__(_RESOLVAE_PYRO_MODULE_NAME)
        self.dispersion = dispersion
        self.z_encoder = z_encoder
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.encode_covariates = encode_covariates
        self.n_input = n_input
        self.n_obs = n_obs
        self.n_neighbors = n_neighbors
        self.median_distance = median_distance
        self.downsample_counts_mean = downsample_counts_mean
        self.downsample_counts_std = downsample_counts_std

        if self.dispersion == "gene":
            init_px_r = torch.full([n_input], 0.01)
        elif self.dispersion == "gene-batch":
            init_px_r = torch.full([n_input, n_batch], 0.01)
        else:
            raise ValueError(
                f"dispersion must be one of ['gene', 'gene-batch'], but input was {dispersion}."
            )
        self.register_buffer("px_r", init_px_r)
        self.register_buffer("per_neighbor_diffusion_init", torch.zeros([n_obs, n_neighbors]))
        self.register_buffer("gene_dummy", torch.ones([n_batch, n_input]))
        self.eps = torch.tensor(1e-6)
        self.diffusion_eps = diffusion_eps

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.diffusion_encoder = Encoder(
            n_input,
            3,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_encoder,
            dropout_rate=0.0,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=torch.nn.Softmax(dim=-1),
            var_eps=1e-3,
        )

    @auto_move_data
    def forward(  # not used arguments to have same set of arguments in model and guide
        self,
        x,
        ind_x,
        library,
        y,
        batch_index,
        cat_covs,
        perturbation_data,
        x_n,
        distances_n,
        n_obs=None,
        kl_weight=1.0,
    ):
        # Amount of true in total counts of Dirichlet
        sparsity_diffusion_est = pyro.param(
            "sparsity_diffusion_est",
            20.0 * x.new_ones([1]),
            constraint=constraints.greater_than(1e-3),
        )
        pyro.sample("sparsity_diffusion", Delta(sparsity_diffusion_est))

        background_concentration = torch.softmax(
            pyro.param("background_concentration", self.gene_dummy, event_dim=2),
            dim=-1,
        )

        # Per gene poisson rate for background
        pyro.sample("per_gene_background", Delta(background_concentration, event_dim=2))

        # Amount of background in total counts of Dirichlet
        background_proportion_est = pyro.param(
            "background_proportion_est",
            0.5 * x.new_ones([1]),
            constraint=constraints.greater_than(1e-6),
        )
        pyro.sample("background_proportion", Delta(background_proportion_est))

        # Amount of diffusion in total counts of Dirichlet
        diffusion_proportion_est = pyro.param(
            "diffusion_proportion_est",
            3.0 * x.new_ones([1]),
            constraint=constraints.greater_than(1e-6),
        )
        pyro.sample("diffusion_proportion", Delta(diffusion_proportion_est))

        # Amount of true in total counts of Dirichlet
        true_proportion_est = pyro.param(
            "true_proportion_est", 5.0 * x.new_ones([1]), constraint=constraints.greater_than(1e-6)
        )
        pyro.sample("true_proportion", Delta(true_proportion_est))

        # Weights to how many neighbors diffusion happens in relation to median distance.
        diffusion_scale_est = pyro.param(
            "diffuse_scale_est",
            x.new_ones([1]),
            constraint=constraints.greater_than(self.eps),
            event_dim=1,
        )
        pyro.sample("diffuse_scale", Delta(diffusion_scale_est, event_dim=1))

        # Weights to which neighbor diffusion happens.
        per_neighbor_diffusion = pyro.param(
            "per_neighbor_diffusion_map",
            self.per_neighbor_diffusion_init,
            constraint=constraints.interval(-10.0, 10.0),
            event_dim=1,
        )

        if self.downsample_counts_mean is not None:
            downsample_counts = (
                int(LogNormal(float(self.downsample_counts_mean), float(self.downsample_counts_std)).sample())
                + 10
            )

        with pyro.plate("obs_plate", size=n_obs or self.n_obs, subsample=ind_x, dim=-1):
            # Dispersion of NB for counts.
            px_r_mle = pyro.param(
                "px_r_mle",
                self.px_r,
                constraint=constraints.greater_than(self.eps),
                event_dim=len(self.px_r.shape),
            )

            if self.dispersion == "gene-batch":
                px_r_inv = F.linear(
                    torch.nn.functional.one_hot(batch_index.flatten(), self.n_batch).to(
                        px_r_mle.dtype
                    ),
                    px_r_mle,
                )
            elif self.dispersion == "gene":
                px_r_inv = px_r_mle
            pyro.sample("px_r_inv", Delta(px_r_inv, event_dim=1))
            # Expected diffusion given distance between cells
            concentration = torch.nn.Softmax(dim=-1)(
                per_neighbor_diffusion[ind_x, :]
                - torch.clamp(torch.sqrt(distances_n / self.median_distance), max=10.0)
            )

            pyro.sample("per_neighbor_diffusion", Delta(concentration, event_dim=1))

            if cat_covs is not None and self.encode_covariates:
                categorical_input = list(torch.split(cat_covs, 1, dim=1))
            else:
                categorical_input = ()

            with pyro.poutine.scale(scale=5.0):
                _, mixture_proportions_est, _ = self.diffusion_encoder(
                    torch.log1p(x), batch_index, *categorical_input
                )
                # Set minimum diffusion to 0.01. This helps with stability
                mixture_proportions_est[..., 1] += self.diffusion_eps
                pyro.sample("mixture_proportions", Delta(mixture_proportions_est, event_dim=1))
            with pyro.poutine.scale(scale=kl_weight):
                # use the encoder to get the parameters used to define q(z|x)
                if self.training and self.downsample_counts_mean is not None:
                    x = Multinomial(total_count=downsample_counts, probs=x).sample()
                # Add numerical stability
                x_log = _safe_log_norm(x)
                
                qz_m, qz_v, _ = self.z_encoder(
                    x_log,
                    batch_index,
                    *categorical_input,
                )
                # sample the latent code z
                pyro.sample("latent", Normal(qz_m, torch.sqrt(qz_v)).to_event(1))

    @auto_move_data
    def guide_simplified(
        self,
        x: torch.Tensor,
        ind_x: torch.Tensor,
        library: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor,
        cat_covs: torch.Tensor,
        perturbation_data: torch.Tensor,
        x_n: torch.Tensor,
        distances_n: torch.Tensor,
        n_obs: int | None = None,
        kl_weight: float = 1.0,
    ):
        simplified_guide = pyro.poutine.block(
            self.forward,
            expose=["latent"],
            hide=[
                "sparsity_diffusion",
                "per_gene_background",
                "background_proportion",
                "diffusion_proportion",
                "true_proportion",
                "diffuse_scale",
                "px_r_inv",
                "per_neighbor_diffusion",
                "mixture_proportions",
            ],
        )

        with pyro.poutine.scale(scale=x.shape[0] / self.n_obs):
            simplified_guide(
                x, ind_x, library, y, batch_index, cat_covs, perturbation_data, x_n, distances_n, n_obs, kl_weight
            )


class RESOLVAE(PyroBaseModuleClass):
    """
    Implementation of resolVI.

    Parameters
    ----------
    n_input
        Number of input genes
    n_obs
        Number of total input cells
    n_neighbors
        Number of spatial neighbors to consider for diffusion.
    expression_anntorchdata
        AnnTorchDataset with expression data.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer in the decoder
    n_hidden_encoder
        Number of nodes per hidden layer in the encoder
    n_latent
        Dimensionality of the latent space
    mixture_k
        Number of components in the Mixture-of-Gaussian prior
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    n_labels
        Number of cell-type labels in the dataset
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    semi_supervised
        Whether to use a semi-supervised model
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder.
        This option only applies when `n_layers` > 1.
        The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    classifier_parameters
        Parameters for the classifier
    prior_true_amount
        Prior for true_proportion.
        Equals Gamma(prior_proportions_rate, prior_proportions_rate/prior_true_amount)
        Default is 1.0
    prior_diffusion_amount
        Prior for diffusion_proportion.
        Equals Gamma(prior_proportions_rate, prior_proportions_rate/prior_diffusion_amount)
        Default is 0.3
    sparsity_diffusion
        Prior for sparsity_diffusion. Controls the concentration of the Dirichlet distribution.
        Equals Gamma(prior_proportions_rate, prior_proportions_rate/sparsity_diffusion)
        Default is 3.0
    background_ratio:
        Prior for background_proportion
        Equals Gamma(prior_proportions_rate,
                     prior_proportions_rate/(10*background_ratio*prior_true_amount))
        Default is 0.1
    prior_proportions_rate:
        Rate parameter for the prior proportions.
    median_distance:
        Kernel size in the RBF kernel to estimate distances between cells and neighbors.
    downsample_counts_mean:
        Mean of the log-normal distribution used to downsample counts.
    downsample_counts_std:
        Standard deviation of the log-normal distribution used to downsample counts.
    diffusion_eps:
        Epsilon value for diffusion. Creates an offset to stabilize training.
    encode_covariates:
        Whether to concatenate covariates to expression in encoder
    latent_distribution:
        Placeholder for compatibility with other models.
    """

    def __init__(
        self,
        n_input: int,
        n_obs: int,
        n_neighbors: int,
        expression_anntorchdata: AnnTorchDataset,
        n_batch: int = 0,
        n_hidden: int = 32,
        n_hidden_encoder: int = 128,
        n_latent: int = 10,
        mixture_k: int = 30,
        n_layers: int = 2,
        n_cats_per_cov: Iterable[int] | None = None,
        n_labels: Iterable[int] | None = None,
        dropout_rate: float = 0.05,
        dispersion: Literal["gene", "gene-batch"] = "gene",
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        semisupervised: bool = False,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        var_activation: Callable | None = None,
        classifier_parameters: dict | None = None,
        prior_true_amount: float = 1.0,
        prior_diffusion_amount: float = 0.3,
        sparsity_diffusion: float = 3.0,
        background_ratio: float = 0.1,
        prior_proportions_rate: float = 10.0,
        median_distance: float = 1.0,
        downsample_counts_mean: float | None = None,
        downsample_counts_std: float = 1.0,
        diffusion_eps: float = 0.01,
        n_perturbs: int = 1,
        perturbation_embed_dim: int = 16,
        perturbation_hidden_dim: int = 64,
        perturbation_idx: int | None = None,
        override_mixture_k_in_semisupervised: bool = True,
        control_penalty_weight: float = 10.0,
        shift_global_k: float = 2.0,
        shift_min_scale: float = 0.05,
        latent_distribution: str | None = None,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.mixture_k = mixture_k
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_obs = n_obs
        self.n_neighbors = n_neighbors
        self.expression_anntorchdata = expression_anntorchdata
        self.semisupervised = semisupervised
        self.eps = torch.tensor(1e-6)
        self.encode_covariates = encode_covariates

        use_batch_norm_encoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if self.encode_covariates else None

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_encoder,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
        )

        self._guide = RESOLVAEGuide(
            z_encoder=self.z_encoder,
            n_input=n_input,
            n_obs=n_obs,
            n_neighbors=n_neighbors,
            n_batch=n_batch,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden_encoder=n_hidden_encoder,
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            median_distance=median_distance,
            downsample_counts_mean=downsample_counts_mean,
            downsample_counts_std=downsample_counts_std,
            diffusion_eps=diffusion_eps,
        )

        self._model = RESOLVAEModel(
            n_input=n_input,
            n_obs=n_obs,
            n_neighbors=n_neighbors,
            z_encoder=self.z_encoder,
            expression_anntorchdata=expression_anntorchdata,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            mixture_k=mixture_k,
            n_layers=n_layers,
            n_cats_per_cov=n_cats_per_cov,
            n_labels=n_labels,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            semisupervised=semisupervised,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            classifier_parameters=classifier_parameters,
            prior_true_amount=prior_true_amount,
            prior_diffusion_amount=prior_diffusion_amount,
            sparsity_diffusion=sparsity_diffusion,
            background_ratio=background_ratio,
            prior_proportions_rate=prior_proportions_rate,
            median_distance=median_distance,
            n_perturbs=n_perturbs,
            perturbation_embed_dim=perturbation_embed_dim,
            perturbation_hidden_dim=perturbation_hidden_dim,
            perturbation_idx=perturbation_idx,
            override_mixture_k_in_semisupervised=override_mixture_k_in_semisupervised,
            control_penalty_weight=control_penalty_weight,
            shift_global_k=shift_global_k,
            shift_min_scale=shift_min_scale,
        )
        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        return self._model

    @property
    def model_corrected(self):
        return self._model.model_corrected

    @property
    def model_residuals(self):
        return self._model.model_residuals

    @property
    def model_unconditioned(self):
        return self._model.model_unconditioned

    @property
    def model_simplified(self):
        return self._model.model_simplified

    @property
    def guide(self):
        return self._guide

    @property
    def guide_simplified(self):
        return self._guide.guide_simplified

    @property
    def list_obs_plate_vars(self):
        """
        Simplified plates adopted from Cell2location.

        1. "name" - the name of observation/minibatch plate;
        2. "event_dim" - the number of event dimensions.
        """
        return {
            "name": "obs_plate",
            "event_dim": 1,
        }