"""Functional translation of the ``test_spatial_perturb`` notebook.

The original notebook performed a series of exploratory analyses with the
RESOLVI model. This module factors the workflow into reusable functions that
can be imported or executed from the command line.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scvi.external.resolvi as RESOLVI  # noqa: E402


def _ensure_matplotlib_figure(obj: Any) -> plt.Figure:
    """Best-effort conversion of Scanpy/seaborn outputs into Matplotlib figures."""

    if isinstance(obj, plt.Figure):
        return obj
    if hasattr(obj, "figure") and isinstance(obj.figure, plt.Figure):
        return obj.figure
    return plt.gcf()


def render_scanpy_plot(
    plot_fn: Callable[..., Any],
    *args: Any,
    show: bool = True,
    **kwargs: Any,
) -> plt.Figure:
    """Call a Scanpy plotting function and capture the backing Matplotlib figure."""

    try:
        obj = plot_fn(*args, show=show, return_fig=True, **kwargs)
    except (TypeError, AttributeError):
        # Some Scanpy wrappers forward unexpected kwargs downstream; retry without return_fig.
        obj = plot_fn(*args, show=show, **kwargs)
    return _ensure_matplotlib_figure(obj)


@dataclass
class SpatialPerturbationConfig:
    """Configuration bundle for the RESOLVI perturbation analysis."""

    adata_path: Path
    target_gene: str = "Ccl5"
    control_label: str = "Other cells"
    control_perturbation: str = "sgCd19"
    spatial_embedding_key: str | None = "X_cellcharter"
    n_samples: int = 1000
    resolvi_setup_kwargs: Dict[str, object] = field(default_factory=dict)
    resolvi_model_kwargs: Dict[str, object] = field(default_factory=dict)
    training_kwargs: Dict[str, object] = field(default_factory=dict)
    umap_colors: Sequence[str] = field(
        default_factory=lambda: ("guide_rnas", "Klf2", "Perturbation_10", "Perturbation_4")
    )
    target_violin_genes: Sequence[str] = field(
        default_factory=lambda: ("Klf2", "Itgae", "Gzma", "Cd8a", "Perturbation_10")
    )
    target_violin_genes_secondary: Sequence[str] = field(
        default_factory=lambda: ("Ccl5", "Cxcr3", "Ccr5", "Cxcr6", "Perturbation_4")
    )
    global_violin_genes: Sequence[str] = field(
        default_factory=lambda: ("Gzma", "Ccl5", "Ccr5", "Cxcr6", "Xcl1", "Neurog3")
    )
    matrixplot_genes: Sequence[str] = field(
        default_factory=lambda: ("Gzma", "Ccl5", "Ccr5", "Cxcr6", "Xcl1", "Neurog3")
    )
    output_root: Path | None = None

    def __post_init__(self) -> None:
        self.adata_path = Path(self.adata_path).expanduser()
        if self.output_root is None:
            self.output_root = Path(__file__).resolve().parent / "figures"
        else:
            self.output_root = Path(self.output_root).expanduser()
        if not self.resolvi_setup_kwargs:
            self.resolvi_setup_kwargs = {
                "labels_key": "resolvi_predicted",
                "layer": "counts",
                "batch_key": "batch",
                "perturbation_key": "guide_rnas",
                "control_perturbation": self.control_perturbation,
                "background_key": "guide_rnas",
                "background_category": self.control_label,
            }
        if (
            self.spatial_embedding_key is not None
            and "spatial_embedding_key" not in self.resolvi_setup_kwargs
        ):
            self.resolvi_setup_kwargs["spatial_embedding_key"] = self.spatial_embedding_key
        if not self.resolvi_model_kwargs:
            self.resolvi_model_kwargs = {
                "semisupervised": True,
                "mixture_k": 1,
                "n_latent": 10,
                "override_mixture_k_in_semisupervised": False,
                "control_penalty_weight": 10_000_000,
            }
        if not self.training_kwargs:
            self.training_kwargs = {
                "max_epochs": 300,
                "check_val_every_n_epoch": 1,
                "log_every_n_steps": 1,
                "lr": 3e-4,
                "train_on_perturbed_only": True,
            }


def load_query_data(path: Path) -> sc.AnnData:
    """Load the reference AnnData object from disk."""

    sc.set_figure_params(dpi=300)
    return sc.read(path)


def setup_resolvi_model(adata: sc.AnnData, cfg: SpatialPerturbationConfig) -> RESOLVI.RESOLVI:
    """Prepare AnnData fields and instantiate the RESOLVI model."""

    RESOLVI.RESOLVI.setup_anndata(adata, **cfg.resolvi_setup_kwargs)
    model = RESOLVI.RESOLVI(adata, **cfg.resolvi_model_kwargs)
    return model


def train_resolvi_model(model: RESOLVI.RESOLVI, training_kwargs: Mapping[str, object]) -> None:
    """Ensure downsampling hyperparameters are floats and train the model."""

    guide = model.module.guide
    guide.downsample_counts_mean = float(guide.downsample_counts_mean)
    guide.downsample_counts_std = float(guide.downsample_counts_std)
    model.train(**training_kwargs)


def subset_non_background(adata: sc.AnnData, perturbation_key: str, control_label: str) -> sc.AnnData:
    """Return a copy of the AnnData without the background control label."""

    mask = adata.obs[perturbation_key] != control_label
    return adata[mask].copy()


def add_denoised_layers(
    model: RESOLVI.RESOLVI, mini_adata: sc.AnnData, n_samples: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach denoised expression matrices with and without shifts."""

    control_expr = model.get_denoised_expression_control(mini_adata, n_samples=n_samples)
    mini_adata.layers["resolvi_expression_no_shift"] = control_expr.loc[mini_adata.obs_names].values

    perturbed_expr = model.get_denoised_expression_perturbed(mini_adata, n_samples=n_samples)
    mini_adata.layers["resolvi_expression_with_shift"] = perturbed_expr.loc[mini_adata.obs_names].values

    return control_expr, perturbed_expr


def plot_umap_panels(
    mini_adata: sc.AnnData, colors: Iterable[str], show: bool = True
) -> Dict[str, object]:
    """Recreate UMAP diagnostics for both denoised layers."""

    colors = tuple(colors)
    plots: Dict[str, object] = {}

    sc.tl.pca(mini_adata, layer="resolvi_expression_no_shift")
    sc.pp.neighbors(mini_adata)
    sc.tl.umap(mini_adata)
    plots["umap_control_layer"] = render_scanpy_plot(
        sc.pl.umap,
        mini_adata,
        color=colors,
        layer="resolvi_expression_no_shift",
        ncols=2,
        show=show,
    )
    plots["umap_control_counts"] = render_scanpy_plot(
        sc.pl.umap,
        mini_adata,
        color=colors,
        ncols=2,
        vmax=5,
        show=show,
    )

    sc.tl.pca(mini_adata, layer="resolvi_expression_with_shift")
    sc.pp.neighbors(mini_adata)
    sc.tl.umap(mini_adata)
    plots["umap_perturbed_layer"] = render_scanpy_plot(
        sc.pl.umap,
        mini_adata,
        color=colors,
        layer="resolvi_expression_with_shift",
        ncols=2,
        show=show,
    )
    plots["umap_perturbed_counts"] = render_scanpy_plot(
        sc.pl.umap,
        mini_adata,
        color=colors,
        ncols=2,
        vmax=1,
        show=show,
    )

    return plots


def evaluate_control_penalty(
    model: RESOLVI.RESOLVI, adata: sc.AnnData, perturbation_key: str, control_label: str
) -> Mapping[str, object]:
    """Mirror the control penalty effectiveness diagnostic."""

    mask = adata.obs[perturbation_key] != control_label
    return model.check_control_penalty_effectiveness(indices=mask)


def get_shift_outputs_for_target(
    model: RESOLVI.RESOLVI, adata: sc.AnnData, perturbation_key: str, target_gene: str
) -> tuple[pd.Series, pd.DataFrame]:
    """Fetch shift-network outputs for a specific perturbation target."""

    label = f"sg{target_gene}"
    indices = adata.obs[perturbation_key] == label
    if not bool(np.any(indices)):
        raise ValueError(f"No cells found for target perturbation '{label}'.")
    shift_outputs = model.get_shift_network_outputs(indices=indices)
    return indices, shift_outputs


def build_shift_adata(
    adata: sc.AnnData,
    shift_outputs: pd.DataFrame,
    perturbation_key: str,
) -> sc.AnnData:
    """Create an AnnData object that mirrors notebook engineering steps."""

    obs = pd.DataFrame({perturbation_key: adata.obs.loc[shift_outputs.index, perturbation_key]}, index=shift_outputs.index)
    var = pd.DataFrame(index=shift_outputs.columns)
    adata_shifts = sc.AnnData(X=shift_outputs.values, obs=obs, var=var)
    return adata_shifts


def add_raw_shift_layer(
    adata_shifts: sc.AnnData,
    control_expr: pd.DataFrame,
    perturbed_expr: pd.DataFrame,
) -> None:
    """Add raw shift residuals (perturbed minus control) to the AnnData."""

    obs_names = adata_shifts.obs_names
    raw_shifts = perturbed_expr.loc[obs_names].values - control_expr.loc[obs_names].values
    adata_shifts.layers["raw_shifts"] = raw_shifts


def summarize_shifts(
    adata_shifts: sc.AnnData,
    query_subset: sc.AnnData,
    counts_layer: str,
    shift_layer: str = "raw_shifts",
) -> pd.DataFrame:
    """Compute per-gene average shift against log mean counts for plotting."""

    shifts = np.asarray(adata_shifts.layers[shift_layer]).mean(axis=0)
    mean_counts = np.asarray(query_subset.layers[counts_layer]).mean(axis=0)
    volcano_df = pd.DataFrame(
        {"shift": shifts, "mean_counts": np.log1p(mean_counts)},
        index=adata_shifts.var_names,
    )
    return volcano_df


def plot_violins(
    adata: sc.AnnData,
    genes: Sequence[str],
    perturbation_key: str,
    layer: str | None = None,
    show: bool = True,
) -> object:
    """Convenience wrapper around ``sc.pl.violin``."""

    available_genes = [gene for gene in genes if gene in adata.var_names]
    missing = sorted(set(genes) - set(available_genes))
    if missing:
        warnings.warn(f"Skipping genes not present in AnnData: {', '.join(missing)}")
    if not available_genes:
        raise ValueError("None of the requested genes are present in the AnnData object.")

    return render_scanpy_plot(
        sc.pl.violin,
        adata,
        available_genes,
        groupby=perturbation_key,
        rotation=90,
        layer=layer,
        show=show,
    )


def plot_matrixplot(
    adata: sc.AnnData,
    genes: Sequence[str],
    groupby: str,
    show: bool = True,
) -> object:
    """Convenience wrapper around ``sc.pl.matrixplot``."""

    available_genes = [gene for gene in genes if gene in adata.var_names]
    missing = sorted(set(genes) - set(available_genes))
    if missing:
        warnings.warn(f"Skipping genes not present in AnnData: {', '.join(missing)}")
    if not available_genes:
        raise ValueError("None of the requested genes are present in the AnnData object.")

    return render_scanpy_plot(
        sc.pl.matrixplot,
        adata,
        var_names=available_genes,
        groupby=groupby,
        standard_scale="var",
        show=show,
    )


def plot_volcano(
    volcano_df: pd.DataFrame,
    highlight_genes: Iterable[str],
    show: bool = True,
) -> plt.Figure:
    """Recreate the volcano-style scatter plot from the notebook."""

    highlight_genes = tuple(highlight_genes)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(volcano_df["mean_counts"], volcano_df["shift"], alpha=0.6, s=30)

    for gene in highlight_genes:
        if gene in volcano_df.index:
            x, y = volcano_df.loc[gene, ["mean_counts", "shift"]]
            ax.annotate(
                gene,
                (x, y),
                fontsize=10,
                alpha=0.9,
                xytext=(5, 5),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.3},
            )

    for gene in volcano_df.index.difference(highlight_genes):
        x, y = volcano_df.loc[gene, ["mean_counts", "shift"]]
        ax.annotate(
            gene,
            (x, y),
            fontsize=8,
            alpha=0.6,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Log Mean Counts", fontsize=12)
    ax.set_ylabel("Shift", fontsize=12)
    ax.set_title("Volcano Plot: Shifts vs Log Mean Counts", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def compare_shift_outputs_effects(
    model: RESOLVI.RESOLVI,
    adata: sc.AnnData,
    perturbation_key: str,
    control_label: str,
) -> Mapping[str, object]:
    """Run RESOLVI's built-in comparison utility on non-control cells."""

    indices = np.where(adata.obs[perturbation_key] != control_label)[0]
    return model.compare_shift_outputs_vs_effects(adata, indices=indices, n_samples=1)


def plot_comparison_panels(results: Mapping[str, object], show: bool = True) -> Dict[str, plt.Figure]:
    """Mirror the exploratory seaborn summaries from the notebook."""

    figures: Dict[str, plt.Figure] = {}

    corr_df = pd.DataFrame.from_dict(results["overall_correlations"], orient="index", columns=["Correlation"])
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax1)
    ax1.set_title("Overall Correlations Between Measures")
    fig1.tight_layout()
    if show:
        plt.show()
    figures["overall_correlations"] = fig1

    all_data = {
        "shift_outputs": [],
        "direct_effects": [],
        "full_effects": [],
        "expected_direct": [],
        "perturbation": [],
    }
    for perturb_key, data in results["comparisons"].items():
        perturb_id = perturb_key.split("_")[1]
        all_data["shift_outputs"].extend(data["shift_outputs"])
        all_data["direct_effects"].extend(data["direct_effects"])
        all_data["full_effects"].extend(data["full_effects"])
        all_data["expected_direct"].extend(data["expected_direct"])
        all_data["perturbation"].extend([perturb_id] * len(data["shift_outputs"]))
    plot_df = pd.DataFrame(all_data)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x="shift_outputs", y="direct_effects", hue="perturbation", alpha=0.6, ax=ax2)
    ax2.set_title("Shift Outputs vs Direct Effects")
    ax2.set_xlabel("Shift Outputs")
    ax2.set_ylabel("Direct Effects")
    ax2.axline((0, 0), slope=1 / np.log(2), color="red", linestyle="--", label="Expected: 1/ln(2)")
    ax2.legend()
    if show:
        plt.show()
    figures["shift_vs_direct"] = fig2

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x="direct_effects", y="full_effects", hue="perturbation", alpha=0.6, ax=ax3)
    correlation = np.corrcoef(plot_df["direct_effects"], plot_df["full_effects"])[0, 1]
    ax3.set_title(f"Direct Effects vs Full Counterfactual Effects (r={correlation:.3f})")
    ax3.set_xlabel("Direct Effects")
    ax3.set_ylabel("Full Effects")
    ax3.axline((0, 0), slope=1, color="red", linestyle="--", label="Expected: Identity")
    ax3.legend()
    if show:
        plt.show()
    figures["direct_vs_full"] = fig3

    gene_avg_expr = []
    full_effects_list = []
    perturbations_list = []
    for perturb_key, data in results["comparisons"].items():
        perturb_id = perturb_key.split("_")[1]
        full_effects_array = np.array(data["full_effects"])
        for gene_effects in full_effects_array:
            gene_avg_expr.append(np.mean(np.abs(gene_effects)))
            full_effects_list.append(np.mean(gene_effects))
            perturbations_list.append(perturb_id)
    expr_df = pd.DataFrame(
        {"gene_avg_expression": gene_avg_expr, "full_effects": full_effects_list, "perturbation": perturbations_list}
    )

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=expr_df, x="gene_avg_expression", y="full_effects", hue="perturbation", alpha=0.6, ax=ax4)
    ax4.set_title("Full Effects vs Gene-wise Average Expression by Perturbation")
    ax4.set_xlabel("Gene-wise Average Expression Level")
    ax4.set_ylabel("Full Effects")
    if show:
        plt.show()
    figures["full_effects_vs_expression"] = fig4

    melt_df = pd.melt(
        plot_df,
        id_vars=["perturbation"],
        value_vars=["shift_outputs", "direct_effects", "full_effects", "expected_direct"],
        var_name="Measure",
        value_name="Value",
    )
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=melt_df, x="Measure", y="Value", hue="perturbation", ax=ax5)
    ax5.set_title("Distribution of Different Measures by Perturbation")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    if show:
        plt.show()
    figures["measure_distributions"] = fig5

    corr_data = {key: data["correlations"] for key, data in results["comparisons"].items()}
    corr_df_per = pd.DataFrame(corr_data).T
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df_per, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax6)
    ax6.set_title("Per-Perturbation Correlations")
    ax6.set_ylabel("Perturbation")
    fig6.tight_layout()
    if show:
        plt.show()
    figures["per_perturbation_correlations"] = fig6

    return figures


def prepare_output_dir(base_dir: Path) -> Path:
    """Create a timestamped output directory for exported figures."""

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_plot_collection(plots: Mapping[str, object], output_dir: Path, prefix: str = "") -> None:
    """Recursively persist figures to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in plots.items():
        current_name = f"{prefix}_{name}" if prefix else name
        if isinstance(obj, Mapping):
            save_plot_collection(obj, output_dir, current_name)
            continue
        if obj is None:
            continue

        fig = None
        if hasattr(obj, "savefig"):
            fig = obj
        elif hasattr(obj, "figure") and hasattr(obj.figure, "savefig"):
            fig = obj.figure

        if fig is None:
            warnings.warn(f"Unable to save figure '{current_name}' (unsupported object type: {type(obj)}).")
            continue

        output_path = output_dir / f"{current_name}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def run_analysis(
    cfg: SpatialPerturbationConfig, show_plots: bool = True, output_dir: Path | None = None
) -> Dict[str, object]:
    """High-level orchestration of the entire perturbation analysis."""

    output_dir = output_dir or prepare_output_dir(cfg.output_root)
    query_adata = load_query_data(cfg.adata_path)
    model = setup_resolvi_model(query_adata, cfg)
    priors = model.compute_dataset_dependent_priors()
    train_resolvi_model(model, cfg.training_kwargs)

    perturbation_key = cfg.resolvi_setup_kwargs["perturbation_key"]
    layer_key = cfg.resolvi_setup_kwargs["layer"]
    mini_adata = subset_non_background(query_adata, perturbation_key, cfg.control_label)
    control_expr, perturbed_expr = add_denoised_layers(model, mini_adata, cfg.n_samples)
    umap_plots = plot_umap_panels(mini_adata, cfg.umap_colors, show=show_plots)

    effectiveness = evaluate_control_penalty(model, query_adata, perturbation_key, cfg.control_label)
    target_indices, shift_outputs = get_shift_outputs_for_target(model, query_adata, perturbation_key, cfg.target_gene)
    adata_shifts = build_shift_adata(query_adata, shift_outputs, perturbation_key)
    add_raw_shift_layer(adata_shifts, control_expr, perturbed_expr)

    violin_primary = plot_violins(adata_shifts, cfg.target_violin_genes, perturbation_key, show=show_plots)
    violin_secondary = plot_violins(
        adata_shifts, cfg.target_violin_genes_secondary, perturbation_key, show=show_plots
    )
    violin_raw = plot_violins(
        adata_shifts,
        cfg.target_violin_genes + cfg.target_violin_genes_secondary[:1],
        perturbation_key,
        layer="raw_shifts",
        show=show_plots,
    )

    query_subset = query_adata[target_indices].copy()
    volcano_df = summarize_shifts(adata_shifts, query_subset, counts_layer=layer_key)
    volcano_fig = plot_volcano(
        volcano_df,
        highlight_genes=("Itgae", "Gzma", "Perturbation_4", "Klf2", "Ccl5"),
        show=show_plots,
    )

    non_control_mask = query_adata.obs[perturbation_key] != cfg.control_label
    violin_global = plot_violins(
        query_adata[non_control_mask].copy(),
        cfg.global_violin_genes,
        perturbation_key,
        show=show_plots,
    )
    matrixplot = plot_matrixplot(
        query_adata,
        cfg.matrixplot_genes,
        groupby=perturbation_key,
        show=show_plots,
    )

    comparison_results = compare_shift_outputs_effects(model, query_adata, perturbation_key, cfg.control_label)
    comparison_figures = plot_comparison_panels(comparison_results, show=show_plots)

    plot_artifacts = {
        "umaps": umap_plots,
        "violin_primary": violin_primary,
        "violin_secondary": violin_secondary,
        "violin_raw": violin_raw,
        "violin_global": violin_global,
        "matrixplot": matrixplot,
        "volcano": volcano_fig,
        "comparison": comparison_figures,
    }
    save_plot_collection(plot_artifacts, output_dir)

    return {
        "adata": query_adata,
        "model": model,
        "priors": priors,
        "effectiveness": effectiveness,
        "mini_adata": mini_adata,
        "adata_shifts": adata_shifts,
        "shift_outputs": shift_outputs,
        "volcano_df": volcano_df,
        "comparison_results": comparison_results,
        "plots": plot_artifacts,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    """Configure command-line arguments for quick experimentation."""

    parser = argparse.ArgumentParser(description="Run the RESOLVI spatial perturbation analysis.")
    parser.add_argument("adata_path", type=Path, help="Path to the reference AnnData (.h5ad) file.")
    parser.add_argument(
        "--target-gene",
        default="Ccl5",
        help="Target gene (excluding the 'sg' prefix) to analyse with shift outputs.",
    )
    parser.add_argument(
        "--control-label",
        default="Other cells",
        help="Label used for non-perturbed background cells in obs[guide_rnas].",
    )
    parser.add_argument(
        "--control-perturbation",
        default="sgCd19",
        help="Perturbation label treated as reference control during setup.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip rendering plots (useful for headless environments).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory under which timestamped figure folders will be created.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point when executed as a script."""

    args = parse_args()
    cfg = SpatialPerturbationConfig(
        adata_path=args.adata_path,
        target_gene=args.target_gene,
        control_label=args.control_label,
        control_perturbation=args.control_perturbation,
        output_root=args.output_root,
    )
    results = run_analysis(cfg, show_plots=not args.no_plots)
    print(f"Saved figures to: {results['output_dir']}")


if __name__ == "__main__":
    main()
