"""Model Autopsy Callback for comprehensive diagnostics.

Automatically triggers when early stopping fires or training ends.
Performs diagnostic analysis including:
- Prediction distribution analysis
- Gradient flow analysis
- Activation statistics
- Captum attributions (optional)
- EEG-specific analyses (optional)

Example usage in config:
    callbacks:
      - class_path: cerebro.callbacks.ModelAutopsyCallback
        init_args:
          run_on_training_end: true
          diagnostics: ["predictions", "gradients", "activations"]
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import lightning as L
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class ModelAutopsyCallback(Callback):
    """Performs comprehensive model diagnostics.

    Triggers:
    - Automatically when early stopping fires (if run_on_early_stop=True)
    - Automatically at training end (if run_on_training_end=True)

    Args:
        run_on_training_end: Run autopsy when training completes (default: True)
        run_on_early_stop: Run autopsy when early stopping triggers (default: True)
        diagnostics: List of diagnostic modules to run. Options:
            - "predictions": Distribution, residuals, baseline comparisons
            - "gradients": Per-layer gradient flow
            - "activations": Dead neurons, layer statistics
            - "integrated_gradients": Captum IG (Phase 2)
            - "layer_gradcam": Captum GradCAM (Phase 2)
            - "channel_importance": EEG channel ablation (Phase 3)
            - "temporal_importance": EEG temporal ablation (Phase 3)
            - "failure_modes": Top-K worst predictions (Phase 4)
        output_dir: Directory for diagnostic outputs (default: trainer.log_dir/autopsy)
        save_plots: Save diagnostic plots to disk (default: True)
        log_to_wandb: Upload plots to wandb (default: True)
        generate_report: Generate markdown autopsy report (default: True)
        num_samples: Number of samples to analyze (default: None = full val set)
        num_ablation_trials: For ablation studies, number of trials (default: 10)
        ig_n_steps: Integrated Gradients steps (default: 50)
        ig_baseline: IG baseline type: "zero", "mean", "random" (default: "zero")
        shap_n_samples: GradientSHAP baseline samples (default: 10)
        target_layers: Layer names for GradCAM (default: ["auto"] = detect conv layers)
        top_k_failures: Number of worst predictions to analyze in detail (default: 100)
        plot_dpi: DPI for saved plots (default: 150)
        plot_format: Plot format: "png", "pdf", "svg" (default: "png")
    """

    def __init__(
        self,
        run_on_training_end: bool = True,
        run_on_early_stop: bool = True,
        diagnostics: List[str] = None,
        output_dir: Optional[Path] = None,
        save_plots: bool = True,
        log_to_wandb: bool = True,
        generate_report: bool = True,
        num_samples: Optional[int] = None,
        num_ablation_trials: int = 10,
        ig_n_steps: int = 50,
        ig_baseline: str = "zero",
        shap_n_samples: int = 10,
        target_layers: List[str] = None,
        top_k_failures: int = 100,
        plot_dpi: int = 150,
        plot_format: str = "png",
    ):
        super().__init__()

        # Default diagnostics (Tier 1 only)
        if diagnostics is None:
            diagnostics = ["predictions", "gradients", "activations"]

        self.run_on_training_end = run_on_training_end
        self.run_on_early_stop = run_on_early_stop
        self.diagnostics = diagnostics
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_plots = save_plots
        self.log_to_wandb = log_to_wandb
        self.generate_report = generate_report
        self.num_samples = num_samples
        self.num_ablation_trials = num_ablation_trials
        self.ig_n_steps = ig_n_steps
        self.ig_baseline = ig_baseline
        self.shap_n_samples = shap_n_samples
        self.target_layers = target_layers or ["auto"]
        self.top_k_failures = top_k_failures
        self.plot_dpi = plot_dpi
        self.plot_format = plot_format

        # State tracking
        self.autopsy_triggered = False
        self.early_stop_detected = False

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Check if early stopping will trigger.

        Early stopping sets trainer.should_stop=True during validation_end.
        We detect this and trigger autopsy before training actually stops.
        """
        if trainer.should_stop and not self.early_stop_detected:
            self.early_stop_detected = True
            if self.run_on_early_stop:
                logger.info(
                    "[bold yellow]⚠️  Early stopping detected. Running model autopsy...[/bold yellow]"
                )
                self._run_autopsy(trainer, pl_module, trigger="early_stop")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Run autopsy at training end if not already run."""
        if self.run_on_training_end and not self.autopsy_triggered:
            logger.info(
                "[bold green]🔬 Training complete. Running model autopsy...[/bold green]"
            )
            self._run_autopsy(trainer, pl_module, trigger="training_end")

    def _run_autopsy(
        self, trainer: L.Trainer, pl_module: L.LightningModule, trigger: str
    ) -> None:
        """Execute comprehensive diagnostics.

        Args:
            trainer: Lightning Trainer
            pl_module: Lightning Module
            trigger: What triggered the autopsy ("early_stop" or "training_end")
        """
        if self.autopsy_triggered:
            return  # Only run once

        self.autopsy_triggered = True

        # 1. Setup output directory
        if self.output_dir is None:
            output_dir = Path(trainer.log_dir) / "autopsy"
        else:
            output_dir = self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Autopsy output directory: {output_dir}")

        # 2. Load best checkpoint (if available)
        best_ckpt = None
        if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback:
            best_ckpt = trainer.checkpoint_callback.best_model_path
            if best_ckpt and Path(best_ckpt).exists():
                logger.info(f"📦 Loading best checkpoint: {best_ckpt}")
                try:
                    pl_module = pl_module.__class__.load_from_checkpoint(best_ckpt)
                except Exception as e:
                    logger.warning(
                        f"⚠️  Failed to load checkpoint: {e}. Using current model state."
                    )
            else:
                logger.info("ℹ️  No checkpoint found. Using current model state.")

        # Get device from pl_module (works in all Lightning versions)
        device = pl_module.device
        pl_module = pl_module.to(device)
        pl_module.eval()

        # 3. Get validation dataloader
        val_loader = trainer.datamodule.val_dataloader()
        logger.info(f"📊 Analyzing validation set ({len(val_loader.dataset)} samples)")

        # 4. Run diagnostics (will be implemented in subsequent tasks)
        results = {}

        logger.info("🔍 Running diagnostic modules...")

        # Import diagnostic modules (will be created in next tasks)
        try:
            from cerebro.diagnostics.activations import analyze_activations
            from cerebro.diagnostics.gradients import analyze_gradient_flow
            from cerebro.diagnostics.predictions import (
                analyze_predictions,
                compute_baseline_scores,
            )
            from cerebro.diagnostics.visualizations import (
                plot_activation_stats,
                plot_gradient_flow,
                plot_prediction_distribution,
            )

            # Prediction analysis
            if "predictions" in self.diagnostics:
                logger.info("  ├─ Analyzing predictions...")
                results["predictions"] = analyze_predictions(
                    pl_module.model, val_loader, device, self.num_samples
                )

                logger.info("  ├─ Computing baseline comparisons...")
                results["baselines"] = compute_baseline_scores(
                    results["predictions"]["predictions"],
                    results["predictions"]["targets"],
                )

            # Gradient flow analysis
            if "gradients" in self.diagnostics:
                logger.info("  ├─ Analyzing gradient flow...")
                batch = next(iter(val_loader))
                results["gradients"] = analyze_gradient_flow(
                    pl_module.model, batch, device
                )

            # Activation analysis
            if "activations" in self.diagnostics:
                logger.info("  ├─ Analyzing activations...")
                batch = next(iter(val_loader))
                results["activations"] = analyze_activations(
                    pl_module.model, batch, device
                )

            # Integrated Gradients (Captum)
            if "integrated_gradients" in self.diagnostics:
                logger.info("  ├─ Computing Integrated Gradients (Captum)...")
                try:
                    from cerebro.diagnostics.captum_attributions import (
                        compute_integrated_gradients,
                        interpret_spatial_pattern,
                        interpret_temporal_pattern,
                    )

                    results["integrated_gradients"] = compute_integrated_gradients(
                        pl_module.model,
                        val_loader,
                        device,
                        num_samples=self.num_samples or 100,
                        n_steps=self.ig_n_steps,
                        baseline_type=self.ig_baseline,
                    )

                    # Add interpretations
                    ig_res = results["integrated_gradients"]
                    results["ig_temporal_interp"] = interpret_temporal_pattern(
                        ig_res["peak_time_sec"], ig_res["temporal_profile"]
                    )
                    results["ig_spatial_interp"] = interpret_spatial_pattern(
                        ig_res["spatial_profile"], ig_res["peak_channel_idx"]
                    )

                except ImportError as e:
                    logger.warning(f"⚠️  Captum not available: {e}")

            # Layer GradCAM (Captum)
            if "layer_gradcam" in self.diagnostics:
                logger.info("  └─ Computing Layer GradCAM (Captum)...")
                try:
                    from cerebro.diagnostics.captum_layers import (
                        compute_layer_gradcam,
                        interpret_layer_hierarchy,
                        summarize_layer_patterns,
                    )

                    # Auto-detect layers if needed
                    target_layers = self.target_layers
                    if target_layers == ["auto"]:
                        from cerebro.diagnostics.captum_layers import detect_conv_layers

                        target_layers = detect_conv_layers(pl_module.model)
                        logger.info(
                            f"    Auto-detected {len(target_layers)} conv layers"
                        )

                    results["layer_gradcam"] = compute_layer_gradcam(
                        pl_module.model,
                        val_loader,
                        device,
                        target_layers=target_layers,
                        num_samples=self.num_samples or 100,
                    )

                    # Add interpretations
                    gc_res = results["layer_gradcam"]
                    results["layer_hierarchy_interp"] = interpret_layer_hierarchy(
                        gc_res["layer_importance"], gc_res["layer_shapes"]
                    )
                    results["layer_patterns"] = summarize_layer_patterns(
                        gc_res["layer_attributions"]
                    )

                except ImportError as e:
                    logger.warning(f"⚠️  Captum not available: {e}")

            # Channel ablation
            if "channel_importance" in self.diagnostics:
                logger.info("  ├─ Running channel ablation study...")
                from cerebro.diagnostics.ablation import (
                    ablate_channels,
                    interpret_channel_importance,
                )

                results["channel_ablation"] = ablate_channels(
                    pl_module.model,
                    val_loader,
                    device,
                    num_samples=self.num_samples or 100,
                    num_trials=self.num_ablation_trials,
                    ablation_strategy="zero",
                )

                # Add interpretation
                ch_abl = results["channel_ablation"]
                results["channel_importance_interp"] = interpret_channel_importance(
                    ch_abl["channel_importance"],
                    ch_abl["most_important_channels"],
                    ch_abl["least_important_channels"],
                )

            # Temporal ablation
            if "temporal_importance" in self.diagnostics:
                logger.info("  └─ Running temporal ablation study...")
                from cerebro.diagnostics.ablation import (
                    ablate_temporal_windows,
                    interpret_temporal_importance,
                )

                results["temporal_ablation"] = ablate_temporal_windows(
                    pl_module.model,
                    val_loader,
                    device,
                    num_samples=self.num_samples or 100,
                    num_trials=self.num_ablation_trials,
                    window_size=20,  # 200ms windows at 100Hz
                    ablation_strategy="zero",
                )

                # Add interpretation
                tmp_abl = results["temporal_ablation"]
                results["temporal_importance_interp"] = interpret_temporal_importance(
                    tmp_abl["window_importance"],
                    tmp_abl["window_centers_sec"],
                    tmp_abl["most_important_time_sec"],
                )

            # Failure mode analysis
            if "failure_modes" in self.diagnostics:
                logger.info("  └─ Analyzing failure modes...")
                from cerebro.diagnostics.failure_modes import analyze_failure_modes

                results["failure_modes"] = analyze_failure_modes(
                    pl_module.model,
                    val_loader,
                    device,
                    top_k=self.top_k_failures,
                    metadata_keys=["subject", "correct", "rt_from_stimulus"],
                    output_dir=output_dir if self.save_plots else None,
                )

            # 5. Generate plots
            plot_paths = []
            if self.save_plots:
                logger.info("📈 Generating diagnostic plots...")

                if "predictions" in results:
                    plot_path = (
                        output_dir / f"prediction_distribution.{self.plot_format}"
                    )
                    plot_prediction_distribution(results["predictions"], plot_path)
                    plot_paths.append(plot_path)
                    logger.info(f"  ├─ {plot_path.name}")

                if "gradients" in results:
                    plot_path = output_dir / f"gradient_flow.{self.plot_format}"
                    plot_gradient_flow(results["gradients"], plot_path)
                    plot_paths.append(plot_path)
                    logger.info(f"  ├─ {plot_path.name}")

                if "activations" in results:
                    plot_path = output_dir / f"activation_stats.{self.plot_format}"
                    plot_activation_stats(results["activations"], plot_path)
                    plot_paths.append(plot_path)
                    logger.info(f"  ├─ {plot_path.name}")

                if "integrated_gradients" in results:
                    from cerebro.diagnostics.visualizations import (
                        plot_integrated_gradients,
                    )

                    plot_path = output_dir / f"integrated_gradients.{self.plot_format}"
                    plot_integrated_gradients(
                        results["integrated_gradients"], plot_path
                    )
                    plot_paths.append(plot_path)
                    logger.info(f"  ├─ {plot_path.name}")

                if "layer_gradcam" in results:
                    from cerebro.diagnostics.visualizations import (
                        plot_layer_gradcam,
                        plot_layer_temporal_profiles,
                    )

                    plot_path = output_dir / f"layer_gradcam.{self.plot_format}"
                    plot_layer_gradcam(results["layer_gradcam"], plot_path)
                    plot_paths.append(plot_path)
                    logger.info(f"  ├─ {plot_path.name}")

                    plot_path = (
                        output_dir / f"layer_temporal_profiles.{self.plot_format}"
                    )
                    plot_layer_temporal_profiles(results["layer_patterns"], plot_path)
                    plot_paths.append(plot_path)
                    logger.info(f"  ├─ {plot_path.name}")

                if "channel_ablation" in results:
                    from cerebro.diagnostics.visualizations import plot_channel_ablation

                    plot_path = output_dir / f"channel_ablation.{self.plot_format}"
                    plot_channel_ablation(results["channel_ablation"], plot_path)
                    plot_paths.append(plot_path)
                    logger.info(f"  ├─ {plot_path.name}")

                if "temporal_ablation" in results:
                    from cerebro.diagnostics.visualizations import (
                        plot_temporal_ablation,
                    )

                    plot_path = output_dir / f"temporal_ablation.{self.plot_format}"
                    plot_temporal_ablation(results["temporal_ablation"], plot_path)
                    plot_paths.append(plot_path)
                    logger.info(f"  └─ {plot_path.name}")

            # 6. Log to wandb
            if (
                self.log_to_wandb
                and trainer.logger
                and hasattr(trainer.logger, "experiment")
            ):
                logger.info("☁️  Uploading plots to wandb...")
                try:
                    import wandb

                    for plot_path in plot_paths:
                        trainer.logger.experiment.log(
                            {f"autopsy/{plot_path.stem}": wandb.Image(str(plot_path))}
                        )
                    logger.info("  ✓ Uploaded to wandb")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to upload to wandb: {e}")

            # 7. Generate report
            if self.generate_report:
                logger.info("📋 Generating autopsy report...")
                report = self._generate_report(results, trigger, best_ckpt)
                report_path = output_dir / "autopsy_report.md"
                report_path.write_text(report)
                logger.info(f"  ✓ Report saved: {report_path}")

            logger.info("[bold green]✅ Model autopsy complete![/bold green]")

        except ImportError as e:
            logger.error(
                f"[bold red]❌ Failed to import diagnostic modules: {e}[/bold red]"
            )
            logger.error("   Please ensure all diagnostic modules are implemented.")

    def _generate_report(
        self, diagnostics: dict, trigger: str, checkpoint_path: Optional[str]
    ) -> str:
        """Generate markdown autopsy report.

        Args:
            diagnostics: Dict with diagnostic results
            trigger: What triggered the autopsy
            checkpoint_path: Path to best checkpoint

        Returns:
            Markdown report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# 🔬 MODEL AUTOPSY REPORT

**Trigger**: {trigger}
**Checkpoint**: {checkpoint_path or 'Current model state (no checkpoint)'}
**Timestamp**: {timestamp}

---

"""

        # Prediction Analysis
        if "predictions" in diagnostics and "baselines" in diagnostics:
            pred = diagnostics["predictions"]
            baseline = diagnostics["baselines"]

            report += f"""## 📊 PREDICTION ANALYSIS

- **Model NRMSE**: {pred['nrmse']:.4f}
- **Naive Mean NRMSE**: {baseline['naive_mean_nrmse']:.4f} (baseline)
- **Improvement**: {baseline['improvement_over_mean']*100:+.2f}%
- **Prediction Std**: {pred['pred_std']:.4f}
- **Target Std**: {pred['target_std']:.4f}
- **Variance Ratio**: {pred['variance_ratio']:.2f}

"""

            # Diagnosis
            if pred["variance_ratio"] < 0.5:
                report += "🚨 **WARNING**: Predictions collapsed to narrow range! Model is not capturing target variance.\n\n"

            if baseline["improvement_over_mean"] < 0:
                report += "🚨 **WARNING**: Model performs worse than naive baseline (always predicting mean)!\n\n"
            elif baseline["improvement_over_mean"] < 0.01:
                report += "⚠️  **WARNING**: Model barely better than naive baseline (<1% improvement).\n\n"

        # Gradient Flow
        if "gradients" in diagnostics:
            grad = diagnostics["gradients"]
            import numpy as np

            avg_grad_to_param = np.mean(grad["grad_to_param_ratio"])

            report += f"""## 🌊 GRADIENT FLOW

- **Layers with gradients**: {len(grad['layer_names']) - len(grad['dead_layers'])}/{len(grad['layer_names'])}
- **Dead layers**: {len(grad['dead_layers'])}
- **Avg grad/param ratio**: {avg_grad_to_param:.6f}

"""

            if len(grad["dead_layers"]) > 0:
                report += (
                    f"⚠️  **Dead layers detected**: {', '.join(grad['dead_layers'])}\n\n"
                )

            if avg_grad_to_param < 1e-5:
                report += "⚠️  **WARNING**: Very small gradients detected. Learning rate may be too low.\n\n"
            elif avg_grad_to_param > 0.1:
                report += "⚠️  **WARNING**: Very large gradients detected. Learning rate may be too high.\n\n"

        # Activation Health
        if "activations" in diagnostics:
            act = diagnostics["activations"]
            import numpy as np

            avg_dead_pct = np.mean(act["dead_neuron_pcts"])

            report += f"""## ⚡ ACTIVATION HEALTH

- **Average dead neurons**: {avg_dead_pct:.1f}%
- **Layers analyzed**: {len(act['layer_names'])}

"""

            if avg_dead_pct > 10:
                report += f"⚠️  **WARNING**: {avg_dead_pct:.1f}% dead neurons on average. Model may be over-regularized or poorly initialized.\n\n"

        # Integrated Gradients Analysis (Captum)
        if "integrated_gradients" in diagnostics:
            ig = diagnostics["integrated_gradients"]

            report += f"""## 🎯 INTEGRATED GRADIENTS (Captum)

- **Peak Attribution Time**: {ig['peak_time_sec']:.2f}s post-stimulus
- **Peak Attribution Channel**: Channel {ig['peak_channel_idx']}

### Temporal Pattern
{diagnostics.get('ig_temporal_interp', 'No interpretation available')}

### Spatial Pattern
{diagnostics.get('ig_spatial_interp', 'No interpretation available')}

"""

        # Layer GradCAM Analysis (Captum)
        if "layer_gradcam" in diagnostics:
            gc = diagnostics["layer_gradcam"]

            report += f"""## 🔍 LAYER GRADCAM (Captum)

- **Most Important Layer**: {gc.get('most_important_layer', 'N/A')}
- **Layers Analyzed**: {len(gc['target_layers'])}

### Layer Hierarchy
{diagnostics.get('layer_hierarchy_interp', 'No interpretation available')}

"""

        # Channel Ablation
        if "channel_ablation" in diagnostics:
            ch_abl = diagnostics["channel_ablation"]

            report += f"""## 🧠 CHANNEL ABLATION STUDY

- **Baseline NRMSE**: {ch_abl['baseline_nrmse']:.4f}
- **Top 3 Most Important Channels**: {ch_abl['most_important_channels'][:3]}

### Channel Importance
{diagnostics.get('channel_importance_interp', 'No interpretation available')}

"""

        # Temporal Ablation
        if "temporal_ablation" in diagnostics:
            tmp_abl = diagnostics["temporal_ablation"]

            report += f"""## ⏱️  TEMPORAL ABLATION STUDY

- **Baseline NRMSE**: {tmp_abl['baseline_nrmse']:.4f}
- **Most Critical Window**: {tmp_abl['most_important_time_sec']:.2f}s post-stimulus

### Temporal Importance
{diagnostics.get('temporal_importance_interp', 'No interpretation available')}

"""

        # Recommendations
        report += """## 💡 RECOMMENDATIONS

Based on the diagnostic results:

"""

        recommendations = []

        if (
            "predictions" in diagnostics
            and diagnostics["predictions"]["variance_ratio"] < 0.5
        ):
            recommendations.append(
                "1. **Increase learning rate** - predictions collapsed to narrow range"
            )
            recommendations.append(
                "2. **Reduce weight decay** - model may be over-regularized"
            )
            recommendations.append("3. **Train longer** - may not have converged yet")

        if (
            "gradients" in diagnostics
            and len(diagnostics["gradients"]["dead_layers"]) > 0
        ):
            recommendations.append(
                f"4. **Investigate dead layers**: {', '.join(diagnostics['gradients']['dead_layers'])}"
            )

        if "predictions" in diagnostics and "baselines" in diagnostics:
            if diagnostics["baselines"]["improvement_over_mean"] < 0.05:
                recommendations.append(
                    "5. **Consider architectural changes** - model barely learning"
                )
                recommendations.append(
                    "6. **Try contrastive pretraining** - may need better initialization"
                )

        # Captum-based recommendations
        if "integrated_gradients" in diagnostics:
            ig = diagnostics["integrated_gradients"]
            # Check if model attends to wrong temporal window
            if ig["peak_time_sec"] < 0.8 or ig["peak_time_sec"] > 1.3:
                recommendations.append(
                    f"7. **Architecture may be inadequate** - model attends to wrong temporal window ({ig['peak_time_sec']:.2f}s, expected 0.8-1.3s)"
                )
            # Check if temporal profile is uniform
            import numpy as np

            profile_std = np.std(ig["temporal_profile"])
            profile_mean = np.mean(ig["temporal_profile"])
            if profile_mean > 0 and profile_std / profile_mean < 0.2:
                recommendations.append(
                    "8. **Model not learning temporal structure** - attribution profile is too uniform"
                )

        if recommendations:
            report += "\n".join(recommendations)
        else:
            report += "No critical issues detected. Model training appears healthy.\n"

        report += "\n\n---\n\n"
        report += f"*Generated by ModelAutopsyCallback at {timestamp}*\n"

        return report
