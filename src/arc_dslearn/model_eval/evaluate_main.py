"""Main evaluation coordinator that runs both DSL and simple prompt evaluations."""

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .evaluate_models_base import evaluate_base_model_simple
from .evaluate_models_finetuned import evaluate_finetuned_models
from .evaluate_rl_only import evaluate_rl_model
from .evaluate_sft_only import evaluate_sft_model


def create_comparison_plots(
    dsl_results: Dict[str, float], simple_results: Dict[str, float]
) -> None:
    """Create comparison plots between DSL and simple prompt evaluations."""
    # Combine results for comparison
    all_results = {}

    # Add DSL results
    for model, acc in dsl_results.items():
        all_results[f"{model}_dsl"] = acc

    # Add simple prompt results
    for model, acc in simple_results.items():
        all_results[model] = acc

    # Create comprehensive comparison plot
    acc_df = pd.Series(all_results).sort_values().to_frame("accuracy")

    # Color code the bars
    colors = []
    for idx in acc_df.index:
        if "simple" in idx:
            colors.append("lightgreen")  # Simple prompt
        elif "base" in idx:
            colors.append("lightcoral")  # Base model with DSL
        elif "sft" in idx:
            colors.append("lightblue")  # SFT model
        else:  # rl
            colors.append("gold")  # RL model

    ax = acc_df["accuracy"].plot(
        kind="barh", figsize=(10, 6), title="Model Comparison: DSL vs Simple Prompt", color=colors
    )
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1)

    # Add value labels on bars
    for i, v in enumerate(acc_df["accuracy"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("chart_results/model_comparison_complete.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_comprehensive_results(
    dsl_results: Dict[str, float], simple_results: Dict[str, float]
) -> None:
    """Print comprehensive comparison of all results."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 DSL Prompt Results:")
    print("-" * 30)
    for model, acc in sorted(dsl_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:>8}: {acc * 100:5.1f}%")

    print("\n📊 Simple Prompt Results:")
    print("-" * 30)
    for model, acc in sorted(simple_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:>12}: {acc * 100:5.1f}%")

    # Base model comparison
    base_dsl = dsl_results.get("base", 0.0)
    base_simple = simple_results.get("base_simple", 0.0)
    improvement = base_simple - base_dsl
    improvement_pct = (improvement / base_dsl * 100) if base_dsl > 0 else float("inf")

    print("\n🔍 BASE MODEL ANALYSIS:")
    print("-" * 30)
    print(f"DSL Prompt:     {base_dsl * 100:5.1f}%")
    print(f"Simple Prompt:  {base_simple * 100:5.1f}%")
    print(f"Improvement:    {improvement * 100:+5.1f}% absolute")
    if base_dsl > 0:
        print(f"Relative:       {improvement_pct:+5.1f}% relative")

    # Best model comparison
    best_dsl_model = max(dsl_results.items(), key=lambda x: x[1])
    best_simple_model = max(simple_results.items(), key=lambda x: x[1])

    print("\n🏆 BEST PERFORMING:")
    print("-" * 30)
    print(f"Best DSL:       {best_dsl_model[0]} ({best_dsl_model[1] * 100:.1f}%)")
    print(f"Best Simple:    {best_simple_model[0]} ({best_simple_model[1] * 100:.1f}%)")

    overall_best = best_dsl_model if best_dsl_model[1] > best_simple_model[1] else best_simple_model
    print(f"Overall Best:   {overall_best[0]} ({overall_best[1] * 100:.1f}%)")


def main():
    """Evaluate ARC DSL models with CLI options."""
    parser = argparse.ArgumentParser(description="Evaluate ARC DSL models")
    parser.add_argument(
        "--mode",
        choices=["all", "dsl", "simple", "sft-only", "rl-only"],
        default="all",
        help="Evaluation mode to run",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip generating comparison plots"
    )

    args = parser.parse_args()

    dsl_results = {}
    simple_results = {}

    if args.mode in ["all", "dsl"]:
        print("=" * 60)
        print("🔍 Running DSL Prompt Evaluation (all models)...")
        print("=" * 60)
        dsl_results = evaluate_finetuned_models()

    if args.mode in ["all", "simple"]:
        print("\n" + "=" * 60)
        print("🔍 Running Simple Prompt Evaluation (all models)...")
        print("=" * 60)
        simple_results = evaluate_base_model_simple()

    if args.mode == "sft-only":
        print("=" * 60)
        print("🔍 Running SFT Model Only Evaluation...")
        print("=" * 60)
        sft_results = evaluate_sft_model()
        print(f"\n✅ SFT evaluation complete! Accuracy: {sft_results['sft']:.3f}")

    if args.mode == "rl-only":
        print("=" * 60)
        print("🔍 Running RL Model Only Evaluation...")
        print("=" * 60)
        rl_results = evaluate_rl_model()
        print(f"\n✅ RL evaluation complete! Accuracy: {rl_results['rl']:.3f}")

    # Create comparison plots only for combined evaluations
    if args.mode == "all" and not args.skip_plots and dsl_results and simple_results:
        print("\n" + "=" * 60)
        print("📊 Creating Comparison Plots...")
        print("=" * 60)
        create_comparison_plots(dsl_results, simple_results)

        print("\n✅ Evaluation complete! Generated plots:")
        print(" • chart_results/model_comparison_complete.png")
        print(" • Individual model accuracy and error rate plots")

    if args.mode in ["dsl", "simple"] and dsl_results:
        print("\n✅ DSL evaluation complete!")
        for model, acc in dsl_results.items():
            print(f" • {model}: {acc:.3f}")

    if args.mode in ["dsl", "simple"] and simple_results:
        print("\n✅ Simple prompt evaluation complete!")
        for model, acc in simple_results.items():
            print(f" • {model}: {acc:.3f}")


if __name__ == "__main__":
    main()
