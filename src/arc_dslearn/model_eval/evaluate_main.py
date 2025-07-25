"""Main evaluation coordinator that runs both DSL and simple prompt evaluations."""

from __future__ import annotations

import argparse
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .evaluate_models_base import evaluate_base_model_simple
from .evaluate_models_finetuned import evaluate_finetuned_models


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
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    plt.savefig("chart_results/model_comparison_all.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create base model specific comparison
    base_comparison = {
        "Base (DSL prompt)": dsl_results.get("base", 0.0),
        "Base (Simple prompt)": simple_results.get("base_simple", 0.0),
    }

    base_df = pd.Series(base_comparison).to_frame("accuracy")
    ax = base_df["accuracy"].plot(
        kind="bar",
        figsize=(8, 5),
        title="Base Model: DSL vs Simple Prompt Comparison",
        color=["lightcoral", "lightgreen"],
        rot=45,
    )
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, max(base_comparison.values()) * 1.2)

    # Add value labels on bars
    for i, v in enumerate(base_df["accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("chart_results/base_model_comparison.png", dpi=300, bbox_inches="tight")
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


def main() -> None:
    """Coordinate both evaluations."""
    parser = argparse.ArgumentParser(description="Run comprehensive model evaluation")
    parser.add_argument(
        "--skip-dsl", action="store_true", help="Skip DSL prompt evaluation (useful for testing)"
    )
    parser.add_argument(
        "--skip-simple",
        action="store_true",
        help="Skip simple prompt evaluation (useful for testing)",
    )
    args = parser.parse_args()

    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 50)

    dsl_results = {}
    simple_results = {}

    # Run DSL prompt evaluation (fine-tuned models + base)
    if not args.skip_dsl:
        print("\n📋 Phase 1: Evaluating with DSL prompts...")
        print("Models: base, sft, rl")
        try:
            dsl_results = evaluate_finetuned_models()
            print("✅ DSL evaluation completed")
        except Exception as e:
            print(f"❌ DSL evaluation failed: {e}")
            return
    else:
        print("⏭️  Skipping DSL evaluation")

    # Run simple prompt evaluation (base model only)
    if not args.skip_simple:
        print("\n📋 Phase 2: Evaluating base model with simple prompt...")
        try:
            simple_results = evaluate_base_model_simple()
            print("✅ Simple prompt evaluation completed")
        except Exception as e:
            print(f"❌ Simple prompt evaluation failed: {e}")
            return
    else:
        print("⏭️  Skipping simple prompt evaluation")

    # Generate comparison plots and analysis
    if dsl_results and simple_results:
        print("\n📊 Generating comparison plots...")
        create_comparison_plots(dsl_results, simple_results)
        print("✅ Comparison plots saved")

        print_comprehensive_results(dsl_results, simple_results)

        print("\n💾 Generated Files:")
        print("   • chart_results/model_comparison_all.png")
        print("   • chart_results/base_model_comparison.png")
        print("   • chart_results/model_accuracy_dsl.png")
        print("   • chart_results/model_accuracy_base_simple.png")
        print("   • chart_results/error_rate_*_dsl.png")
        print("   • chart_results/error_rate_base_simple.png")

    elif dsl_results:
        print("\n📊 Only DSL results available - check DSL-specific plots")
        print_comprehensive_results(dsl_results, {})

    elif simple_results:
        print("\n📊 Only simple prompt results available")
        print_comprehensive_results({}, simple_results)

    else:
        print("❌ No results to display")

    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
