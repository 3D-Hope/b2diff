"""
Plot metrics vs mean reward for comparing two methods.

Usage:
    python plot_metrics_comparison.py

Make sure to update the file paths in the data_config dictionary below
to match your local file locations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# CONFIGURATION - Update these paths to match your file locations
# ============================================================================
data_config = {
    "tv_bed_top_of_universal": {
        "metrics": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/tv_bed_top_of_universal/metrics_table.csv",
        "rewards": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/reward_csvs/tv_bed_top_of_universal.csv"
    },
    "tv_bed_universal": {
        "metrics": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/3d_layout_generation/MiDiffusion/output/full_predicted_results/tv_bed_universal/metrics_table.csv",
        "rewards": "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_b2diff/b2diff/reward_csvs/tv_bed_universal.csv"
    }
}

# Output directory for plots
OUTPUT_DIR = "plots_output"

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_method_data(metrics_path, reward_path, method_name):
    """Load and merge metrics and rewards data."""
    print(f"\nLoading data for {method_name}...")
    
    # Load metrics
    metrics_df = pd.read_csv(metrics_path)
    print(f"  Metrics shape: {metrics_df.shape}")
    print(f"  Metrics columns: {list(metrics_df.columns)}")
    
    # Load rewards
    reward_df = pd.read_csv(reward_path)
    print(f"  Rewards shape: {reward_df.shape}")
    print(f"  Rewards columns: {list(reward_df.columns)}")
    
    # Find the reward column (second column, contains actual reward values)
    reward_col = reward_df.columns[1]
    print(f"  Using reward column: {reward_col}")
    
    # Create a simplified reward dataframe with stage index
    reward_simple = pd.DataFrame({
        'stage': range(len(reward_df)),
        'mean_reward': reward_df[reward_col].values
    })
    
    # Ensure stage column exists in metrics_df
    if 'stage' not in metrics_df.columns:
        metrics_df['stage'] = range(len(metrics_df))
    
    # Merge on stage
    merged_df = pd.merge(metrics_df, reward_simple, on='stage', how='inner')
    print(f"  Merged shape: {merged_df.shape}")
    
    return merged_df


def create_comparison_plots(df1, df2, method1_name, method2_name, output_dir):
    """Create comparison plots for all metrics."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot (excluding stage and reward columns)
    # Based on the actual columns in the CSV
    exclude_cols = ['stage', 'mean_reward', 'mean_tv_bed_reward', 'Step']
    metrics_to_plot = [col for col in df1.columns if col not in exclude_cols]
    
    print(f"\nMetrics to plot ({len(metrics_to_plot)}): {metrics_to_plot}")
    
    # Expected metrics based on user's specification:
    # col_obj, col_scene, avg_num_obj, kl_div, out_of_bound_rate, 
    # walkable_average_rate, accessable_rate, box_wall_rate, 
    # object_category_entropy_synthesized, pairwise_scene_embedding_distance_synthesized,
    # furniture_weighted_variance_synthesized
    
    # ========================================================================
    # 1. Create ONE comprehensive plot with all metrics (4x3 grid)
    # ========================================================================
    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    # Create larger figure to accommodate all plots clearly
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Define colors for the two methods
    color1 = '#2E86AB'  # Blue for method 1
    color2 = '#E63946'  # Red for method 2
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Plot method 1
        ax.plot(df1['mean_reward'], df1[metric], 'o-', 
                label=method1_name.replace('_', ' ').replace('tv bed', 'TV-Bed'), 
                linewidth=2.5, markersize=9, alpha=0.8, color=color1)
        
        # Plot method 2
        ax.plot(df2['mean_reward'], df2[metric], 's-', 
                label=method2_name.replace('_', ' ').replace('tv bed', 'TV-Bed'), 
                linewidth=2.5, markersize=9, alpha=0.8, color=color2)
        
        # Format metric name for better readability
        metric_display = metric.replace('_', ' ').title()
        if 'synthesized' in metric.lower():
            metric_display = metric_display.replace(' Synthesized', '')
        
        ax.set_xlabel('Mean Reward', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_display, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_display}', fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=9)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle('Metrics vs Mean Reward Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    combined_plot_path = os.path.join(output_dir, 'all_metrics_vs_reward_comparison.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Combined plot saved: {combined_plot_path}")
    plt.close()
    
    return metrics_to_plot


def print_summary_statistics(df1, df2, method1_name, method2_name, metrics_to_plot):
    """Print summary statistics for both methods."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for method_name, df in [(method1_name, df1), (method2_name, df2)]:
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        print(f"  Reward range: [{df['mean_reward'].min():.4f}, {df['mean_reward'].max():.4f}]")
        print(f"  Number of stages: {len(df)}")
        print(f"\n  Metrics:")
        for metric in metrics_to_plot:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            print(f"    {metric:25s}: mean={mean_val:8.4f}, std={std_val:8.4f}, range=[{min_val:.4f}, {max_val:.4f}]")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("METRICS VS REWARD COMPARISON")
    print("="*80)
    print("\nExpected metrics:")
    print("  - col_obj, col_scene, avg_num_obj, kl_div")
    print("  - out_of_bound_rate, walkable_average_rate, accessable_rate, box_wall_rate")
    print("  - object_category_entropy_synthesized")
    print("  - pairwise_scene_embedding_distance_synthesized")
    print("  - furniture_weighted_variance_synthesized")
    print()
    
    # Extract method names
    method1_name = list(data_config.keys())[0]
    method2_name = list(data_config.keys())[1]
    
    # Load data for both methods
    df1 = load_method_data(
        data_config[method1_name]["metrics"],
        data_config[method1_name]["rewards"],
        method1_name
    )
    
    df2 = load_method_data(
        data_config[method2_name]["metrics"],
        data_config[method2_name]["rewards"],
        method2_name
    )
    
    # Create plots (only one comprehensive plot now)
    metrics_to_plot = create_comparison_plots(
        df1, df2, method1_name, method2_name, OUTPUT_DIR
    )
    
    # Print summary statistics
    print_summary_statistics(df1, df2, method1_name, method2_name, metrics_to_plot)
    
    print("\n" + "="*80)
    print(f"✓ PLOT SAVED TO: {os.path.abspath(OUTPUT_DIR)}/all_metrics_vs_reward_comparison.png")
    print("="*80)