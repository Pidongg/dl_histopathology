import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ray.tune import ExperimentAnalysis
import numpy as np

# Set the experiment path
experiment_path = "/home/pz286/ray_results/train_function_2024-12-26_23-39-11"

# Load the experiment analysis
analysis = ExperimentAnalysis(experiment_path)

# Get all trials dataframe
df = analysis.dataframe()

# Print best trial configuration and metrics
best_trial = analysis.get_best_trial(metric="fitness", mode="min")
print("\nBest Trial Configuration:")
print("========================")
for key, value in best_trial.config.items():
    if key.startswith('space/'):  # Only print hyperparameters
        print(f"{key.replace('space/', '')}: {value}")

print("\nBest Trial Metrics:")
print("=================")
metrics = best_trial.last_result
for key in ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']:
    if key in metrics:
        print(f"{key}: {metrics[key]:.4f}")

# Create visualizations directory
os.makedirs("tune_analysis", exist_ok=True)

# Plot hyperparameter distributions
hyperparams = ['degrees', 'shear', 'translate', 'scale', 'flipud', 'fliplr']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, param in enumerate(hyperparams):
    param_key = f'space/{param}'
    if param_key in df.columns:
        sns.histplot(data=df, x=param_key, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {param}')
        axes[idx].set_xlabel(param)

plt.tight_layout()
plt.savefig('tune_analysis/hyperparameter_distributions.png')
plt.close()

# Plot correlation matrix of hyperparameters and metrics
metrics_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
                'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']
hyperparam_cols = [f'space/{param}' for param in hyperparams]

correlation_cols = hyperparam_cols + metrics_cols
correlation_df = df[correlation_cols].copy()

# Rename columns for better visualization
correlation_df.columns = [col.replace('space/', '').replace('metrics/', '').replace('val/', '') 
                        for col in correlation_df.columns]

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Hyperparameters and Metrics')
plt.tight_layout()
plt.savefig('tune_analysis/correlation_matrix.png')
plt.close()

# Plot parallel coordinates plot for top trials
top_n = 10  # Number of top trials to visualize
metrics_to_optimize = 'metrics/mAP50(B)'  # Change this to your target metric

# Get the top N trials based on the metric
top_trials = df.nlargest(top_n, metrics_to_optimize)

# Prepare data for parallel coordinates plot
parallel_plot_data = top_trials[hyperparam_cols + [metrics_to_optimize]].copy()
parallel_plot_data.columns = [col.replace('space/', '') for col in parallel_plot_data.columns]

# Normalize the data for better visualization
normalized_data = (parallel_plot_data - parallel_plot_data.min()) / (parallel_plot_data.max() - parallel_plot_data.min())

plt.figure(figsize=(15, 8))
pd.plotting.parallel_coordinates(normalized_data, metrics_to_optimize.replace('metrics/', ''))
plt.title(f'Parallel Coordinates Plot of Top {top_n} Trials')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('tune_analysis/parallel_coordinates.png')
plt.close()

# Save summary statistics to a text file
with open('tune_analysis/summary_statistics.txt', 'w') as f:
    f.write("Summary Statistics\n")
    f.write("==================\n\n")
    
    f.write("Hyperparameter Ranges:\n")
    for param in hyperparams:
        param_key = f'space/{param}'
        if param_key in df.columns:
            f.write(f"{param}:\n")
            f.write(f"  Min: {df[param_key].min():.4f}\n")
            f.write(f"  Max: {df[param_key].max():.4f}\n")
            f.write(f"  Mean: {df[param_key].mean():.4f}\n")
            f.write(f"  Std: {df[param_key].std():.4f}\n\n")
    
    f.write("\nMetrics Statistics:\n")
    for metric in metrics_cols:
        if metric in df.columns:
            f.write(f"{metric}:\n")
            f.write(f"  Min: {df[metric].min():.4f}\n")
            f.write(f"  Max: {df[metric].max():.4f}\n")
            f.write(f"  Mean: {df[metric].mean():.4f}\n")
            f.write(f"  Std: {df[metric].std():.4f}\n\n")

print("\nAnalysis complete! Results saved in the 'tune_analysis' directory.")
print("Generated files:")
print("- hyperparameter_distributions.png")
print("- correlation_matrix.png")
print("- parallel_coordinates.png")
print("- summary_statistics.txt")