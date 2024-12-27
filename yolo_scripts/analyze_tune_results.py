import os
import matplotlib.pyplot as plt
import torch
from ray import tune
from ray.tune.examples.mnist_pytorch import train_mnist, ConvNet, get_data_loaders

def analyze_tune_results(experiment_path):
    """Analyze Ray Tune experiment results"""
    print(f"Loading results from {experiment_path}...")
    
    # 1. Restore results
    restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
    result_grid = restored_tuner.get_results()

    # 2. Check for errors
    if result_grid.errors:
        print("One of the trials failed!")
        for i, result in enumerate(result_grid):
            if result.error:
                print(f"Trial #{i} had an error:", result.error)
        return
    else:
        print("No errors!")

    # 3. Basic experiment stats
    num_results = len(result_grid)
    print(f"\nNumber of trials: {num_results}")

    # 4. Get trial metrics
    results_df = result_grid.get_dataframe()
    print("\nTraining times:")
    print("Shortest training time:", results_df["time_total_s"].min())
    print("Longest training time:", results_df["time_total_s"].max())

    # 5. Get best results
    best_result_df = result_grid.get_dataframe(
        filter_metric="mean_accuracy", 
        filter_mode="max"
    )
    print("\nBest results per trial:")
    print(best_result_df[["training_iteration", "mean_accuracy"]])

    # 6. Analyze best performing trial
    best_result = result_grid.get_best_result()
    print("\nBest trial config:", best_result.config)
    print("Best trial final accuracy:", best_result.metrics["mean_accuracy"])
    print("Best trial path:", best_result.path)

    # 7. Plot learning curves
    plt.figure(figsize=(10, 5))
    ax = None
    for result in result_grid:
        label = f"lr={result.config['lr']:.3f}, momentum={result.config['momentum']}"
        if ax is None:
            ax = result.metrics_dataframe.plot(
                "training_iteration", 
                "mean_accuracy", 
                label=label
            )
        else:
            result.metrics_dataframe.plot(
                "training_iteration", 
                "mean_accuracy", 
                ax=ax, 
                label=label
            )
    ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
    ax.set_ylabel("Mean Test Accuracy")
    plt.tight_layout()
    plt.show()

    # 8. Load best model and test inference
    model = ConvNet()
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model.pt"))
        )

    # Test inference on one sample
    _, test_loader = get_data_loaders()
    test_img = next(iter(test_loader))[0][0]
    predicted_class = torch.argmax(model(test_img)).item()

    # Plot test image
    plt.figure(figsize=(2, 2))
    test_img = test_img.numpy().reshape((1, 1, 28, 28))
    plt.imshow(test_img.reshape((28, 28)))
    plt.title(f"Predicted: {predicted_class}")
    plt.show()

    return result_grid, best_result

if __name__ == "__main__":
    # Specify your experiment path
    EXPERIMENT_PATH = "/home/pz286/ray_results/train_function_2024-12-26_23-39-11"
    
    # Run analysis
    result_grid, best_result = analyze_tune_results(EXPERIMENT_PATH)