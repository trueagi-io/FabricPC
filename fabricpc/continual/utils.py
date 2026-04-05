"""
Utility functions for Continual Learning.

Provides plotting, serialization, and analysis helpers for
continual learning experiments.
"""

from typing import Dict, List, Tuple, Optional, Any, Sequence
from pathlib import Path
import json
import numpy as np

from fabricpc.continual.trainer import TaskRunSummary


def summaries_to_dataframe(summaries: Sequence[TaskRunSummary]):
    """
    Convert task summaries to a pandas DataFrame.

    Args:
        summaries: List of TaskRunSummary objects

    Returns:
        DataFrame with one row per task
    """
    try:
        import pandas as pd

        rows = [s.to_dict() for s in summaries]
        return pd.DataFrame(rows)
    except ImportError:
        return [s.to_dict() for s in summaries]


def save_summaries_json(
    summaries: Sequence[TaskRunSummary],
    path: str,
):
    """Save task summaries to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [s.to_dict() for s in summaries]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_summaries_json(path: str) -> List[TaskRunSummary]:
    """Load task summaries from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    summaries = []
    for d in data:
        summaries.append(
            TaskRunSummary(
                task_id=d["task_id"],
                classes=tuple(d["classes"]),
                train_accuracy=d["train_accuracy"],
                test_accuracy=d["test_accuracy"],
                train_loss=d["train_loss"],
                test_loss=d["test_loss"],
                train_energy=d.get("train_energy", 0.0),
                test_energy=d.get("test_energy", 0.0),
                epochs_trained=d["epochs_trained"],
                training_time=d["training_time"],
                support_cols=tuple(d["support_cols"]),
                epoch_accuracies=d.get("epoch_accuracies", []),
                epoch_losses=d.get("epoch_losses", []),
            )
        )
    return summaries


def save_accuracy_matrix(
    matrix: np.ndarray,
    path: str,
):
    """Save accuracy matrix to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd

        df = pd.DataFrame(
            matrix,
            index=[f"after_task_{i}" for i in range(matrix.shape[0])],
            columns=[f"task_{j}" for j in range(matrix.shape[1])],
        )
        df.to_csv(path)
    except ImportError:
        np.savetxt(path, matrix, delimiter=",", fmt="%.4f")


def compute_forgetting(accuracy_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-task forgetting from accuracy matrix.

    Args:
        accuracy_matrix: Matrix where [i,j] is accuracy on task j
                        after training up to task i

    Returns:
        Forgetting per task (drop from peak to final accuracy)
    """
    num_tasks = accuracy_matrix.shape[1]
    forgetting = np.zeros(num_tasks)

    for task in range(num_tasks):
        # Get accuracies on this task over time
        task_accs = accuracy_matrix[:, task]
        # Only consider after task was first learned
        if task < len(task_accs):
            peak_acc = np.max(task_accs[task:])
            final_acc = task_accs[-1] if len(task_accs) > 0 else 0
            forgetting[task] = max(0, peak_acc - final_acc)

    return forgetting


def compute_backward_transfer(accuracy_matrix: np.ndarray) -> float:
    """
    Compute backward transfer (BWT).

    BWT measures the average influence of learning new tasks
    on the performance of previously learned tasks.

    Args:
        accuracy_matrix: Accuracy matrix

    Returns:
        Backward transfer score (negative = forgetting)
    """
    num_tasks = accuracy_matrix.shape[1]
    if num_tasks < 2:
        return 0.0

    bwt_sum = 0.0
    count = 0

    for task in range(num_tasks - 1):
        # Accuracy right after training task vs final accuracy
        if task < accuracy_matrix.shape[0]:
            initial = accuracy_matrix[task, task]
            final = accuracy_matrix[-1, task]
            bwt_sum += final - initial
            count += 1

    return bwt_sum / count if count > 0 else 0.0


def compute_forward_transfer(
    accuracy_matrix: np.ndarray,
    baseline_accuracy: Optional[float] = None,
) -> float:
    """
    Compute forward transfer (FWT).

    FWT measures how much learning previous tasks helps
    learning new tasks.

    Args:
        accuracy_matrix: Accuracy matrix
        baseline_accuracy: Baseline accuracy without transfer
                          (if None, uses first epoch accuracy)

    Returns:
        Forward transfer score (positive = beneficial)
    """
    # This requires a baseline comparison
    # For now, return 0 as placeholder
    return 0.0


def plot_accuracy_curves(
    summaries: Sequence[TaskRunSummary],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training accuracy curves for all tasks.

    Args:
        summaries: List of task summaries
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot epoch accuracies
    ax1 = axes[0]
    for summary in summaries:
        if summary.epoch_accuracies:
            epochs = range(1, len(summary.epoch_accuracies) + 1)
            ax1.plot(
                epochs,
                summary.epoch_accuracies,
                label=f"Task {summary.task_id} ({summary.classes})",
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Training Progress per Task")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot final accuracies
    ax2 = axes[1]
    task_ids = [s.task_id for s in summaries]
    train_accs = [s.train_accuracy for s in summaries]
    test_accs = [s.test_accuracy for s in summaries]

    x = np.arange(len(task_ids))
    width = 0.35

    ax2.bar(x - width / 2, train_accs, width, label="Train", alpha=0.8)
    ax2.bar(x + width / 2, test_accs, width, label="Test", alpha=0.8)

    ax2.set_xlabel("Task ID")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Final Accuracy per Task")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T{i}" for i in task_ids])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_accuracy_matrix(
    matrix: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot accuracy matrix as a heatmap.

    Args:
        matrix: Accuracy matrix [trained_up_to, evaluated_on]
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

    # Set labels
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels([f"T{i}" for i in range(matrix.shape[1])])
    ax.set_yticklabels([f"After T{i}" for i in range(matrix.shape[0])])

    ax.set_xlabel("Evaluated on Task")
    ax.set_ylabel("Trained up to Task")
    ax.set_title("Accuracy Matrix")

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if matrix[i, j] > 0.5 else "white",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_forgetting_analysis(
    accuracy_matrix: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot forgetting analysis.

    Args:
        accuracy_matrix: Accuracy matrix
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot per-task forgetting
    ax1 = axes[0]
    forgetting = compute_forgetting(accuracy_matrix)
    task_ids = range(len(forgetting))

    ax1.bar(task_ids, forgetting, alpha=0.8)
    ax1.set_xlabel("Task ID")
    ax1.set_ylabel("Forgetting (accuracy drop)")
    ax1.set_title("Per-Task Forgetting")
    ax1.set_xticks(task_ids)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot accuracy evolution per task
    ax2 = axes[1]
    num_tasks = accuracy_matrix.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))

    for task in range(num_tasks):
        # Plot accuracy on this task over training
        task_accs = accuracy_matrix[:, task]
        # Only show from when task was trained
        x = range(task, len(task_accs))
        y = task_accs[task:]
        ax2.plot(
            x, y, marker="o", markersize=4, label=f"Task {task}", color=colors[task]
        )

    ax2.set_xlabel("After Training Task")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Evolution per Task")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_support_selection(
    summaries: Sequence[TaskRunSummary],
    num_columns: int,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot support column selection across tasks.

    Args:
        summaries: List of task summaries
        num_columns: Total number of columns
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create selection matrix
    num_tasks = len(summaries)
    selection = np.zeros((num_tasks, num_columns))

    for i, summary in enumerate(summaries):
        for col in summary.support_cols:
            if col < num_columns:
                selection[i, col] = 1

    im = ax.imshow(selection, cmap="Blues", aspect="auto")

    ax.set_xlabel("Column Index")
    ax.set_ylabel("Task ID")
    ax.set_title("Support Column Selection")

    # Set ticks
    ax.set_yticks(range(num_tasks))
    ax.set_yticklabels([f"Task {s.task_id}" for s in summaries])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def print_summary_table(summaries: Sequence[TaskRunSummary]):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("Task Summary")
    print("=" * 80)
    print(
        f"{'Task':<6} {'Classes':<12} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)':<10}"
    )
    print("-" * 80)

    for s in summaries:
        print(
            f"{s.task_id:<6} {str(s.classes):<12} {s.train_accuracy:<12.4f} "
            f"{s.test_accuracy:<12.4f} {s.training_time:<10.1f}"
        )

    print("=" * 80)

    # Overall statistics
    avg_train = np.mean([s.train_accuracy for s in summaries])
    avg_test = np.mean([s.test_accuracy for s in summaries])
    total_time = sum(s.training_time for s in summaries)

    print(f"\nAverage Train Accuracy: {avg_train:.4f}")
    print(f"Average Test Accuracy:  {avg_test:.4f}")
    print(f"Total Training Time:    {total_time:.1f}s")


def create_run_directory(
    base_path: str,
    run_tag: str,
    seed: int,
) -> Path:
    """
    Create a directory for experiment outputs.

    Args:
        base_path: Base path for experiments
        run_tag: Tag for this run
        seed: Random seed

    Returns:
        Path to the created directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_tag}_seed{seed}_{timestamp}"

    run_dir = Path(base_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_experiment_config(
    config: "ExperimentConfig",
    path: str,
):
    """Save experiment configuration to JSON."""
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    config_dict = asdict(config)

    # Convert tuples to lists for JSON serialization
    def convert_tuples(obj):
        if isinstance(obj, dict):
            return {k: convert_tuples(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, list):
            return [convert_tuples(v) for v in obj]
        return obj

    config_dict = convert_tuples(config_dict)

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
