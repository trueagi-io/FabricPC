"""
Utility functions for Continual Learning.

Provides plotting, serialization, and analysis helpers for
continual learning experiments.
"""

from typing import Dict, List, Tuple, Optional, Any, Sequence
from pathlib import Path
import json
import os
import numpy as np

from fabricpc.continual.trainer import TaskRunSummary


def _save_plotly_figure(fig, save_path: Optional[str]) -> None:
    """
    Save a Plotly figure with a graceful fallback when static image export fails.

    Plotly's static export path may require Kaleido plus a Chrome installation.
    When that is unavailable, save an interactive HTML file instead of failing.
    """
    if not save_path:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plotly_browser()

    try:
        fig.write_image(save_path)
    except Exception as exc:
        html_path = save_path.with_suffix(".html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(
            f"Static plot export failed for {save_path.name}: {exc}. "
            f"Saved interactive HTML to {html_path.name} instead."
        )


def _configure_plotly_browser() -> None:
    """
    Help Kaleido discover Chrome/Chromium in common non-standard install locations.

    In particular, Ubuntu systems often have Chrome at `/opt/google/chrome/chrome`
    without a matching executable on PATH.
    """
    browser_candidates = [
        Path("/opt/google/chrome/chrome"),
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/google-chrome-stable"),
        Path("/usr/bin/chromium"),
        Path("/usr/bin/chromium-browser"),
    ]

    if os.environ.get("BROWSER_PATH"):
        return

    for candidate in browser_candidates:
        if candidate.exists():
            os.environ["BROWSER_PATH"] = str(candidate)
            os.environ.setdefault("CHROME_PATH", str(candidate))
            browser_dir = str(candidate.parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if browser_dir not in path_entries:
                os.environ["PATH"] = (
                    browser_dir + os.pathsep + os.environ.get("PATH", "")
                )
            return


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
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not available for plotting")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Training Progress per Task", "Final Accuracy per Task"),
    )

    # Plot epoch accuracies
    for summary in summaries:
        if summary.epoch_accuracies:
            epochs = list(range(1, len(summary.epoch_accuracies) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary.epoch_accuracies,
                    mode="lines+markers",
                    name=f"Task {summary.task_id} ({summary.classes})",
                ),
                row=1,
                col=1,
            )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Test Accuracy", row=1, col=1)

    # Plot final accuracies
    task_ids = [s.task_id for s in summaries]
    train_accs = [s.train_accuracy for s in summaries]
    test_accs = [s.test_accuracy for s in summaries]

    fig.add_trace(
        go.Bar(
            x=[f"T{i}" for i in task_ids],
            y=train_accs,
            name="Train",
            opacity=0.8,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[f"T{i}" for i in task_ids],
            y=test_accs,
            name="Test",
            opacity=0.8,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Task ID", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    fig.update_layout(height=500, width=1200, showlegend=True)

    _save_plotly_figure(fig, save_path)

    if show:
        fig.show()


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
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not available for plotting")
        return

    # Create text annotations
    text_matrix = [
        [f"{matrix[i, j]:.2f}" for j in range(matrix.shape[1])]
        for i in range(matrix.shape[0])
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"T{i}" for i in range(matrix.shape[1])],
            y=[f"After T{i}" for i in range(matrix.shape[0])],
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Accuracy"),
        )
    )

    fig.update_layout(
        title="Accuracy Matrix",
        xaxis_title="Evaluated on Task",
        yaxis_title="Trained up to Task",
        height=600,
        width=800,
    )

    _save_plotly_figure(fig, save_path)

    if show:
        fig.show()


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
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError:
        print("plotly not available for plotting")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Per-Task Forgetting", "Accuracy Evolution per Task"),
    )

    # Plot per-task forgetting
    forgetting = compute_forgetting(accuracy_matrix)
    task_ids = list(range(len(forgetting)))

    fig.add_trace(
        go.Bar(
            x=[f"T{i}" for i in task_ids],
            y=forgetting,
            opacity=0.8,
            name="Forgetting",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Task ID", row=1, col=1)
    fig.update_yaxes(title_text="Forgetting (accuracy drop)", row=1, col=1)

    # Plot accuracy evolution per task
    num_tasks = accuracy_matrix.shape[1]
    colors = px.colors.qualitative.Plotly

    for task in range(num_tasks):
        # Plot accuracy on this task over training
        task_accs = accuracy_matrix[:, task]
        # Only show from when task was trained
        x = list(range(task, len(task_accs)))
        y = task_accs[task:].tolist()
        fig.add_trace(
            go.Scatter(
                x=[f"T{i}" for i in x],
                y=y,
                mode="lines+markers",
                name=f"Task {task}",
                marker=dict(size=6),
                line=dict(color=colors[task % len(colors)]),
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="After Training Task", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    fig.update_layout(height=500, width=1200, showlegend=True)

    _save_plotly_figure(fig, save_path)

    if show:
        fig.show()


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
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not available for plotting")
        return

    # Create selection matrix
    num_tasks = len(summaries)
    selection = np.zeros((num_tasks, num_columns))

    for i, summary in enumerate(summaries):
        for col in summary.support_cols:
            if col < num_columns:
                selection[i, col] = 1

    fig = go.Figure(
        data=go.Heatmap(
            z=selection,
            x=list(range(num_columns)),
            y=[f"Task {s.task_id}" for s in summaries],
            colorscale="Blues",
            showscale=False,
        )
    )

    fig.update_layout(
        title="Support Column Selection",
        xaxis_title="Column Index",
        yaxis_title="Task ID",
        height=600,
        width=1000,
    )

    _save_plotly_figure(fig, save_path)

    if show:
        fig.show()


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
    print(f"Summed Task Train Time: {total_time:.1f}s")


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
