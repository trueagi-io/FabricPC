"""
Plotting utilities for continual learning benchmarks.

Provides functions for visualizing accuracy matrices,
forgetting analysis, and method comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cl_benchmark.evaluation.results import BenchmarkResults


def plot_accuracy_matrix(
    matrix: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Accuracy Matrix",
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot accuracy matrix as a heatmap.

    Args:
        matrix: Accuracy matrix of shape (T, T)
        save_path: Optional path to save the figure
        show: Whether to display the plot
        title: Plot title
        figsize: Figure size (width, height)
    """
    import plotly.graph_objects as go

    num_tasks = matrix.shape[0]

    # Create text annotations (only for lower triangle where data exists)
    text_matrix = [
        [f"{matrix[i, j]:.2f}" if i >= j else "" for j in range(num_tasks)]
        for i in range(num_tasks)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"T{i}" for i in range(num_tasks)],
            y=[f"T{i}" for i in range(num_tasks)],
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="YlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Accuracy"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Evaluated on Task",
        yaxis_title="After Training Task",
        height=figsize[1] * 100,
        width=figsize[0] * 100,
    )

    if save_path:
        fig.write_image(save_path)

    if show:
        fig.show()

    return fig


def plot_forgetting_analysis(
    matrix: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 4),
):
    """
    Plot forgetting analysis with two subplots:
    1. Per-task forgetting bars
    2. Accuracy evolution over time for each task

    Args:
        matrix: Accuracy matrix of shape (T, T)
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size (width, height)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    from cl_benchmark.metrics import compute_forgetting

    num_tasks = matrix.shape[0]
    forgetting = compute_forgetting(matrix)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Per-Task Forgetting", "Accuracy Evolution"),
    )

    # Plot 1: Forgetting bars
    colors = px.colors.sequential.Reds[3:]
    bar_colors = [colors[min(i, len(colors) - 1)] for i in range(num_tasks)]

    fig.add_trace(
        go.Bar(
            x=[f"T{i}" for i in range(num_tasks)],
            y=forgetting,
            marker_color=bar_colors,
            text=[f"{v:.2f}" if v > 0.01 else "" for v in forgetting],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Forgetting",
        range=[0, max(0.5, max(forgetting) * 1.1)],
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Task", row=1, col=1)

    # Plot 2: Accuracy evolution
    colors = px.colors.qualitative.Plotly

    for task in range(num_tasks):
        accuracies = matrix[task:, task]  # Accuracy on task 'task' over time
        x = list(range(task, num_tasks))
        fig.add_trace(
            go.Scatter(
                x=[f"T{i}" for i in x],
                y=accuracies.tolist(),
                mode="lines+markers",
                name=f"Task {task}",
                marker=dict(size=6),
                line=dict(color=colors[task % len(colors)]),
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="After Training Task", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=1, col=2)

    fig.update_layout(
        height=figsize[1] * 100,
        width=figsize[0] * 100,
        showlegend=True,
    )

    if save_path:
        fig.write_image(save_path)

    if show:
        fig.show()

    return fig


def plot_learning_curves(
    results: "BenchmarkResults",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot learning curves showing accuracy on each task over time.

    Args:
        results: BenchmarkResults containing accuracy matrices
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size (width, height)
    """
    import plotly.graph_objects as go
    import plotly.express as px

    if not results.accuracy_matrices:
        print("No accuracy matrices to plot.")
        return

    mean_matrix = results.get_mean_accuracy_matrix()
    std_matrix = results.get_std_accuracy_matrix()
    num_tasks = mean_matrix.shape[0]

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for task in range(num_tasks):
        # Get accuracy on this task over time (after it was learned)
        mean_accs = mean_matrix[task:, task]
        std_accs = std_matrix[task:, task]
        x = list(range(task, num_tasks))
        x_labels = [f"T{i}" for i in x]

        color = colors[task % len(colors)]

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=mean_accs.tolist(),
                mode="lines+markers",
                name=f"Task {task}",
                line=dict(color=color),
                marker=dict(size=6),
            )
        )

        # Add std band
        fig.add_trace(
            go.Scatter(
                x=x_labels + x_labels[::-1],
                y=(mean_accs + std_accs).tolist()
                + (mean_accs - std_accs).tolist()[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Learning Curves: {results.model_name}",
        xaxis_title="After Training Task",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        height=figsize[1] * 100,
        width=figsize[0] * 100,
        showlegend=True,
    )

    if save_path:
        fig.write_image(save_path)

    if show:
        fig.show()

    return fig


def plot_method_comparison(
    results_dict: Dict[str, "BenchmarkResults"],
    metric: str = "average_accuracy",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot comparison of multiple methods.

    Args:
        results_dict: Dictionary mapping method names to BenchmarkResults
        metric: Which metric to compare ("average_accuracy", "forgetting", "bwt")
        save_path: Optional path to save the figure
        show: Whether to display the plot
        figsize: Figure size (width, height)
    """
    import plotly.graph_objects as go
    import plotly.express as px

    if not results_dict:
        print("No results to compare.")
        return

    # Get metric values
    methods = list(results_dict.keys())
    means = []
    stds = []

    for name in methods:
        results = results_dict[name]
        if metric == "average_accuracy":
            means.append(results.accuracy_mean)
            stds.append(results.accuracy_std)
        elif metric == "forgetting":
            means.append(results.forgetting_mean)
            stds.append(results.forgetting_std)
        elif metric == "bwt":
            means.append(results.backward_transfer)
            stds.append(0.0)  # No std for BWT currently
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Sort by mean value
    sorted_idx = np.argsort(means)[::-1]
    methods = [methods[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    # Create bar chart
    colors = px.colors.sequential.Viridis[2:-2]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=methods,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=bar_colors,
            text=[f"{m:.3f}" for m in means],
            textposition="outside",
        )
    )

    y_max = 1.1 if metric == "average_accuracy" else max(means) * 1.3 if means else 0.5

    fig.update_layout(
        title=f"Method Comparison: {metric.replace('_', ' ').title()}",
        xaxis_title="Method",
        yaxis_title=metric.replace("_", " ").title(),
        yaxis_range=[0, y_max],
        height=figsize[1] * 100,
        width=figsize[0] * 100,
    )

    if save_path:
        fig.write_image(save_path)

    if show:
        fig.show()

    return fig
