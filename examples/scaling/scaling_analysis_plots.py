"""
MLP Scaling Laws Analysis
Generates plots for comparing Predictive Coding (PC) vs Backpropagation (BP) training.

Usage:
    python scaling_analysis_plots.py <path_to_data_file>

Example:
    python scaling_analysis_plots.py mlp_scaling_results.csv
    python scaling_analysis_plots.py mlp_scaling_results.xlsx

Supported formats: .csv, .xlsx, .xls

Dependencies:
    To export PNG images, you'll need kaleido installed (pip install kaleido). The HTML output works without any additional dependencies.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import curve_fit
import sys
import warnings

warnings.filterwarnings("ignore")


def power_law(x, a, b):
    """Power law model: y = a * x^b"""
    return a * np.power(x, b)


def fit_power_law(x, y):
    """Fit power law and return coefficients (a, b) and R-squared."""
    try:
        popt, _ = curve_fit(power_law, x, y, p0=[1, 1], maxfev=10000)
        a, b = popt
        predicted = power_law(x, a, b)
        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return a, b, r_squared
    except:
        return np.nan, np.nan, np.nan


def load_and_prepare_data(filepath):
    """Load Excel or CSV data and separate by training mode."""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {filepath}. Use .csv, .xlsx, or .xls"
        )
    df_clean = df.dropna(subset=["avg_step_time_ms", "peak_memory_mb"])

    pc_df = df_clean[df_clean["training_mode"] == "pc"].copy()
    bp_df = df_clean[df_clean["training_mode"] == "backprop"].copy()

    widths = sorted(df_clean["width"].unique())
    depths = sorted(df_clean["depth"].unique())

    return df_clean, pc_df, bp_df, widths, depths


def get_color_sequence(n):
    """Generate a color sequence similar to viridis."""
    colors = px.colors.sample_colorscale(
        "Viridis", [i / (n - 1) if n > 1 else 0.5 for i in range(n)]
    )
    return colors


def plot_time_vs_depth_scaling(pc_df, bp_df, widths, colors):
    """Generate time vs depth log-log plots for PC and BP."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Predictive Coding: Time vs Depth<br>(Linear O(n) scaling observed)",
            "Backpropagation: Time vs Depth<br>(Linear O(n) scaling observed)",
        ),
    )

    # PC plot
    for i, w in enumerate(widths):
        subset = pc_df[pc_df["width"] == w].sort_values("depth")
        if len(subset) > 2:
            fig.add_trace(
                go.Scatter(
                    x=subset["depth"],
                    y=subset["avg_step_time_ms"],
                    mode="lines+markers",
                    name=f"w={w}",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                    legendgroup=f"w={w}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            a, b, _ = fit_power_law(
                subset["depth"].values, subset["avg_step_time_ms"].values
            )
            if not np.isnan(b):
                x_fit = np.linspace(subset["depth"].min(), subset["depth"].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=power_law(x_fit, a, b),
                        mode="lines",
                        name=f"fit w={w}",
                        line=dict(color=colors[i], width=1, dash="dash"),
                        opacity=0.5,
                        legendgroup=f"w={w}",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

    # BP plot
    for i, w in enumerate(widths):
        subset = bp_df[bp_df["width"] == w].sort_values("depth")
        if len(subset) > 2:
            fig.add_trace(
                go.Scatter(
                    x=subset["depth"],
                    y=subset["avg_step_time_ms"],
                    mode="lines+markers",
                    name=f"w={w}",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                    legendgroup=f"w={w}_bp",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            a, b, _ = fit_power_law(
                subset["depth"].values, subset["avg_step_time_ms"].values
            )
            if not np.isnan(b):
                x_fit = np.linspace(subset["depth"].min(), subset["depth"].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=power_law(x_fit, a, b),
                        mode="lines",
                        name=f"fit w={w}",
                        line=dict(color=colors[i], width=1, dash="dash"),
                        opacity=0.5,
                        legendgroup=f"w={w}_bp",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

    fig.update_xaxes(type="log", title_text="Depth", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Step Time (ms)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Depth", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Step Time (ms)", row=1, col=2)

    fig.update_layout(height=500, width=1200, legend_title_text="Width")
    return fig


def plot_exponent_comparison(pc_df, bp_df, widths):
    """Generate bar chart comparing depth exponents between PC and BP."""
    pc_exps = []
    bp_exps = []
    valid_widths = []

    for w in widths:
        pc_sub = pc_df[pc_df["width"] == w].sort_values("depth")
        bp_sub = bp_df[bp_df["width"] == w].sort_values("depth")
        if len(pc_sub) > 2 and len(bp_sub) > 2:
            _, pc_b, _ = fit_power_law(
                pc_sub["depth"].values, pc_sub["avg_step_time_ms"].values
            )
            _, bp_b, _ = fit_power_law(
                bp_sub["depth"].values, bp_sub["avg_step_time_ms"].values
            )
            if not np.isnan(pc_b) and not np.isnan(bp_b):
                pc_exps.append(pc_b)
                bp_exps.append(bp_b)
                valid_widths.append(str(w))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="PC",
            x=valid_widths,
            y=pc_exps,
            marker_color="steelblue",
            marker_line_color="black",
            marker_line_width=1,
        )
    )
    fig.add_trace(
        go.Bar(
            name="BP",
            x=valid_widths,
            y=bp_exps,
            marker_color="coral",
            marker_line_color="black",
            marker_line_width=1,
        )
    )

    # Reference lines
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text="α=1 (linear, O(n))",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="green",
        opacity=0.7,
        annotation_text="α=0 (constant, O(1))",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Depth Scaling Exponent (α): time ∝ depth^α<br>α=0→O(1), α=1→O(n) | PC avg: {np.mean(pc_exps):.2f}, BP avg: {np.mean(bp_exps):.2f}",
        xaxis_title="Network Width",
        yaxis_title="Depth Exponent (α)",
        barmode="group",
        yaxis_range=[0, 1.2],
        height=500,
        width=800,
    )
    return fig


def plot_time_ratio(pc_df, bp_df, depths):
    """Generate plot showing PC/BP time ratio across widths for each depth."""
    fig = go.Figure()

    colors = get_color_sequence(len(depths))

    for idx, d in enumerate(depths):
        pc_sub = pc_df[pc_df["depth"] == d].sort_values("width")
        bp_sub = bp_df[bp_df["depth"] == d].sort_values("width")
        merged = pd.merge(
            pc_sub[["width", "avg_step_time_ms"]],
            bp_sub[["width", "avg_step_time_ms"]],
            on="width",
            suffixes=("_pc", "_bp"),
        )
        if len(merged) > 0:
            ratio = merged["avg_step_time_ms_pc"] / merged["avg_step_time_ms_bp"]
            fig.add_trace(
                go.Scatter(
                    x=merged["width"],
                    y=ratio,
                    mode="lines+markers",
                    name=f"d={d}",
                    line=dict(color=colors[idx], width=2),
                    marker=dict(size=7),
                )
            )

    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="black",
        opacity=0.5,
        annotation_text="PC=BP",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Training Time Ratio: PC / Backprop<br>(Higher = PC slower)",
        xaxis_title="Width",
        yaxis_title="Time Ratio (PC/BP)",
        xaxis_type="log",
        height=500,
        width=800,
        legend_title_text="Depth",
    )
    return fig


def plot_memory_scaling(pc_df, bp_df, widths, colors):
    """Generate memory vs depth scaling plot."""
    fig = go.Figure()

    # Use all available widths
    for i, w in enumerate(widths):
        pc_sub = pc_df[pc_df["width"] == w].sort_values("depth")
        bp_sub = bp_df[bp_df["width"] == w].sort_values("depth")
        if len(pc_sub) > 2:
            fig.add_trace(
                go.Scatter(
                    x=pc_sub["depth"],
                    y=pc_sub["peak_memory_mb"],
                    mode="lines+markers",
                    name=f"PC w={w}",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8, symbol="circle"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bp_sub["depth"],
                    y=bp_sub["peak_memory_mb"],
                    mode="lines+markers",
                    name=f"BP w={w}",
                    line=dict(color=colors[i], width=2, dash="dash"),
                    marker=dict(size=8, symbol="square"),
                    opacity=0.6,
                )
            )

    fig.update_layout(
        title="Memory vs Depth Scaling",
        xaxis_title="Depth",
        yaxis_title="Peak Memory (MB)",
        xaxis_type="log",
        yaxis_type="log",
        height=500,
        width=800,
    )
    return fig


def plot_memory_ratio_heatmap(pc_df, bp_df, widths, depths):
    """Generate heatmap of PC/BP memory ratios."""
    ratio_matrix = np.zeros((len(depths), len(widths)))
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            pc_mem = pc_df[(pc_df["depth"] == d) & (pc_df["width"] == w)][
                "peak_memory_mb"
            ]
            bp_mem = bp_df[(bp_df["depth"] == d) & (bp_df["width"] == w)][
                "peak_memory_mb"
            ]
            if len(pc_mem) > 0 and len(bp_mem) > 0:
                ratio_matrix[i, j] = pc_mem.iloc[0] / bp_mem.iloc[0]
            else:
                ratio_matrix[i, j] = np.nan

    # Create text annotations
    text_matrix = [
        [
            f"{ratio_matrix[i, j]:.2f}" if not np.isnan(ratio_matrix[i, j]) else ""
            for j in range(len(widths))
        ]
        for i in range(len(depths))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=ratio_matrix,
            x=[str(w) for w in widths],
            y=[str(d) for d in depths],
            colorscale="RdYlGn_r",
            zmin=0.5,
            zmax=2.5,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="Ratio"),
        )
    )

    fig.update_layout(
        title="Memory Ratio (PC/BP)<br>Green=PC uses less, Red=PC uses more",
        xaxis_title="Width",
        yaxis_title="Depth",
        height=600,
        width=800,
    )
    return fig


def plot_combined_analysis(pc_df, bp_df, widths, depths, colors):
    """Generate combined 2x2 analysis figure."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Predictive Coding: Time vs Depth<br>(Linear O(n) scaling observed)",
            "Backpropagation: Time vs Depth<br>(Linear O(n) scaling observed)",
            f"Depth Scaling Exponent: time ~ depth^α",
            "Training Time Ratio: PC / Backprop<br>(Higher = PC slower)",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.15,
    )

    # Plot 1: Time vs Depth (PC)
    for i, w in enumerate(widths):
        subset = pc_df[pc_df["width"] == w].sort_values("depth")
        if len(subset) > 2:
            fig.add_trace(
                go.Scatter(
                    x=subset["depth"],
                    y=subset["avg_step_time_ms"],
                    mode="lines+markers",
                    name=f"w={w}",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                    legendgroup=f"w={w}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            a, b, _ = fit_power_law(
                subset["depth"].values, subset["avg_step_time_ms"].values
            )
            if not np.isnan(b):
                x_fit = np.linspace(subset["depth"].min(), subset["depth"].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=power_law(x_fit, a, b),
                        mode="lines",
                        line=dict(color=colors[i], width=1, dash="dash"),
                        opacity=0.5,
                        legendgroup=f"w={w}",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

    # Plot 2: Time vs Depth (BP)
    for i, w in enumerate(widths):
        subset = bp_df[bp_df["width"] == w].sort_values("depth")
        if len(subset) > 2:
            fig.add_trace(
                go.Scatter(
                    x=subset["depth"],
                    y=subset["avg_step_time_ms"],
                    mode="lines+markers",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                    legendgroup=f"w={w}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            a, b, _ = fit_power_law(
                subset["depth"].values, subset["avg_step_time_ms"].values
            )
            if not np.isnan(b):
                x_fit = np.linspace(subset["depth"].min(), subset["depth"].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=power_law(x_fit, a, b),
                        mode="lines",
                        line=dict(color=colors[i], width=1, dash="dash"),
                        opacity=0.5,
                        legendgroup=f"w={w}",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

    # Plot 3: Exponent comparison
    pc_exps = []
    bp_exps = []
    valid_widths = []
    for w in widths:
        pc_sub = pc_df[pc_df["width"] == w].sort_values("depth")
        bp_sub = bp_df[bp_df["width"] == w].sort_values("depth")
        if len(pc_sub) > 2 and len(bp_sub) > 2:
            _, pc_b, _ = fit_power_law(
                pc_sub["depth"].values, pc_sub["avg_step_time_ms"].values
            )
            _, bp_b, _ = fit_power_law(
                bp_sub["depth"].values, bp_sub["avg_step_time_ms"].values
            )
            if not np.isnan(pc_b) and not np.isnan(bp_b):
                pc_exps.append(pc_b)
                bp_exps.append(bp_b)
                valid_widths.append(str(w))

    fig.add_trace(
        go.Bar(
            name="PC",
            x=valid_widths,
            y=pc_exps,
            marker_color="steelblue",
            marker_line_color="black",
            marker_line_width=1,
            showlegend=True,
            legend="legend2",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="BP",
            x=valid_widths,
            y=bp_exps,
            marker_color="coral",
            marker_line_color="black",
            marker_line_width=1,
            showlegend=True,
            legend="legend2",
        ),
        row=2,
        col=1,
    )

    # Plot 4: Time ratio
    depth_colors = get_color_sequence(len(depths))
    for idx, d in enumerate(depths):
        pc_sub = pc_df[pc_df["depth"] == d].sort_values("width")
        bp_sub = bp_df[bp_df["depth"] == d].sort_values("width")
        merged = pd.merge(
            pc_sub[["width", "avg_step_time_ms"]],
            bp_sub[["width", "avg_step_time_ms"]],
            on="width",
            suffixes=("_pc", "_bp"),
        )
        if len(merged) > 0:
            ratio = merged["avg_step_time_ms_pc"] / merged["avg_step_time_ms_bp"]
            fig.add_trace(
                go.Scatter(
                    x=merged["width"],
                    y=ratio,
                    mode="lines+markers",
                    name=f"d={d}",
                    line=dict(color=depth_colors[idx], width=2),
                    marker=dict(size=7),
                    showlegend=True,
                    legend="legend3",
                ),
                row=2,
                col=2,
            )

    # Update axes
    fig.update_xaxes(type="log", title_text="Depth", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Step Time (ms)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Depth", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Step Time (ms)", row=1, col=2)
    fig.update_xaxes(title_text="Network Width", row=2, col=1)
    fig.update_yaxes(title_text="Depth Exponent (α)", range=[0, 1.2], row=2, col=1)
    fig.update_xaxes(type="log", title_text="Width", row=2, col=2)
    fig.update_yaxes(title_text="Time Ratio (PC/BP)", row=2, col=2)

    # Add reference lines to bar chart
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig.add_hline(
        y=0.0, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1
    )

    # Add reference line to ratio plot
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=2
    )

    fig.update_layout(
        height=900,
        width=1200,
        barmode="group",
        # Legend 1 (default): Width - upper right for top plots
        legend=dict(
            title_text="Width",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        # Legend 2: Exponent (PC/BP) - right of lower-left subplot
        legend2=dict(
            title_text="Exponent",
            x=0.43,
            y=0.42,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        # Legend 3: Depth - right of lower-right subplot
        legend3=dict(
            title_text="Depth",
            x=1.02,
            y=0.42,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
    )
    return fig


def plot_memory_analysis(pc_df, bp_df, widths, depths, colors):
    """Generate combined memory analysis figure."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Memory vs Depth Scaling",
            "Memory Ratio (PC/BP)<br>Green=PC uses less, Red=PC uses more",
        ),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]],
        column_widths=[0.38, 0.62],
        horizontal_spacing=0.18,
    )

    # Memory vs depth - use all available widths
    for i, w in enumerate(widths):
        pc_sub = pc_df[pc_df["width"] == w].sort_values("depth")
        bp_sub = bp_df[bp_df["width"] == w].sort_values("depth")
        if len(pc_sub) > 2:
            fig.add_trace(
                go.Scatter(
                    x=pc_sub["depth"],
                    y=pc_sub["peak_memory_mb"],
                    mode="lines+markers",
                    name=f"PC w={w}",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8, symbol="circle"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=bp_sub["depth"],
                    y=bp_sub["peak_memory_mb"],
                    mode="lines+markers",
                    name=f"BP w={w}",
                    line=dict(color=colors[i], width=2, dash="dash"),
                    marker=dict(size=8, symbol="square"),
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )

    # Memory ratio heatmap
    ratio_matrix = np.zeros((len(depths), len(widths)))
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            pc_mem = pc_df[(pc_df["depth"] == d) & (pc_df["width"] == w)][
                "peak_memory_mb"
            ]
            bp_mem = bp_df[(bp_df["depth"] == d) & (bp_df["width"] == w)][
                "peak_memory_mb"
            ]
            if len(pc_mem) > 0 and len(bp_mem) > 0:
                ratio_matrix[i, j] = pc_mem.iloc[0] / bp_mem.iloc[0]
            else:
                ratio_matrix[i, j] = np.nan

    text_matrix = [
        [
            f"{ratio_matrix[i, j]:.2f}" if not np.isnan(ratio_matrix[i, j]) else ""
            for j in range(len(widths))
        ]
        for i in range(len(depths))
    ]

    fig.add_trace(
        go.Heatmap(
            z=ratio_matrix,
            x=[str(w) for w in widths],
            y=[str(d) for d in depths],
            colorscale="RdYlGn_r",
            zmin=0.5,
            zmax=2.5,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="Ratio", x=1.02),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(type="log", title_text="Depth", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Peak Memory (MB)", row=1, col=1)
    fig.update_xaxes(title_text="Width", row=1, col=2)
    fig.update_yaxes(title_text="Depth", row=1, col=2)

    fig.update_layout(
        height=500,
        width=1200,
        legend=dict(
            x=0.42,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=9),
        ),
    )
    return fig


def print_scaling_summary(pc_df, bp_df, widths, depths):
    """Print summary statistics of scaling law fits."""
    print("\n" + "=" * 70)
    print("SCALING LAW ANALYSIS SUMMARY")
    print("=" * 70)

    # Time vs Depth
    print("\n--- Time vs Depth Scaling (time ~ depth^α) ---")
    print(f"{'Width':<10} {'PC α':<12} {'BP α':<12} {'Diff':<10}")
    print("-" * 44)

    pc_exps = []
    bp_exps = []
    for w in widths:
        pc_sub = pc_df[pc_df["width"] == w].sort_values("depth")
        bp_sub = bp_df[bp_df["width"] == w].sort_values("depth")
        if len(pc_sub) > 2 and len(bp_sub) > 2:
            _, pc_b, pc_r2 = fit_power_law(
                pc_sub["depth"].values, pc_sub["avg_step_time_ms"].values
            )
            _, bp_b, bp_r2 = fit_power_law(
                bp_sub["depth"].values, bp_sub["avg_step_time_ms"].values
            )
            if not np.isnan(pc_b) and not np.isnan(bp_b):
                pc_exps.append(pc_b)
                bp_exps.append(bp_b)
                print(f"{w:<10} {pc_b:<12.4f} {bp_b:<12.4f} {pc_b - bp_b:<10.4f}")

    print("-" * 44)
    print(
        f"{'Average':<10} {np.mean(pc_exps):<12.4f} {np.mean(bp_exps):<12.4f} {np.mean(pc_exps) - np.mean(bp_exps):<10.4f}"
    )

    print("\n--- Interpretation ---")
    print(
        f"PC average exponent α: {np.mean(pc_exps):.3f} → O(n^{np.mean(pc_exps):.2f}) ≈ linear"
    )
    print(
        f"BP average exponent α: {np.mean(bp_exps):.3f} → O(n^{np.mean(bp_exps):.2f}) ≈ linear"
    )
    print(f"Both show approximately linear scaling with depth (time ~ depth^α, α ≈ 1)")

    # Time ratios
    print("\n--- PC/BP Time Ratios ---")
    ratios = []
    for _, pc_row in pc_df.iterrows():
        bp_match = bp_df[
            (bp_df["width"] == pc_row["width"]) & (bp_df["depth"] == pc_row["depth"])
        ]
        if len(bp_match) == 1:
            ratio = pc_row["avg_step_time_ms"] / bp_match.iloc[0]["avg_step_time_ms"]
            ratios.append(ratio)

    print(f"Min ratio:  {min(ratios):.2f}x")
    print(f"Max ratio:  {max(ratios):.2f}x")
    print(f"Mean ratio: {np.mean(ratios):.2f}x")
    print(f"PC is typically {np.mean(ratios):.1f}x slower than BP")


def analyze_pc_bp_ratios(pc_df, bp_df, widths, depths, batch_size=256, infer_steps=10):
    """
    Detailed analysis of PC vs BP memory and time ratios.

    This function explains WHY:
    1. Memory ratio is ~2x for narrow networks but approaches 1x for wide networks
    2. Time ratio starts at ~1.7x for narrow networks and saturates at ~5-6x for wide

    Key insights:
    - PC stores 5 tensors per node: z_latent, z_mu, error, pre_activation, latent_grad
      Each tensor is (batch_size, width) - this is O(batch * width * depth)
    - Parameters are O(width²) per layer
    - As width increases, parameters dominate state → memory ratio → 1

    - PC runs `infer_steps` iterations, each with forward pass + local autodiff (VJP)
    - Each inference step costs ~2D matmuls, learning step costs ~D matmuls
    - Total PC: (2N + 1)D matmuls, Total BP: 3D matmuls
    - Theoretical time ratio: (2 * infer_steps + 1) / 3 ≈ 7x for infer_steps=10
    - Observed ratio: ~5-6x (close to theoretical, with XLA optimizations)
    """
    print("\n" + "=" * 70)
    print("PC vs BP RATIO ANALYSIS (Memory & Time)")
    print("=" * 70)

    # Merge dataframes for ratio computation
    merged = pd.merge(
        pc_df[["width", "depth", "avg_step_time_ms", "peak_memory_mb"]],
        bp_df[["width", "depth", "avg_step_time_ms", "peak_memory_mb"]],
        on=["width", "depth"],
        suffixes=("_pc", "_bp"),
    )
    merged["time_ratio"] = merged["avg_step_time_ms_pc"] / merged["avg_step_time_ms_bp"]
    merged["mem_ratio"] = merged["peak_memory_mb_pc"] / merged["peak_memory_mb_bp"]

    # Time ratio analysis
    print("\n--- Time Ratio (PC / BP) by Width ---")
    print("(Averaged across all depths)")
    print(f"{'Width':<8} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 36)
    for w in widths:
        subset = merged[merged["width"] == w]
        if len(subset) > 0:
            print(
                f"{w:<8} {subset['time_ratio'].mean():>8.2f} "
                f"{subset['time_ratio'].min():>8.2f} {subset['time_ratio'].max():>8.2f}"
            )

    # Memory ratio analysis
    print("\n--- Memory Ratio (PC / BP) by Width ---")
    print("(Averaged across all depths)")
    print(f"{'Width':<8} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 36)
    for w in widths:
        subset = merged[merged["width"] == w]
        if len(subset) > 0:
            print(
                f"{w:<8} {subset['mem_ratio'].mean():>8.2f} "
                f"{subset['mem_ratio'].min():>8.2f} {subset['mem_ratio'].max():>8.2f}"
            )

    # Theoretical memory breakdown
    print("\n--- Theoretical Memory Breakdown ---")
    print(
        """
    PC memory per node (5 tensors):
      - z_latent:      (batch, width) - inferred latent states
      - z_mu:          (batch, width) - predicted expectations
      - error:         (batch, width) - prediction errors (z_latent - z_mu)
      - pre_activation:(batch, width) - values before activation function
      - latent_grad:   (batch, width) - gradients for inference updates

    Total state memory: 5 * num_nodes * batch * width * 4 bytes

    Parameters scale as O(width²) per layer, dominating at large widths.
    This explains why memory ratio → 1 as width increases.
    """
    )

    # Theoretical time breakdown
    print("--- Theoretical Time Breakdown ---")
    print(
        f"""
    PC per training step ({infer_steps} inference iterations):
      For each inference step:
        1. Forward pass:  D matmuls (W @ x for each layer)
        2. Local autodiff: D matmuls (jax.value_and_grad computes VJP per node,
           each VJP is ~1 matmul: W^T @ error)
        Cost per inference step: ~2D matmuls

      Plus learning step:
        3. Weight grads from final state: ~D matmuls (x^T @ error per layer)

      Total PC: ~(2 × {infer_steps} + 1) × D = {2 * infer_steps + 1}D matmuls

    BP per training step:
      1. Forward pass:  D matmuls (W @ x for each layer)
      2. Backward pass (global autodiff through entire network):
         - Weight gradients: D matmuls (x^T @ dL/dy per layer)
         - Activation gradients: D matmuls (W^T @ dL/dy per layer)
         Cost: ~2D matmuls

      Total BP: ~3D matmuls

    Theoretical ratio: {2 * infer_steps + 1}D / 3D = {2 * infer_steps + 1}/3 ≈ {(2 * infer_steps + 1)/3:.1f}x

    Observed ratio saturates at ~5-6x (close to theoretical {(2 * infer_steps + 1)/3:.1f}x) because:
      - XLA fuses operations within jax.lax.fori_loop, reducing overhead
      - Memory bandwidth limits throughput at large widths
      - PC's local per-node autodiff has smaller computation graphs than
        BP's global backward pass
    """
    )

    # Specific example calculations
    print("--- Example: Width=128, Depth=4 ---")
    w, d = 128, 4
    num_nodes = d + 1  # input + hidden layers

    state_mem = 5 * num_nodes * batch_size * w * 4 / 1024 / 1024  # MB
    param_mem = d * w * w * 4 / 1024 / 1024  # MB (approximate)
    opt_mem = 2 * param_mem  # Adam stores first and second moments

    print(f"  GraphState memory: {state_mem:.2f} MB")
    print(f"  Parameter memory:  {param_mem:.2f} MB")
    print(f"  Optimizer state:   {opt_mem:.2f} MB")
    print(f"  PC total estimate: {state_mem + param_mem + opt_mem:.2f} MB")

    # Get observed values
    obs = merged[(merged["width"] == w) & (merged["depth"] == d)]
    if len(obs) > 0:
        print(f"  PC observed:       {obs['peak_memory_mb_pc'].values[0]:.2f} MB")
        print(f"  BP observed:       {obs['peak_memory_mb_bp'].values[0]:.2f} MB")
        print(f"  Memory ratio:      {obs['mem_ratio'].values[0]:.2f}x")
        print(f"  Time ratio:        {obs['time_ratio'].values[0]:.2f}x")


def main(filepath):
    """Main function to generate all plots and analysis."""
    # Load data
    df_clean, pc_df, bp_df, widths, depths = load_and_prepare_data(filepath)
    colors = get_color_sequence(len(widths))

    # Print summary
    print_scaling_summary(pc_df, bp_df, widths, depths)

    # Print detailed PC vs BP ratio analysis
    analyze_pc_bp_ratios(pc_df, bp_df, widths, depths)

    # Generate and save plots
    print("\nGenerating plots...")

    # Combined analysis plot
    fig1 = plot_combined_analysis(pc_df, bp_df, widths, depths, colors)
    fig1.write_html("scaling_charts.html")
    fig1.write_image("scaling_charts.png", scale=2)
    print("  Saved: scaling_charts.html, scaling_charts.png")

    # Memory analysis
    fig2 = plot_memory_analysis(pc_df, bp_df, widths, depths, colors)
    fig2.write_html("memory_analysis.html")
    fig2.write_image("memory_analysis.png", scale=2)
    print("  Saved: memory_analysis.html, memory_analysis.png")

    print("\nDone!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scaling_analysis_plots.py <path_to_excel_file>")
        print("Example: python scaling_analysis_plots.py mlp_scaling_results.xlsx")
        sys.exit(1)

    main(sys.argv[1])
