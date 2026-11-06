"""Interactive Plotly chart builders for the Simulator and Demo pages.

Mirrors the plots in src/visualization.py but as interactive Plotly
figures instead of static matplotlib ones. src/visualization.py (used
by main.py) is untouched - these are additional, app-only renderers.
"""

import networkx as nx
import plotly.graph_objects as go

ACCENT = "#2563EB"
BASELINE_COLOR = "#DC2626"
BEST_COLOR = "#16A34A"
OTHER_COLOR = "#2563EB"
GRID_COLOR = "#E2E8F0"


def plotly_graph(graph, entry_points=None, target=None, title="System Attack Graph"):
    """Node-link plot of `graph`, colored by criticality, hover showing
    each node's vulnerability/criticality, with the entry point(s) and
    target node marked by distinct marker symbols."""
    entry_points = entry_points or {}
    pos = nx.spring_layout(graph, seed=7, k=1.8, iterations=200)

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.3, color="#94A3B8"),
        hoverinfo="none", showlegend=False,
    )

    node_x, node_y, node_color, node_text = [], [], [], []
    node_symbol, node_line_color, node_line_width = [], [], []
    for n in graph.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        attrs = graph.nodes[n]
        vuln = attrs.get("vulnerability")
        crit = attrs.get("criticality")
        node_color.append(crit)

        role = []
        if n in entry_points:
            role.append("Entry point")
        if n == target:
            role.append("Target")
        role_text = f"<br>{' / '.join(role)}" if role else ""
        node_text.append(
            f"<b>{n}</b>{role_text}<br>Vulnerability: {vuln:.2f}<br>Criticality: {crit:.2f}"
        )

        if n == target:
            node_symbol.append("square")
            node_line_color.append(BASELINE_COLOR)
            node_line_width.append(3)
        elif n in entry_points:
            node_symbol.append("diamond")
            node_line_color.append(ACCENT)
            node_line_width.append(3)
        else:
            node_symbol.append("circle")
            node_line_color.append("#1E293B")
            node_line_width.append(1)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=list(graph.nodes()), textposition="top center", textfont=dict(size=10),
        hovertext=node_text, hoverinfo="text",
        marker=dict(
            size=30, color=node_color, colorscale="RdYlGn_r", cmin=0, cmax=1,
            showscale=True, colorbar=dict(title="Criticality"),
            symbol=node_symbol, line=dict(color=node_line_color, width=node_line_width),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="diamond", size=12, color=ACCENT, line=dict(color=ACCENT, width=2)),
        name="Entry point",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=12, color=BASELINE_COLOR, line=dict(color=BASELINE_COLOR, width=2)),
        name="Target",
    ))

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=60, b=10),
        height=550,
        plot_bgcolor="white",
    )
    return fig


def plotly_defense_comparison(entries, target_node, title="Defense Resource Allocation Comparison"):
    """Bar chart with 95% CI error bars over `entries`, a list of dicts
    with 'label', 'probability', 'ci_low', 'ci_high', and 'kind' in
    {'baseline', 'best', 'other'} (controls bar color)."""
    colors = {"baseline": BASELINE_COLOR, "best": BEST_COLOR, "other": OTHER_COLOR}
    labels = [e["label"] for e in entries]
    values = [e["probability"] for e in entries]
    err_plus = [e["ci_high"] - e["probability"] for e in entries]
    err_minus = [e["probability"] - e["ci_low"] for e in entries]
    bar_colors = [colors[e["kind"]] for e in entries]
    hover = [
        f"{e['label']}<br>P(fail)={e['probability']:.4f}<br>95% CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]"
        for e in entries
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=bar_colors,
        error_y=dict(
            type="data", array=err_plus, arrayminus=err_minus,
            visible=True, color="#334155", thickness=1.2,
        ),
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title=f"{title}<br><sup>Error bars = 95% Wilson CI. Red = baseline, Green = best option.</sup>",
        yaxis_title=f"P({target_node} fails)",
        xaxis_tickangle=-45,
        height=480,
        margin=dict(l=40, r=20, t=70, b=140),
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor=GRID_COLOR, range=[0, 1])
    return fig


def plotly_failure_over_time(series, target_node, title="Failure probability over time"):
    """Line chart with CI bands over `series`, a list of dicts with
    'label', 'color', 'band_color', 'thresholds', 'p', 'ci_low', 'ci_high'
    (parallel lists over the same thresholds)."""
    fig = go.Figure()
    for s in series:
        thresholds = s["thresholds"]
        fig.add_trace(go.Scatter(
            x=thresholds + thresholds[::-1],
            y=s["ci_high"] + s["ci_low"][::-1],
            fill="toself", fillcolor=s["band_color"],
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=s["p"], mode="lines+markers", name=s["label"],
            line=dict(color=s["color"], width=2.5), marker=dict(size=7),
            hovertemplate="t=%{x}h<br>P(fail)=%{y:.4f}<extra>" + s["label"] + "</extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time horizon (hours)",
        yaxis_title=f"P({target_node} fails by t)",
        height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(gridcolor=GRID_COLOR, range=[0, 1])
    fig.update_xaxes(gridcolor=GRID_COLOR)
    return fig
