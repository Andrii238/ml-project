"""Matplotlib visualization for gridworld value functions, policies, and trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrow, Rectangle

from task1_gridworld.environment import ACTION_DELTAS, GridWorld


def plot_value_and_policy(
    env: GridWorld,
    V: np.ndarray,
    pi: np.ndarray | None = None,
    ax: Axes | None = None,
    title: str = "",
    show_values: bool = True,
) -> Axes:
    """Render a gridworld with value numbers and (optional) policy arrows.

    Uses the unstunned state's V(cell) for each cell. Traps are red-outlined,
    walls are gray-filled, start is a green outline, goal is a gold outline.
    Policy arrows show actions with non-zero probability under `pi`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(env.cols * 0.9, env.rows * 0.9))

    # Background grid
    for r in range(env.rows):
        for c in range(env.cols):
            cell = (r, c)
            if cell in env.walls:
                ax.add_patch(Rectangle((c, env.rows - r - 1), 1, 1,
                                       facecolor="#888", edgecolor="black"))
            else:
                ax.add_patch(Rectangle((c, env.rows - r - 1), 1, 1,
                                       facecolor="white", edgecolor="black"))

    # Highlights
    for cell in env.traps:
        r, c = cell
        ax.add_patch(Rectangle((c, env.rows - r - 1), 1, 1,
                               facecolor="#ffb0b0", edgecolor="red", linewidth=1.5, zorder=1))
    sr, sc = env.start
    ax.add_patch(Rectangle((sc + 0.02, env.rows - sr - 1 + 0.02), 0.96, 0.96,
                           facecolor="none", edgecolor="green", linewidth=2.5))
    gr, gc = env.goal
    ax.add_patch(Rectangle((gc + 0.02, env.rows - gr - 1 + 0.02), 0.96, 0.96,
                           facecolor="none", edgecolor="gold", linewidth=2.5))

    # Value text
    for r in range(env.rows):
        for c in range(env.cols):
            cell = (r, c)
            if cell in env.walls:
                continue
            s = env.cell_to_state(cell)
            v = V[s]
            if show_values:
                ax.text(c + 0.5, env.rows - r - 1 + 0.7, f"{v:+.1f}",
                        ha="center", va="center", fontsize=8, color="black")
            if cell == env.goal:
                ax.text(c + 0.5, env.rows - r - 1 + 0.3, "GOAL",
                        ha="center", va="center", fontsize=7, color="darkgoldenrod")
                continue
            if cell == env.start:
                ax.text(c + 0.5, env.rows - r - 1 + 0.15, "START",
                        ha="center", va="center", fontsize=7, color="darkgreen")

    # Policy arrows
    if pi is not None:
        for r in range(env.rows):
            for c in range(env.cols):
                cell = (r, c)
                if cell in env.walls or cell == env.goal:
                    continue
                s = env.cell_to_state(cell)
                for a, p in enumerate(pi[s]):
                    if p < 1e-6:
                        continue
                    dr, dc = ACTION_DELTAS[a]
                    # arrow length scaled by probability
                    length = 0.28 * (0.5 + 0.5 * p)
                    ax.add_patch(FancyArrow(
                        c + 0.5, env.rows - r - 1 + 0.35,
                        dc * length, -dr * length,
                        width=0.02, head_width=0.11, head_length=0.09,
                        length_includes_head=True, color="steelblue", alpha=0.85,
                    ))

    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_iteration_grid(env: GridWorld, history: list[dict],
                        max_cols: int = 4) -> plt.Figure:
    """Render one panel per policy-iteration step (initial + each iter)."""
    n = len(history)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * env.cols * 0.9,
                                                   rows * env.rows * 0.9))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, snap in enumerate(history):
        title = "π₀ (random init)" if i == 0 else f"After improvement {i}"
        plot_value_and_policy(env, snap["V"], snap["pi"], ax=axes_flat[i], title=title)
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.tight_layout()
    return fig


def _draw_grid_background(env: GridWorld, ax: Axes) -> None:
    """Draw the static grid: walls, traps, start, goal outlines. No agent."""
    for r in range(env.rows):
        for c in range(env.cols):
            cell = (r, c)
            if cell in env.walls:
                face = "#888"
            elif cell in env.traps:
                face = "#ffb0b0"
            else:
                face = "white"
            ax.add_patch(Rectangle((c, env.rows - r - 1), 1, 1,
                                   facecolor=face, edgecolor="black", zorder=1))
    sr, sc = env.start
    ax.add_patch(Rectangle((sc + 0.02, env.rows - sr - 1 + 0.02), 0.96, 0.96,
                           facecolor="none", edgecolor="green", linewidth=2.5))
    gr, gc = env.goal
    ax.add_patch(Rectangle((gc + 0.02, env.rows - gr - 1 + 0.02), 0.96, 0.96,
                           facecolor="none", edgecolor="gold", linewidth=2.5))
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def animate_rollout(
    env: GridWorld,
    trajectory: dict,
    policy: np.ndarray | None = None,
    iter_label: str = "",
    interval_ms: int = 350,
) -> animation.FuncAnimation:
    """Animate a rollout on the gridworld.

    If `policy` is provided, greedy-action arrows are drawn at every non-goal,
    non-wall cell so it's immediately visible which direction the policy
    picks at each state — especially useful for spotting "stuck" cells where
    the arrow points into a wall/boundary.
    """
    states = trajectory["states"]
    rewards = trajectory["rewards"]
    n_frames = len(states)

    fig, ax = plt.subplots(figsize=(env.cols * 0.7 + 0.5, env.rows * 0.7 + 1.0))
    plt.close(fig)

    cum_reward = [0.0]
    for t in range(1, n_frames):
        cum_reward.append(cum_reward[-1] + rewards[t - 1])

    def draw(t: int):
        ax.clear()
        _draw_grid_background(env, ax)

        pts = [env.state_to_cell(s) for s in states[: t + 1]]
        if len(pts) > 1:
            xs = [c + 0.5 for (_r, c) in pts]
            ys = [env.rows - r - 1 + 0.5 for (r, _c) in pts]
            ax.plot(xs, ys, color="darkorange", alpha=0.6, linewidth=2.0, zorder=3)

        r, c = env.state_to_cell(states[t])
        ax.add_patch(Circle((c + 0.5, env.rows - r - 1 + 0.5), 0.28,
                            facecolor="crimson", edgecolor="black",
                            linewidth=1.5, zorder=4))

        # Arrows drawn AFTER the ball so they're visible on top.
        if policy is not None:
            for r_ in range(env.rows):
                for c_ in range(env.cols):
                    cell = (r_, c_)
                    if cell in env.walls or cell == env.goal:
                        continue
                    s = env.cell_to_state(cell)
                    for a, p in enumerate(policy[s]):
                        if p < 1e-6:
                            continue
                        dr, dc = ACTION_DELTAS[a]
                        length = 0.32
                        ax.add_patch(FancyArrow(
                            c_ + 0.5, env.rows - r_ - 1 + 0.5,
                            dc * length, -dr * length,
                            width=0.03, head_width=0.15, head_length=0.11,
                            length_includes_head=True,
                            color="black", alpha=0.85, zorder=6,
                        ))

        reached = env.is_terminal(states[t])
        status = "reached goal" if reached else f"step {t}/{n_frames - 1}"
        cum_trap = sum(1 for s in states[1 : t + 1] if env.state_to_cell(s) in env.traps)
        title = (
            f"{iter_label} — {status}\n"
            f"undiscounted return: {cum_reward[t]:+.1f}   "
            f"trap hits: {cum_trap}"
        )
        ax.set_title(title, fontsize=10)
        return []

    anim = animation.FuncAnimation(
        fig, draw, frames=n_frames, interval=interval_ms, blit=False, repeat=True,
    )
    return anim


def summarize_rollout(trajectory: dict) -> str:
    """Compact one-line summary string of a rollout's stats."""
    reached = "reached goal" if trajectory["reached_goal"] else "did NOT reach goal"
    return (
        f"steps={trajectory['steps']}  "
        f"undiscounted_return={trajectory['undiscounted_return']:+.1f}  "
        f"stunned_steps={trajectory['stunned_steps']}  "
        f"[{reached}]"
    )
