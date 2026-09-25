"""Visualisation utilities used by the optimal-control examples."""

from __future__ import annotations

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


COLORS = {
    "background": "white",
    "ink": "black",
    "muted": "0.40",
    "rail": "0.50",
    "cart": "C0",
    "cart_edge": "C0",
    "force": "C3",
    "link_1": "C1",
    "link_2": "C2",
    "mass_1": "black",
    "mass_2": "black",
    "trace": "0.5",
}


def double_cartpole_trajectory(
    x,
    tend,
    dt,
    segparms,
    pause=0.0,
    *,
    control=None,
    colors=None,
    width=None,
    height=None,
    show=True,
    repeat=True,
    hud=True,
    static=False,
):
    """Visualize a double cart-pole trajectory.

    Parameters
    ----------
    x : array-like, shape (6, n_frames)
        Rows contain ``x_c, theta_1, theta_2`` and their velocities.
    tend : float
        Duration of the optimized trajectory in seconds.
    dt : float
        Sampling interval between the supplied states.
    segparms : mapping
        Must contain ``l1`` and ``l2``; masses are used to scale the markers.
    pause : float, optional
        Number of seconds for which the final pose remains visible.
    control : array-like, optional
        Horizontal cart force. Supply one value per state or per interval.
    colors : mapping, optional
        Colour palette. Missing entries are taken from ``COLORS``.
    width, height : float, optional
        Figure size in inches. Set either width or height; the other dimension
        is inferred from the trajectory bounds. If neither is set, width is
        10 inches. The supplied dimension is respected exactly.
    show : bool, optional
        Call ``plt.show()`` before returning.
    repeat : bool, optional
        Repeat the animation after the last frame. Enabled by default.
    hud : bool, optional
        Show the time and force badge.
    static : bool, optional
        Render the first frame as a normal Matplotlib figure without creating
        a ``FuncAnimation``. Returns ``(figure, axes)`` in this mode.

    Returns
    -------
    matplotlib.animation.FuncAnimation or tuple
        Animation by default; ``(figure, axes)`` when ``static=True``.
    """

    states = np.asarray(x, dtype=float)
    duration = float(tend)
    parameters = segparms
    palette = COLORS | ({} if colors is None else dict(colors))
    if states.ndim != 2 or states.shape[0] < 3:
        raise ValueError("states must have shape (at least 3, n_frames)")
    if states.shape[1] < 2:
        raise ValueError("at least two trajectory frames are required")
    if dt <= 0:
        raise ValueError("dt must be positive")

    l1 = float(parameters["l1"])
    l2 = float(parameters["l2"])
    m1 = float(parameters.get("m1", 1.0))
    m2 = float(parameters.get("m2", 1.0))

    original_frame_count = states.shape[1]

    if control is None:
        force = np.zeros(original_frame_count)
        show_force = False
    else:
        force = np.asarray(control, dtype=float).reshape(-1)
        if force.size == original_frame_count - 1:
            force = np.append(force, force[-1])
        elif force.size != original_frame_count:
            raise ValueError(
                "control must contain one value per state or per interval"
            )
        show_force = True

    # Repeat the final state and force to create a deliberate end pause.
    extra_frames = max(0, int(round(float(pause) / dt)))
    if extra_frames:
        final_state = np.repeat(states[:, -1:], extra_frames, axis=1)
        states = np.hstack((states, final_state))
        force = np.append(force, np.repeat(force[-1], extra_frames))

    x_c = states[0]
    theta1 = states[1]
    theta2 = states[2]

    x1 = x_c + l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta2)
    y2 = y1 + l2 * np.sin(theta2)

    cart_width = 0.48
    cart_height = 0.22
    wheel_radius = 0.075
    rail_y = -cart_height / 2 - 2.1 * wheel_radius

    # Leave room for the force arrow when the cart reaches a trajectory bound.
    horizontal_margin = 1.25
    x_min = min(np.min(x_c), np.min(x1), np.min(x2)) - horizontal_margin
    x_max = max(np.max(x_c), np.max(x1), np.max(x2)) + horizontal_margin
    y_min = min(rail_y - 0.35, np.min(y1), np.min(y2)) - 0.25
    y_max = max(cart_height, np.max(y1), np.max(y2)) + 0.35

    data_width = max(x_max - x_min, 1e-9)
    data_height = max(y_max - y_min, 1e-9)
    data_aspect = data_width / data_height
    # Preserve enough canvas for the badges and axis labels even when
    # the physical trajectory itself is extremely narrow or wide.
    figure_aspect = float(np.clip(data_aspect, 1.35, 2.0))

    # Match the data limits to the canvas aspect ratio. Without this padding,
    # ``aspect="equal"`` shrinks the axes box when the horizontal trajectory
    # range is small, leaving unused space at either side of the figure.
    if data_aspect < figure_aspect:
        extra_width = data_height * figure_aspect - data_width
        x_min -= extra_width / 2
        x_max += extra_width / 2
    elif data_aspect > figure_aspect:
        extra_height = data_width / figure_aspect - data_height
        y_min -= extra_height / 2
        y_max += extra_height / 2

    if width is None and height is None:
        width = 10.0
    if width is not None and width <= 0:
        raise ValueError("width must be positive")
    if height is not None and height <= 0:
        raise ValueError("height must be positive")
    if width is None:
        width = float(height) * figure_aspect
    elif height is None:
        height = float(width) / figure_aspect

    fig, ax = plt.subplots(
        figsize=(float(width), float(height)),
        constrained_layout=True,
    )
    # Transparent canvas: it appears white in a normal window and blends into
    # the background when embedded in a webpage or exported with transparency.
    fig.patch.set_facecolor(palette["background"])
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"cart position $x_c$ [m]")
    ax.set_ylabel("height [m]")
    ax.tick_params(colors=palette["muted"], length=0)
    ax.grid(axis="y", color=palette["ink"], alpha=0.07, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # A single neutral rail provides a clean reference for cart motion.
    ax.axhline(rail_y, color=palette["rail"], linewidth=3.0, zorder=1)

    trace, = ax.plot(
        [], [],
        color=palette["trace"],
        linewidth=1.8,
        alpha=0.6,
        linestyle=(0, (2, 3)),
        zorder=2,
    )
    link1, = ax.plot(
        [], [],
        color=palette["link_1"],
        linewidth=5,
        solid_capstyle="round",
        zorder=5,
    )
    link2, = ax.plot(
        [], [],
        color=palette["link_2"],
        linewidth=5,
        solid_capstyle="round",
        zorder=5,
    )

    mass_scale = 190
    mass1 = ax.scatter(
        [], [], s=mass_scale * np.sqrt(max(m1, 0.05)),
        color=palette["mass_1"], edgecolor="white", linewidth=1.8, zorder=7,
    )
    mass2 = ax.scatter(
        [], [], s=mass_scale * np.sqrt(max(m2, 0.05)),
        color=palette["mass_2"], edgecolor="white", linewidth=1.8, zorder=7,
    )
    pivot = ax.scatter(
        [], [], s=38, color=palette["ink"], edgecolor="white",
        linewidth=1.0, zorder=8,
    )

    cart = FancyBboxPatch(
        (0, 0),
        cart_width,
        cart_height,
        boxstyle="round,pad=0.025,rounding_size=0.055",
        linewidth=2,
        edgecolor=palette["cart_edge"],
        facecolor=palette["cart"],
        zorder=6,
    )
    ax.add_patch(cart)

    wheel_left = Circle((0, 0), wheel_radius, color=palette["ink"], zorder=7)
    wheel_right = Circle((0, 0), wheel_radius, color=palette["ink"], zorder=7)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)

    force_arrow = FancyArrowPatch(
        (0, 0),
        (0, 0),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.5,
        color=palette["force"],
        # Above the cart body, but below the black center-of-mass marker.
        zorder=7.5,
    )
    force_arrow.set_visible(show_force)
    ax.add_patch(force_arrow)

    time_text = ax.text(
        0.98, 0.90, "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=palette["ink"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
    )
    force_text = ax.text(
        0.98, 0.82, "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=palette["force"],
    )
    force_text.set_visible(show_force)

    if not hud:
        for artist in (
            time_text,
            force_text,
            trace,
        ):
            artist.set_visible(False)

    max_force = max(float(np.max(np.abs(force))), np.finfo(float).eps)
    max_arrow_length = (x_max - x_min) / 3

    def update(frame):
        cart_x = x_c[frame]
        cart.set_bounds(
            cart_x - cart_width / 2,
            -cart_height / 2,
            cart_width,
            cart_height,
        )
        wheel_y = -cart_height / 2 - wheel_radius
        wheel_left.center = (cart_x - 0.145, wheel_y)
        wheel_right.center = (cart_x + 0.145, wheel_y)

        link1.set_data([cart_x, x1[frame]], [0, y1[frame]])
        link2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        mass1.set_offsets([[x1[frame], y1[frame]]])
        mass2.set_offsets([[x2[frame], y2[frame]]])
        pivot.set_offsets([[cart_x, 0]])
        trace.set_data(x2[: frame + 1], y2[: frame + 1])

        current_force = force[frame]
        arrow_length = max_arrow_length * current_force / max_force
        if arrow_length > 0:
            arrow_length = min(arrow_length, x_max - cart_x - 0.05)
        elif arrow_length < 0:
            arrow_length = -min(-arrow_length, cart_x - x_min - 0.05)
        # Apply the force at the marked cart center of mass (x_c, 0).
        arrow_y = 0.0
        force_arrow.set_visible(show_force and abs(current_force) > 1e-8)
        force_arrow.set_positions(
            (cart_x, arrow_y),
            (cart_x + arrow_length, arrow_y),
        )
        force_text.set_text(rf"$F_x = {current_force:+5.1f}\,\mathrm{{N}}$")

        elapsed = min(frame * dt, float(duration))
        time_text.set_text(f"t = {elapsed:4.2f} s")

        return (
            cart,
            wheel_left,
            wheel_right,
            link1,
            link2,
            mass1,
            mass2,
            pivot,
            trace,
            time_text,
            force_arrow,
            force_text,
        )

    if static:
        update(0)
        if show:
            plt.show()
        return fig, ax

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=states.shape[1],
        interval=1000 * dt,
        blit=True,
        repeat=repeat,
        repeat_delay=700,
    )

    # Draw the first pose immediately, which also makes static notebook output
    # and screenshots useful before the animation starts.
    update(0)
    if show:
        plt.show()

    return ani


def save_animation_gif(ani, filename, *, fps, dpi=300):
    """Save a Matplotlib animation as a looping GIF on a white canvas."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    fig = ani._fig
    
    fig.patch.set_alpha(1)
    fig.patch.set_facecolor("white")
    writer = animation.PillowWriter(
        fps=fps,
        metadata={"title": "Double cart-pole swing-up"},
    )
    ani.save(filename, writer=writer, dpi=dpi)
