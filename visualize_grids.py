"""Visualize grids from JSON data."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ARC color palette (standard colors used in ARC challenges)
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: light blue
    "#870C25",  # 9: dark red
]

cmap = ListedColormap(ARC_COLORS)


def parse_tuple(data):
    """Parse the __tuple__ format into a Python list/numpy array."""
    if isinstance(data, dict) and "__tuple__" in data:
        tuple_data = data["__tuple__"]
        if isinstance(tuple_data, list):
            # Check if it's a 2D grid (list of dicts)
            if tuple_data and isinstance(tuple_data[0], dict):
                return np.array([parse_tuple(row) for row in tuple_data])
            # Check if it's a 1D row (list of numbers)
            else:
                return np.array(tuple_data)
        else:
            return np.array([])
    return data


def parse_io_data(json_str):
    """Parse JSON string containing I/O data."""
    try:
        data = json.loads(json_str)
        if "I" in data:
            return parse_tuple(data["I"])
        elif "__tuple__" in data:
            return parse_tuple(data)
        return None
    except Exception:
        return None


def plot_grid(ax, grid, title):
    """Plot a single grid with ARC colors."""
    if grid is None or (isinstance(grid, np.ndarray) and grid.size == 0):
        ax.text(0.5, 0.5, "Empty Grid", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return

    height, width = grid.shape

    # Display the grid
    _ = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")

    # Add grid lines
    for i in range(height + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1)

    # Add cell values as text
    for i in range(height):
        for j in range(width):
            text_color = "white" if grid[i, j] in [0, 1, 9] else "black"
            ax.text(
                j,
                i,
                str(int(grid[i, j])),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")


def visualize_example(task_name, shot_idx, input_grid, output_grid):
    """Visualize a single input-output pair."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"{task_name}\nExample {shot_idx + 1}", fontsize=14, fontweight="bold")

    plot_grid(axes[0], input_grid, "Input")
    plot_grid(axes[1], output_grid, "Output")

    plt.tight_layout()
    return fig


def visualize_all_shots(task):
    """Visualize all shots for a given task."""
    task_name = task["name"]
    shots = task["shots"]

    num_shots = len(shots)
    if num_shots == 0:
        print(f"No shots available for task: {task_name}")
        return

    # Create a figure for each shot
    for idx, shot in enumerate(shots):
        input_grid = parse_io_data(shot["inputs"])
        output_grid = parse_io_data(shot["output"])

        _ = visualize_example(task_name, idx, input_grid, output_grid)
        plt.show()


def browse_tasks(file_path):
    """Interactive browser for tasks."""
    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} tasks from {file_path}")
    print("\nAvailable tasks:")
    for idx, task in enumerate(data):
        num_shots = len(task["shots"])
        print(f"{idx}: {task['name']} ({num_shots} examples)")

    while True:
        try:
            choice = input("\nEnter task number to visualize (or 'q' to quit, 'all' to see all): ")

            if choice.lower() == "q":
                break
            elif choice.lower() == "all":
                for task in data:
                    visualize_all_shots(task)
            else:
                task_idx = int(choice)
                if 0 <= task_idx < len(data):
                    visualize_all_shots(data[task_idx])
                else:
                    print(
                        f"Invalid task number. Please enter a number between 0 and {len(data) - 1}"
                    )
        except ValueError:
            print("Invalid input. Please enter a number or 'q'/'all'")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def visualize_task_by_index(file_path, task_idx=0):
    """Visualize a specific task by index."""
    with open(file_path, "r") as f:
        data = json.load(f)

    if 0 <= task_idx < len(data):
        visualize_all_shots(data[task_idx])
    else:
        print(f"Task index {task_idx} out of range. Available: 0-{len(data) - 1}")


if __name__ == "__main__":
    import sys

    file_path = "eval_split.json"

    if len(sys.argv) > 1:
        # If a task index is provided as command line argument
        try:
            task_idx = int(sys.argv[1])
            visualize_task_by_index(file_path, task_idx)
        except ValueError:
            print("Usage: python visualize_grids.py [task_index]")
    else:
        # Interactive mode
        browse_tasks(file_path)
