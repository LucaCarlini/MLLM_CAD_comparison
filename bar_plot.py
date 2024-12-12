import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_grouped_bar_plot_with_min_max_and_p_values(
    data,
    min_max_values,
    p_values,
    group_labels,
    bar_labels,
    title="Grouped Bar Plot",
    xlabel=None,
    ylabel="Values (%)",
    save_path=None,
    show_plot=False,
    color_choice="blue",
    offsets=None,
    apply_bonferroni_correction=False,
    bonferroni_div=4
):
    """
    Creates a bar plot with grouped columns, min-max error bars, and significant p-values,
    using different colors for each bar to avoid color repetition.

    Parameters:
        data (list of list): 2D list where each sublist represents a group of bars.
                             Each group has the same number of bars.
        min_max_values (list of list): 2D list of [min, max] values for each bar.
        p_values (list of list): 2D list where each sublist contains p-values for pairs of bars.
                                 The length of each sublist should match C(n,2) for n bars.
                                 The pairs should be in ascending order of indices:
                                 (0 vs 1), (0 vs 2), ..., (0 vs n-1), (1 vs 2), (1 vs 3), ..., (n-2 vs n-1).
                                 Only pairs with p<significance_threshold are represented.
        group_labels (list of str): Labels for each group.
        bar_labels (list of str): Labels for each bar.
        title (str), xlabel (str), ylabel (str): Plot labeling.
        save_path (str): Where to save the figure (optional).
        show_plot (bool): Whether to show the plot.
        color_choice (str): Color scheme ("blue", "green", "orange", "grayscale").
        offsets (list of float): Offsets for lines connecting the first and last bar.
                                 Must match the number of groups. If None, zero offsets by default.
        apply_bonferroni_correction (bool): Whether to apply Bonferroni correction to p-values.
        bonferroni_div (int): The division factor for the Bonferroni correction. The significance
                              threshold (0.05) is adjusted to (0.05/bonferroni_div).
    """

    # Determine significance threshold
    significance_threshold = 0.05 / bonferroni_div if apply_bonferroni_correction else 0.05

    num_groups = len(data)
    max_bars_per_group = max(len(group) for group in data)
    indices = np.arange(num_groups)
    bar_width = 0.4 / max_bars_per_group

    # If no offsets are provided, default to zero for all groups
    if offsets is None:
        offsets = [0]*num_groups
    else:
        if len(offsets) != num_groups:
            raise ValueError("offsets must have the same length as the number of groups.")

    # Convert data to percentages
    data = [[val * 100 for val in group] for group in data]
    min_max_values = [
        [[min_val * 100, max_val * 100] for min_val, max_val in group]
        for group in min_max_values
    ]

    # Choose colors
    if color_choice.lower() == "green":
        colors = ["#006400", "#32CD32", "#98FB98", "#ADFF2F", "#7FFF00"]
    elif color_choice.lower() == "orange":
        colors = ["#FF8C00", "#FFA500", "#FFD700", "#FFB347", "#FFCC33"]
    elif color_choice.lower() == "grayscale":
        colors = ["#333333", "#7F7F7F", "#AAAAAA", "#C0C0C0", "#E0E0E0"]
    else:
        colors = ["#0B3D91", "#1E90FF", "#87CEFA", "#00BFFF", "#ADD8E6"]

    plt.figure(figsize=(12, 8))
    plt.grid(axis="y", linestyle="--", color="gray", alpha=0.7)

    # Plot bars
    for i in range(max_bars_per_group):
        for j, group in enumerate(data):
            if i < len(group):
                color_idx = i % len(colors)
                data_value = group[i]
                min_val = min_max_values[j][i][0]
                max_val = round(min_max_values[j][i][1], 1)

                lower_error = data_value - min_val
                upper_error = max_val - data_value
                y_errors = [[lower_error], [upper_error]]

                # Position bars
                if max_bars_per_group == 2:
                    # Center them if only two bars
                    x_position = indices[j] + i * bar_width - (bar_width / 2)
                else:
                    x_position = indices[j] + i * bar_width

                plt.bar(
                    x_position,
                    data_value,
                    width=bar_width,
                    color=colors[color_idx],
                    edgecolor='black',
                    label=bar_labels[i] if j == 0 else "",
                    yerr=y_errors,
                    capsize=5,
                )

    # Function to map pair index to (i,j)
    def pair_index_to_ij(k, n):
        count = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if count == k:
                    return i, j
                count += 1

    # Draw significance lines
    for group_idx, group_p_values in enumerate(p_values):
        max_bar_height = max(data[group_idx]) if data[group_idx] else 0
        line_start_y = max_bar_height + 5

        # Collect significant pairs
        significant_pairs = []
        for k, p_val in enumerate(group_p_values):
            if p_val < significance_threshold:
                i, j = pair_index_to_ij(k, len(data[group_idx]))
                significant_pairs.append((i, j, p_val))

        # Sort pairs by their "span"
        significant_pairs.sort(key=lambda x: (x[1]-x[0]), reverse=True)

        # Assign each pair a vertical level
        current_y = line_start_y
        line_increment = 5
        for (i, j, p_val) in significant_pairs:
            # Compute line positions
            if max_bars_per_group == 2:
                x1 = indices[group_idx] - (bar_width / 2)
                x2 = indices[group_idx] + (bar_width / 2)
            else:
                x1 = indices[group_idx] + i * bar_width
                x2 = indices[group_idx] + j * bar_width

            # Apply offset only for first vs last bar if needed
            if i == 0 and j == (max_bars_per_group-1):
                current_y += offsets[group_idx]

            # Draw line
            plt.plot([x1, x2], [current_y, current_y], "k-", lw=1.5)
            plt.text(
                (x1 + x2) / 2,
                current_y - 1,
                "*",
                ha="center",
                va="bottom",
                color="black",
                fontsize=16,
            )
            current_y += line_increment

    # Final formatting
    plt.ylim(0, None)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=20, fontweight="bold")
    plt.ylabel(ylabel, fontsize=20, fontweight="bold")
    plt.title(title, fontsize=20, y=1.03, fontweight="bold")
    plt.xticks(
        indices + bar_width * (max_bars_per_group - 1) / 2,
        group_labels,
        y=-0.02,
        fontsize=20,
        ha="center",
    )
    plt.yticks(fontsize=20)
    plt.legend(title="Models", fontsize=16, title_fontsize=20, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()
