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
    color_choice="blue"
):
    """
    Creates a bar plot with grouped columns, min-max error bars, and significant p-values,
    using different colors for each bar to avoid color repetition.

    Parameters:
        data (list of list): A 2D list where each sublist represents a group of bars.
                             Each group should have the same number of bars.
        min_max_values (list of list): A 2D list containing min and max values for each bar.
                                       Each sublist has pairs [min, max] for each bar in the group.
        p_values (list of list): A 2D list where each sublist contains the p-values between each pair of bars
                                 within a group. Only pairs with p < 0.05 are represented with a line.
        group_labels (list of str): Labels for each group of bars.
        bar_labels (list of str): Labels for each individual bar within groups.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        save_path (str): Optional path to save the plot image.
        show_plot (bool): Whether to show the plot or not. Default is True.
        color_choice (str): Color tone to use. Options are "blue", "green", "orange", or "grayscale".
    """
    num_groups = len(data)
    max_bars_per_group = max(len(group) for group in data)
    indices = np.arange(num_groups)
    bar_width = 0.4 / max_bars_per_group  # Adjust bar width to fit all bars in the group

    # Convert data to percentages for display
    data = [[val * 100 for val in group] for group in data]
    min_max_values = [
        [[min_val * 100, max_val * 100] for min_val, max_val in group]
        for group in min_max_values
    ]

    # Define distinct colors for bars
    if color_choice.lower() == "green":
        colors = ["#006400", "#32CD32", "#98FB98"]
    elif color_choice.lower() == "orange":
        colors = ["#FF8C00", "#FFA500", "#FFD700"]
    elif color_choice.lower() == "grayscale":
        colors = ["#333333", "#7F7F7F", '#F0F0F0']  # Changed third color to a darker gray
    else:  # Default to blue
        colors = ["#0B3D91", "#1E90FF", "#87CEFA"]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.grid(axis="y", linestyle="--", color="gray", alpha=0.7)  # Add horizontal grid
    for i in range(max_bars_per_group):
        for j, group in enumerate(data):
            if i < len(group):  # Check if bar exists in this group
                if len(group) == 2 and i == 1:
                    color_idx = 2
                else:
                    color_idx = i % len(colors)
                data_value = group[i]
                min_val = min_max_values[j][i][0]
                max_val = round(min_max_values[j][i][1], 1)
                
                # Calculate errors relative to data_value
                lower_error = data_value - min_val
                upper_error = max_val - data_value

                if upper_error < 0:
                    print(f"Upper error is negative: {upper_error}")
                    print('min_val:', min_val)
                    print('max_val:', max_val)
                    print('data_value:', data_value)
                    
                
                y_errors = [[lower_error], [upper_error]]  # Must be a list of lists
                
                
                plt.bar(
                    indices[j] + i * bar_width,
                    group[i],
                    width=bar_width,
                    color=colors[color_idx],
                    edgecolor='black',
                    label=bar_labels[i] if j == 0 else "",  # Label only once in the legend
                    yerr=y_errors,
                    capsize=5,  # Controls the width of the error bars
                )

    # # Add percentage values at the top of the CI bars
    # for i in range(max_bars_per_group):
    #     for j, group in enumerate(data):
    #         if i < len(group):
    #             data_value = group[i]
    #             max_val = round(min_max_values[j][i][1], 1)
    #             plt.text(
    #                 indices[j] + i * bar_width,
    #                 max_val + 1,
    #                 f"{max_val:.1f}%",
    #                 rotation=90,
    #                 ha="center",
    #                 va="bottom",
    #                 color="black",
    #                 fontsize=14,
    #             )


    

    # Add horizontal lines to represent p-values (only when p < 0.05)
    for group_idx, group_p_values in enumerate(p_values):
        for pair_idx in range(len(group_p_values)):
            if pair_idx < max_bars_per_group - 1 and group_p_values[pair_idx] < 0.05:
                # Position of adjacent bars
                x1 = indices[group_idx] + pair_idx * bar_width
                x2 = x1 + bar_width
                y = (
                    max(data[group_idx][pair_idx], data[group_idx][pair_idx + 1]) + 10
                )  # Increase height above bars

                # Draw horizontal line and add asterisk
                plt.plot([x1, x2], [y, y], "k-", lw=1.5)
                plt.text(
                    (x1 + x2) / 2,
                    y - 1,
                    "*",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=16,
                    
                )

        # Check if the p-value for first and third bars is < 0.05 and add a connecting line if true
        if len(group_p_values) > 2 and group_p_values[2] < 0.05:
            x1 = indices[group_idx]  # Position of the first bar in the group
            x3 = indices[group_idx] + 2 * bar_width  # Position of the third bar in the group
            y = max(data[group_idx][0], data[group_idx][2]) + 15  # Increase height more

            # Draw horizontal line between first and third bars and add asterisk
            plt.plot([x1, x3], [y, y], "k-", lw=1.5)
            plt.text(
                (x1 + x3) / 2,
                y - 1,
                "*",
                ha="center",
                va="bottom",
                color="black",
                fontsize=16,
            )

    # Customize plot appearance
    plt.ylim(0, 115)  # Set y-axis limits to 0-110%
    if xlabel is not None:

        plt.xlabel(xlabel, fontsize=20, fontweight="bold")
        
    plt.ylabel(ylabel, fontsize=20, fontweight="bold")
    plt.title(title, fontsize=20, y=1.03, fontweight="bold")
    plt.xticks(
        indices + bar_width * (max_bars_per_group - 1) / 2,
        group_labels,
        # move it lower
        y=-0.02,

        fontsize=20,
        #rotation=45,
        ha="center",
        #ha="right",
    )
    plt.yticks(fontsize=20)
    #set legend outside of plot on the right
    plt.legend(title="Models", fontsize=16, title_fontsize=20, loc='upper left', bbox_to_anchor=(1, 1),)
    plt.tight_layout()

    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Show plot if the flag is set to True
    if show_plot:
        plt.show()
    else:
        plt.close()

# Example usage
if __name__ == "__main__":
    # Example data
    data = [
        [0.8352337514253135, 0.8095781071835804, 0.6111744583808438],
        [0.7881231671554252, 0.7551319648093842, 0.500733137829912],
        [1.0, 1.0, 0.9974358974358974],
        [0.8352337514253135, 0.6111744583808438],
    ]
    min_max_values = [
        [
            [0.817030375885863, 0.8523078233279693],
            [0.7904063951111765, 0.8277081454896709],
            [0.5879052720788016, 0.6340710067484406],
        ],
        [
            [0.765461178120247, 0.8095365603683053],
            [0.7314102292100351, 0.7777492550801952],
            [0.47385482425537295, 0.5276082868173512],
        ],
        [
            [0.9905859272015295, 1.0],
            [0.9905859272015295, 1.0],
            [0.9857971872149763, 0.9999350846507224],
        ],
        [
            [0.817030375885863, 0.8523078233279693],
            [0.7904063951111765, 0.8277081454896709],
            [0.5879052720788016, 0.6340710067484406],
        ],
    ]
    p_values = [
        [0.04680713680277496, 9.319365307373174e-50, 2.243909830126267e-38],
        [0.01994364304021059, 1.018091869165664e-70, 8.997992671314285e-55],
        [1, 0.03383130316782743, 0.03383130316782743],
        [0.04680713680277496, 9.319365307373174e-50, 2.243909830126267e-38],
    ]

    bar_labels = ["CAD", "GPT", "Gemini"]
    group_labels = ["Precision", "Recall", "Specificity", "Accuracy"]

    # Color choice
    color_choice = "blue"

    # Create the plot with min-max values and p-values, using the chosen color tone
    create_grouped_bar_plot_with_min_max_and_p_values(
        data,
        min_max_values,
        p_values,
        group_labels,
        bar_labels,
        color_choice=color_choice,
        show_plot=True,
    )
