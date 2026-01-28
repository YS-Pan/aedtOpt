import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
from datetime import datetime
import math  # For mathematical operations
import argparse  # Added for command line argument parsing

# Change the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set general matplotlib styles
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

plt.rcParams['axes.prop_cycle'] = cycler(
    'color', ['#FF0000', '#FFAA00', '#58A500', '#00BFE9', '#2000AA', '#960096', '#808080']
) + cycler('marker', ['o', 's', 'D', '^', 'v', '<', '>']) + \
    cycler('linestyle', ['None'] * 7)  # Removed fixed markersize and alpha from cycler


def read_data(filename, skip_rows=0):
    """
    Reads the optimization results from a CSV file.

    Parameters:
        filename (str): The path to the CSV file.
        skip_rows (int): Number of rows to skip from the beginning of the file (after the header).

    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        # First read just the header row (first row)
        header_df = pd.read_csv(filename, nrows=0)
        header_names = header_df.columns.tolist()
        
        # Now read the actual data, skipping the specified number of data rows but keeping the header
        if skip_rows > 0:
            # Read actual data, skipping specified rows after the header
            data = pd.read_csv(filename, on_bad_lines='skip', skiprows=range(1, skip_rows + 1))
        else:
            # No rows to skip, read normally
            data = pd.read_csv(filename, on_bad_lines='skip')
            
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filename}' is empty.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
        exit(1)


def get_cost_columns(data, start_column='endTime', end_column='fileName'):
    """
    Identifies the cost columns located between start_column and end_column.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        start_column (str): The column name indicating the start of cost columns.
        end_column (str): The column name indicating the end of cost columns.

    Returns:
        list: A list of cost column names.
    """
    try:
        start_idx = data.columns.get_loc(start_column) + 1
        end_idx = data.columns.get_loc(end_column)
        cost_columns = list(data.columns[start_idx:end_idx])
        return cost_columns
    except KeyError as e:
        print(f"Error: Column {e} not found in the data.")
        exit(1)


def is_pareto_efficient(costs):
    """
    Finds the Pareto-efficient points.

    Parameters:
        costs (np.ndarray): A 2D array where each row is a point and each column is an objective.

    Returns:
        np.ndarray: A boolean array indicating whether each point is Pareto efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # A point is not efficient if any other point is better in all objectives
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Keep current point
    return is_efficient


def gaussian_kernel_smoother(x_data, y_data, fine_x, sigma):
    """Compute Gaussian kernel smoothed values on fine_x."""
    smoothed = np.zeros_like(fine_x)
    for i, fx in enumerate(fine_x):
        weights = np.exp(- (x_data - fx)**2 / (2 * sigma**2))
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * y_data) / np.sum(weights)
    return smoothed


def plot_costs(data, cost_columns, pareto_mask, markersize_ordinary, alpha_ordinary):
    """
    Plots the individual and combined costs from the optimization results using scatter plots,
    highlighting the Pareto front.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        cost_columns (list): List of cost column names.
        pareto_mask (np.ndarray): Boolean array indicating Pareto front points.
        markersize_ordinary (float): Marker size for ordinary points.
        alpha_ordinary (float): Transparency for ordinary points.
    """
    num_goals = len(cost_columns)
    num_rows = data.shape[0]
    x = np.arange(1, num_rows + 1)

    # Extract individual costs
    try:
        individual_costs = data[cost_columns].astype(float)
    except ValueError as e:
        print(f"Error converting cost columns to float: {e}")
        exit(1)

    # Calculate combined cost
    combined_cost = individual_costs.sum(axis=1)

    # Retrieve colors and markers from the cycler
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = plt.rcParams['axes.prop_cycle'].by_key()['marker']

    # Define fixed marker size and border size for Pareto points
    fixed_markersize_pareto = 200  # Area in points^2 (adjust as needed)
    border_size_multiplier = 1.5   # Multiplier for the white border size

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot individual costs as scatter points
    for idx, col in enumerate(cost_columns):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Ordinary points
        ax1.scatter(
            x[~pareto_mask],
            individual_costs[col][~pareto_mask],
            label=col,
            marker=marker,
            edgecolors='none',
            facecolors=color,
            alpha=alpha_ordinary,
            s=markersize_ordinary ** 2  # 's' is marker area in points^2
        )

        if np.any(pareto_mask):
            # Pareto points: Solid white border
            ax1.scatter(
                x[pareto_mask],
                individual_costs[col][pareto_mask],
                label=None,  # Exclude Pareto points from legend to reduce clutter
                marker=marker,
                edgecolors='white',  # White border
                facecolors='white',  # Solid white fill for the border
                linewidths=0,         # No border line
                s=(math.sqrt(fixed_markersize_pareto) * border_size_multiplier) ** 2,  # Larger size for border
                zorder=2              # Ensure it's beneath the actual Pareto points
            )

            # Pareto points: Actual Pareto points with colored edges
            ax1.scatter(
                x[pareto_mask],
                individual_costs[col][pareto_mask],
                label=None,  # Exclude Pareto points from legend to reduce clutter
                marker=marker,
                edgecolors=color,       # Colored edges
                facecolors='none',      # Hollow markers
                linewidths=1.5,
                s=fixed_markersize_pareto,  # Fixed marker size
                zorder=3              # Ensure it's on top of the white border
            )

    ax1.set_xlabel('Row Number')
    ax1.set_ylabel('Individual Costs')
    ax1.set_ylim(0, 1.1)
    ax1.set_xlim(1, num_rows)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Create a second y-axis for combined cost
    ax2 = ax1.twinx()

    # Add local average line for combined cost using Line2D to bypass prop_cycle
    if num_rows > 0:
        fine_x = np.linspace(1, num_rows, 500)  # Reduced to 500 points to minimize density/artifacts
        total_points = num_rows
        window_points = max(1, int(0.03 * total_points))
        avg_spacing = (num_rows - 1) / (num_rows - 1) if num_rows > 1 else 1
        sigma = (window_points * avg_spacing) / 3.0
        local_avg = gaussian_kernel_smoother(x, combined_cost, fine_x, sigma)
        
        # Create a Line2D object directly to bypass any prop_cycle issues
        avg_line = Line2D(fine_x, local_avg, color='black', linewidth=10, 
                         alpha=0.3, linestyle='-', marker='', markersize=0)
        ax2.add_line(avg_line)
        avg_line.set_zorder(0)  # Ensure it's below other elements

    # Ordinary combined costs
    ax2.scatter(
        x[~pareto_mask],
        combined_cost[~pareto_mask],
        color='black',
        label='Combined Cost',
        marker='o',
        alpha=alpha_ordinary,
        s=markersize_ordinary ** 2
    )

    if np.any(pareto_mask):
        # Pareto combined costs: Solid white border
        ax2.scatter(
            x[pareto_mask],
            combined_cost[pareto_mask],
            facecolors='white',    # Solid white fill for the border
            edgecolors='white',    # White border
            linewidths=0,          # No border line
            marker='o',
            s=(math.sqrt(fixed_markersize_pareto) * border_size_multiplier) ** 2,  # Larger size for border
            zorder=2                # Ensure it's beneath the actual Pareto points
        )

        # Pareto combined costs: Actual Pareto points with black edges
        ax2.scatter(
            x[pareto_mask],
            combined_cost[pareto_mask],
            facecolors='none',
            edgecolors='black',         # Colored edges
            linewidths=1.5,
            marker='o',
            s=fixed_markersize_pareto,  # Fixed marker size
            zorder=3                  # Ensure it's on top of the white border
        )

    ax2.set_ylabel('Combined Cost')
    ax2.set_ylim(0, num_goals * 1.1)

    # Combine legends from both axes, excluding Pareto points
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Create a unique list of handles and labels
    unique = dict(zip(labels1 + labels2, handles1 + handles2))
    ax1.legend(unique.values(), unique.keys(), loc='lower left', frameon=True, fontsize=10)

    plt.title('Optimization Costs')
    plt.tight_layout()

    # Generate filename with current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"cost_{current_datetime}.png"

    # Save the plot in the current directory
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{output_filename}'")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize optimization costs with Pareto front.')
    parser.add_argument('--skip-rows', type=int, default=0, 
                        help='Number of rows to skip from the beginning of the CSV file (after the header).')
    args = parser.parse_args()
    
    # Specify the default CSV filename
    filename = 'output.csv'  # Default file name

    print(f"Reading data from '{filename}' (skipping {args.skip_rows} rows after header)")
    
    # Read the data with skip_rows parameter
    data = read_data(filename, args.skip_rows)

    # Identify cost columns
    cost_columns = get_cost_columns(data)

    if not cost_columns:
        print("No cost columns found between 'endTime' and 'fileName'.")
        exit(1)

    print(f"Identified cost columns: {cost_columns}")

    # Extract individual costs
    try:
        individual_costs = data[cost_columns].astype(float).values
    except ValueError as e:
        print(f"Error converting cost columns to float: {e}")
        exit(1)

    # Find Pareto front
    pareto_mask = is_pareto_efficient(individual_costs)

    # Calculate combined cost
    combined_cost = data[cost_columns].astype(float).sum(axis=1).values

    # If more than 10 Pareto points, select top 10 with lowest combined cost
    num_pareto = np.sum(pareto_mask)
    if num_pareto > 10:
        # Get indices of Pareto points
        pareto_indices = np.where(pareto_mask)[0]
        # Get combined costs of Pareto points
        pareto_combined_costs = combined_cost[pareto_mask]
        # Get indices of the 10 smallest combined costs
        top10_sorted_indices = np.argsort(pareto_combined_costs)[:10]
        # Get the actual indices in the data
        top10_indices = pareto_indices[top10_sorted_indices]
        # Create a new mask with only the top 10 Pareto points
        new_pareto_mask = np.zeros_like(pareto_mask)
        new_pareto_mask[top10_indices] = True
        pareto_mask = new_pareto_mask
        print(f"\nNumber of Pareto front points exceeds 10. Displaying top 10 with lowest combined cost.")
    else:
        pareto_indices = np.where(pareto_mask)[0]

    # Print Pareto front details
    if np.sum(pareto_mask) == 0:
        print("No Pareto front found.")
    else:
        print("\nPareto Front:")
        header = "Row\t" + "\t".join(cost_columns) + "\tCombined Cost\tFileName"
        print(header)
        for idx in np.where(pareto_mask)[0]:
            row_number = idx + 1  # Assuming row numbering starts at 1
            try:
                costs = "\t".join([f"{data[col].iloc[idx]}" for col in cost_columns])
            except IndexError:
                costs = "\t".join(['N/A'] * len(cost_columns))
            try:
                total_cost = combined_cost[idx]
                filename_p = data['fileName'].iloc[idx]
                print(f"{row_number}\t{costs}\t{total_cost:.4f}\t{filename_p}")
            except KeyError:
                print(f"{row_number}\t{costs}\t{total_cost:.4f}\tN/A")

    # Determine markersize and alpha based on number of rows
    num_rows = data.shape[0]
    threshold = 1000  # Threshold after which to adjust markersize

    if num_rows <= threshold:
        markersize_ordinary = 6
        alpha_ordinary = 0.6
    else:
        # Calculate markersize inversely proportional to sqrt(num_rows)
        # This ensures that markersize decreases as num_rows increases
        # while keeping the total area of all points roughly constant
        scaling_constant = 6 * math.sqrt(threshold)  # 6 * sqrt(1000) ≈ 189.73
        markersize_ordinary = scaling_constant / math.sqrt(num_rows)
        # Ensure markersize does not become too small
        markersize_ordinary = max(1, markersize_ordinary)
        # Adjust alpha similarly, set a minimum of 0.1
        alpha_ordinary = max(0.3, 0.6 * (math.sqrt(threshold) / math.sqrt(num_rows)))

    # Plot the costs with Pareto front highlighted
    plot_costs(data, cost_columns, pareto_mask, markersize_ordinary, alpha_ordinary)


if __name__ == "__main__":
    main()
