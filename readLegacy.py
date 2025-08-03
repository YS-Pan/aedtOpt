import pandas as pd
import numpy as np
import os
import re
import random
import json

def read_legacy_csv(filename='legacy.csv', delimiter=','):
    """
    Reads the legacy.csv file and extracts optimization variable names, their values, units, and other information.

    Args:
        filename (str): The name of the CSV file to read.
        delimiter (str): The delimiter used in the CSV file. Default is ','.

    Returns:
        optVarNames (list): List of optimization variable names.
        var_values (list of lists): Nested list containing values for each variable across all rows.
        var_units (list): List containing the unit for each optimization variable.
        other_info (pd.DataFrame): DataFrame containing all other columns not part of optimization variables.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{filename}' was not found in '{script_dir}'.")

    try:
        df = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    df.columns = df.columns.str.strip()
    lower_columns = [col.lower() for col in df.columns]

    if 'starttime' not in lower_columns:
        raise KeyError(
            "The CSV file does not contain a 'startTime' column to separate the parameter section.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    start_idx = lower_columns.index('starttime')
    param_columns = df.columns[:start_idx]
    optVarNames = list(param_columns)

    var_values = []
    var_units = [None] * len(optVarNames)

    pattern = re.compile(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(.*)$')

    for _, row in df.iterrows():
        row_values = []
        for i, col in enumerate(param_columns):
            cell = str(row[col]).strip()
            if pd.isna(cell) or cell == '':
                row_values.append(None)
                continue
            match = pattern.match(cell)
            if match:
                value_str, unit = match.groups()
                try:
                    value = float(value_str)
                except ValueError:
                    value = None
                row_values.append(value)
                # If we haven't assigned a unit yet and found one, keep it
                if var_units[i] is None and unit:
                    var_units[i] = unit
                elif unit and var_units[i] != unit:
                    var_units[i] = None  # Inconsistent unit found, set to None
            else:
                row_values.append(None)
        var_values.append(row_values)

    other_info = df.iloc[:, start_idx:].copy()
    return optVarNames, var_values, var_units, other_info

def read_optimization_cost(filename='legacy.csv', delimiter=','):
    """
    Reads the optimization cost data from the CSV file.

    Args:
        filename (str): The name of the CSV file to read.
        delimiter (str): The delimiter used in the CSV file. Default is ','.

    Returns:
        cost_columns (list): List of optimization cost column names.
        cost_data (pd.DataFrame): DataFrame containing the cost values.
        file_names (list): List of file names corresponding to each variation.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{filename}' was not found in '{script_dir}'.")

    try:
        df = pd.read_csv(file_path, delimiter=delimiter, on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    df.columns = df.columns.str.strip()
    lower_columns = [col.lower() for col in df.columns]

    try:
        start_time_idx = lower_columns.index('starttime')
        end_time_idx = lower_columns.index('endtime')
        file_name_idx = lower_columns.index('filename')
    except ValueError as e:
        raise KeyError(f"Required column not found: {e}")

    if start_time_idx >= end_time_idx:
        raise ValueError("'endTime' column should come after 'startTime' column.")

    cost_columns = df.columns[end_time_idx + 1:-1]
    file_names = df['fileName'].tolist()
    cost_data = df[cost_columns].copy()

    # Convert cost_data to numeric
    cost_data = cost_data.apply(pd.to_numeric, errors='coerce')

    return cost_columns.tolist(), cost_data, file_names

def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points.

    :param costs: An (n_points, n_costs) array
    :return: A boolean array indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Any point that is dominated by this point is not efficient
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def select_variations(total_picks=10, ratio=0.6, filename='legacy.csv', delimiter=','):
    """
    Selects variations based on Pareto front and random sampling within the legacy data.

    Args:
        total_picks (int): Total number of variations to pick.
        ratio (float): Ratio of Pareto front variations to total picks (0.6 by default).
        filename (str): The name of the CSV file to read.
        delimiter (str): The delimiter used in the CSV file. Default is ','.

    Returns:
        optVarNames (list): List of optimization variable names.
        selected_var_values (list of lists): Nested list of param values for selected variations.
        selected_var_units (list): List containing the unit for each optimization variable.
        selected_other_info (pd.DataFrame): Additional info for the selected rows.
    """
    # Read all data
    optVarNames, var_values, var_units, other_info = read_legacy_csv(filename, delimiter)
    cost_columns, cost_data, file_names = read_optimization_cost(filename, delimiter)

    # Align other_info with cost_data and file_names
    other_info = other_info.copy()
    other_info[cost_columns] = cost_data
    other_info['fileName'] = file_names

    # Convert cost_data to a numpy array
    costs = cost_data.values

    # Handle rows with any cost as NaN or >= 1.1 as invalid
    valid_mask = (~cost_data.isna().any(axis=1)) & (cost_data < 1.1).all(axis=1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("No valid variations found with all cost values between 0 and 1.")

    valid_costs = costs[valid_mask]
    pareto_mask = is_pareto_efficient(valid_costs)
    pareto_indices = valid_indices[pareto_mask]

    # Decide how many picks come from Pareto vs random among valid legacy
    pareto_picks = int(round(total_picks * ratio))
    random_picks = total_picks - pareto_picks

    selected_pareto = []
    if len(pareto_indices) > pareto_picks:
        combined_costs = cost_data.iloc[pareto_indices].sum(axis=1).values
        sorted_pareto_indices = pareto_indices[np.argsort(combined_costs)]
        selected_pareto = sorted_pareto_indices[:pareto_picks].tolist()
    else:
        selected_pareto = pareto_indices.tolist()
        remaining_picks = pareto_picks - len(selected_pareto)
        if remaining_picks > 0:
            non_pareto_mask = valid_mask.copy()
            non_pareto_mask[pareto_indices] = False
            non_pareto_indices = np.where(non_pareto_mask)[0]
            if len(non_pareto_indices) < remaining_picks:
                raise ValueError("Not enough variations to fill the Pareto front picks.")
            combined_costs_non_pareto = cost_data.iloc[non_pareto_indices].sum(axis=1).values
            sorted_non_pareto_indices = non_pareto_indices[np.argsort(combined_costs_non_pareto)]
            selected_pareto.extend(sorted_non_pareto_indices[:remaining_picks].tolist())

    selected_pareto = list(set(selected_pareto))

    remaining_valid_indices = list(set(valid_indices) - set(selected_pareto))
    if len(remaining_valid_indices) < random_picks:
        raise ValueError("Not enough variations to fulfill the random picks without duplication.")

    selected_random = random.sample(remaining_valid_indices, random_picks)
    selected_indices = selected_pareto + selected_random

    selected_var_values = [var_values[idx] for idx in selected_indices]
    selected_var_units = var_units.copy()
    selected_other_info = other_info.iloc[selected_indices].reset_index(drop=True)

    return optVarNames, selected_var_values, selected_var_units, selected_other_info

def generate_first_generation(MU, param_file='parameters.json', legacy_file='legacy.csv', legacy_ratio=0.5):
    """
    Generate the first generation of solutions from an existing legacy.csv file.

    Args:
        MU (int): Number of individuals in the population.
        param_file (str): Path to the parameters.json file (from get_opt_parameters).
        legacy_file (str): Path to the legacy.csv file.
        legacy_ratio (float): Fraction of the population to sample from legacy.csv. Default=0.4.

    This function proceeds as follows:
      1) Reads parameter data (names, min, max) from param_file.
      2) If legacy_file exists, attempts to pick 'legacy_count = int(round(MU * legacy_ratio))'.
      3) For each parameter that exists in parameters.json but not in legacy.csv, assign a random value.
         For each parameter that exists in legacy.csv but not in parameters.json, discard those values.
      4) Scale the resulting legacy parameter values to [0,1] based on min/max from parameters.json,
         and clamp out-of-range values to [0,1].
      5) If the number of valid picks from the legacy is fewer than legacy_count, we accept what we have.
      6) Fill the remainder up to MU with pure random in [0,1].
      7) Returns an MU×NDIM list of lists (each sub-list is an individual in normalized space).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    param_path = os.path.join(script_dir, param_file)
    legacy_path = os.path.join(script_dir, legacy_file)

    # Bail out if parameter file doesn't exist
    if not os.path.isfile(param_path):
        return None

    # Bail out if legacy file doesn't exist
    if not os.path.isfile(legacy_path):
        return None

    # Load parameter data
    with open(param_path, 'r') as f:
        params = json.load(f)
    refNames = params['optVarNames']
    refMins = params['var_mins']
    refMaxs = params['var_maxs']

    NDIM = len(refNames)
    if NDIM == 0:
        return None

    # Number of picks from legacy, rest is random
    legacy_picks = int(round(MU * legacy_ratio))

    # We'll accumulate final results in "generation"
    generation = []

    # If we want some (legacy_picks > 0) from legacy.csv
    if legacy_picks > 0:
        try:
            # We'll use ratio=0.6 inside select_variations to decide Pareto vs random within the legacy
            legacyNames, legacyValues, _, _ = select_variations(
                total_picks=legacy_picks,
                ratio=0.6,
                filename=legacy_file
            )

            # Build a mapping from legacyName -> index among legacy columns
            legacyIndexMap = {name: i for i, name in enumerate(legacyNames)}

            # Scale each row from legacy
            for row in legacyValues:
                newInd = []
                for j, refN in enumerate(refNames):
                    if refN in legacyIndexMap:
                        val_idx = legacyIndexMap[refN]
                        val = row[val_idx]
                        if val is None:
                            scaled_val = random.random()
                        else:
                            mn = refMins[j]
                            mx = refMaxs[j]
                            if mx == mn:
                                scaled_val = 0.0
                            else:
                                scaled_val = (val - mn) / (mx - mn)
                            # clamp to [0,1]
                            if scaled_val < 0:
                                scaled_val = 0.0
                            elif scaled_val > 1:
                                scaled_val = 1.0
                    else:
                        # random for any param not in legacy
                        scaled_val = random.random()
                    newInd.append(scaled_val)
                generation.append(newInd)
        except Exception as e:
            print(f"Error selecting legacy variations: {e}")
            # If it fails, generation remains empty

    # Fill the remainder randomly if needed
    if len(generation) < MU:
        needed = MU - len(generation)
        for _ in range(needed):
            newInd = []
            for j in range(NDIM):
                newInd.append(random.random())
            generation.append(newInd)

    # If we somehow got too many, truncate
    if len(generation) > MU:
        generation = generation[:MU]

    # Return final generation
    return generation
