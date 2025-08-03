"""
Configuration file for HFSS optimization
This file centralizes all parameters and data extraction functions
"""
import numpy as np

# Project and design configuration
PROJECT_NAME = '20250801 lp helix.aedt'
DESIGN_NAME = '2 helixCage'

# Simulation configuration
TIMEOUT = 1800  # Timeout for each HFSS simulation in seconds
HFSS_LAUNCH_INTERVAL = 10.0
TOLERANCE = 1e-3  # Tolerance for floating-point comparisons, in GHz

# Optimization algorithm configuration
P = 12  # Number of reference points per objective
NGEN = 50  # Number of generations
CXPB = 1.0  # Crossover probability
MUTPB = 1.0  # Mutation probability
NUM_PROCESSES = 3  # Number of parallel processes

# Cost when error occurs
COST_ERROR = 1.1

# Define action types
class ActionType:
    COST_CALCULATION = "cost_calculation"
    MODIFICATION = "modification"

# Data extraction functions
def max_dB_S11_S22(hfssApp, designName):
    """
    Calculate the maximum of dB(S(1,1))+dB(S(2,2)) across all available frequency points.
    Expression: dB(S(1,1))+dB(S(2,2))
    Best result: -20 dB, Worst: -2 dB
    """
    try:
        # Set the active design
        hfssApp.set_active_design(designName)
        postApp = hfssApp.post

        # Extract dB(S(1,1))+dB(S(2,2))
        expression = "dB(S(1,1))+dB(S(2,2))"
        sData = postApp.get_solution_data(expression, setup_sweep_name='opt')
        if not sData:
            print("Failed to retrieve dB(S(1,1))+dB(S(2,2)) data.")
            return False

        # Extract frequency and data
        freq = np.array(sData.variation_values("Freq"))
        data_db = np.array(sData.data_real())

        # Use all available frequency points (no filtering)
        if len(data_db) == 0:
            print("No frequency points available.")
            return False

        # Find the worst (maximum) value across all frequency points
        worst_max = np.max(data_db)*0.9 + np.average(data_db)*0.1
        return worst_max
    except Exception as e:
        print(f"Error in max_dB_S11_S22: {e}")
        return False

def min_db_RealizedGainTheta(hfssApp, designName):
    """
    Calculate min*0.5 + avg*0.5 of dB(RealizedGainTheta) at theta=90deg (or closest), phi=all, across all frequencies.
    Best result: 8 dB, Worst: -10 dB
    """
    try:
        hfssApp.set_active_design(designName)
        
        expression = "dB(RealizedGainTheta)"
        sData = hfssApp.post.get_solution_data(
            expressions=[expression],
            setup_sweep_name='opt',
            report_category="Far Fields",
            context="3D",
            variations={
                "Theta": ["All"],
                "Phi": ["All"],
                "Freq": ["All"]
            },
        )
        
        if not sData:
            print("Failed to retrieve dB(RealizedGainTheta) data.")
            return False
            
        # Process the data
        result1 = sData.full_matrix_real_imag
        result2 = sData.intrinsics
        
        # Extract Theta, Phi, Freq points
        ThetaPoints = [float(theta) for theta in result2['Theta']]
        PhiPoints = [float(phi) for phi in result2['Phi']]
        FreqPoints = [float(freq) for freq in result2['Freq']]
        
        # Create the data dictionary
        data_dict = {}
        data_dict[expression] = np.zeros((len(FreqPoints), len(PhiPoints), len(ThetaPoints)))
        
        # Populate the data dictionary
        expr_data = result1[0][expression]
        for (freq, phi, theta), value in expr_data.items():
            freq = float(freq)
            phi = float(phi)
            theta = float(theta)
            try:
                freq_idx = next(i for i, f in enumerate(FreqPoints) if abs(f - freq) < TOLERANCE)
                phi_idx = next(i for i, p in enumerate(PhiPoints) if abs(p - phi) < TOLERANCE)
                theta_idx = next(i for i, t in enumerate(ThetaPoints) if abs(t - theta) < TOLERANCE)
                data_dict[expression][freq_idx, phi_idx, theta_idx] = value
            except (StopIteration, ValueError):
                continue
        
        # Use all frequency points (no filtering)
        freq_indices = list(range(len(FreqPoints)))
        if not freq_indices:
            print("No frequency points available.")
            return False

        # Find the theta closest to 90deg
        theta_diffs = [abs(t - 90.0) for t in ThetaPoints]
        if not theta_diffs:
            print("No Theta points available.")
            return False
        min_diff = min(theta_diffs)
        theta_idx_closest = theta_diffs.index(min_diff)

        # Phi range: all
        phi_indices = list(range(len(PhiPoints)))

        # Extract relevant data (all freq, all phi, closest theta)
        data = data_dict[expression]
        relevant_data = data[np.ix_(freq_indices, phi_indices, [theta_idx_closest])]
        if relevant_data.size == 0:
            print("No relevant data found for dB(RealizedGainTheta).")
            return False
        
        min_val = np.min(relevant_data)
        avg_val = np.average(relevant_data)
        return (min_val * 0.5 + avg_val * 0.5)
    except Exception as e:
        print(f"Error in min_db_RealizedGainTheta: {e}")
        return False

def min_db_AxialRatio(hfssApp, designName):
    """
    Calculate min*0.5 + avg*0.5 of dB(AxialRatioValue) within theta=[80deg,100deg], phi=all, across all frequencies.
    Best result: 30 dB, Worst: 3 dB
    """
    try:
        hfssApp.set_active_design(designName)
        
        expression = "dB(AxialRatioValue)"
        sData = hfssApp.post.get_solution_data(
            expressions=[expression],
            setup_sweep_name='opt',
            report_category="Far Fields",
            context="3D",
            variations={
                "Theta": ["All"],
                "Phi": ["All"],
                "Freq": ["All"]
            },
        )
        
        if not sData:
            print("Failed to retrieve dB(AxialRatioValue) data.")
            return False
            
        # Process the data
        result1 = sData.full_matrix_real_imag
        result2 = sData.intrinsics
        
        # Extract Theta, Phi, Freq points
        ThetaPoints = [float(theta) for theta in result2['Theta']]
        PhiPoints = [float(phi) for phi in result2['Phi']]
        FreqPoints = [float(freq) for freq in result2['Freq']]
        
        # Create the data dictionary
        data_dict = {}
        data_dict[expression] = np.zeros((len(FreqPoints), len(PhiPoints), len(ThetaPoints)))
        
        # Populate the data dictionary
        expr_data = result1[0][expression]
        for (freq, phi, theta), value in expr_data.items():
            freq = float(freq)
            phi = float(phi)
            theta = float(theta)
            try:
                freq_idx = next(i for i, f in enumerate(FreqPoints) if abs(f - freq) < TOLERANCE)
                phi_idx = next(i for i, p in enumerate(PhiPoints) if abs(p - phi) < TOLERANCE)
                theta_idx = next(i for i, t in enumerate(ThetaPoints) if abs(t - theta) < TOLERANCE)
                data_dict[expression][freq_idx, phi_idx, theta_idx] = value
            except (StopIteration, ValueError):
                continue
        
        # Use all frequency points (no filtering)
        freq_indices = list(range(len(FreqPoints)))
        if not freq_indices:
            print("No frequency points available.")
            return False

        # Define Theta range [80deg,100deg]
        theta_indices = [i for i, theta in enumerate(ThetaPoints) if 80.0 <= theta <= 100.0]
        if not theta_indices:
            print("No Theta points found within 80deg to 100deg.")
            return False

        # Phi range: all
        phi_indices = list(range(len(PhiPoints)))

        # Extract relevant data
        data = data_dict[expression]
        relevant_data = data[np.ix_(freq_indices, phi_indices, theta_indices)]
        if relevant_data.size == 0:
            print("No relevant data found for dB(AxialRatioValue).")
            return False
        
        min_val = np.min(relevant_data)
        avg_val = np.average(relevant_data)
        return (min_val * 0.5 + avg_val * 0.5)
    except Exception as e:
        print(f"Error in min_db_AxialRatio: {e}")
        return False

def max_db_DirTotal(hfssApp, designName):
    """
    Calculate max*0.5 + avg*0.5 of dB(DirTotal) within theta=[0deg,30deg] and [150deg,180deg], phi=all, across all frequencies.
    Best result: -10 dB, Worst: 10 dB
    """
    try:
        hfssApp.set_active_design(designName)
        
        expression = "dB(DirTotal)"
        sData = hfssApp.post.get_solution_data(
            expressions=[expression],
            setup_sweep_name='opt',
            report_category="Far Fields",
            context="3D",
            variations={
                "Theta": ["All"],
                "Phi": ["All"],
                "Freq": ["All"]
            },
        )
        
        if not sData:
            print("Failed to retrieve dB(DirTotal) data.")
            return False
            
        # Process the data
        result1 = sData.full_matrix_real_imag
        result2 = sData.intrinsics
        
        # Extract Theta, Phi, Freq points
        ThetaPoints = [float(theta) for theta in result2['Theta']]
        PhiPoints = [float(phi) for phi in result2['Phi']]
        FreqPoints = [float(freq) for freq in result2['Freq']]
        
        # Create the data dictionary
        data_dict = {}
        data_dict[expression] = np.zeros((len(FreqPoints), len(PhiPoints), len(ThetaPoints)))
        
        # Populate the data dictionary
        expr_data = result1[0][expression]
        for (freq, phi, theta), value in expr_data.items():
            freq = float(freq)
            phi = float(phi)
            theta = float(theta)
            try:
                freq_idx = next(i for i, f in enumerate(FreqPoints) if abs(f - freq) < TOLERANCE)
                phi_idx = next(i for i, p in enumerate(PhiPoints) if abs(p - phi) < TOLERANCE)
                theta_idx = next(i for i, t in enumerate(ThetaPoints) if abs(t - theta) < TOLERANCE)
                data_dict[expression][freq_idx, phi_idx, theta_idx] = value
            except (StopIteration, ValueError):
                continue
        
        # Use all frequency points (no filtering)
        freq_indices = list(range(len(FreqPoints)))
        if not freq_indices:
            print("No frequency points available.")
            return False

        # Define Theta ranges [0deg,30deg] and [150deg,180deg]
        theta_indices = [i for i, theta in enumerate(ThetaPoints) if (0.0 <= theta <= 30.0) or (150.0 <= theta <= 180.0)]
        if not theta_indices:
            print("No Theta points found within [0-30] or [150-180] deg.")
            return False

        # Phi range: all
        phi_indices = list(range(len(PhiPoints)))

        # Extract relevant data
        data = data_dict[expression]
        relevant_data = data[np.ix_(freq_indices, phi_indices, theta_indices)]
        if relevant_data.size == 0:
            print("No relevant data found for dB(DirTotal).")
            return False
        
        max_val = np.max(relevant_data)
        avg_val = np.average(relevant_data)
        return (max_val * 0.5 + avg_val * 0.5)
    except Exception as e:
        print(f"Error in max_db_DirTotal: {e}")
        return False

# Modification function
def set_design(hfssApp, design_name):
    """
    Set the active design to the specified design name.
    """
    try:
        hfssApp.set_active_design(design_name)
        return True
    except Exception as e:
        print(f"Error setting design to {design_name}: {e}")
        return False

# Define actions to be performed during optimization
# Format: (action_type, function, [args for cost calculation])
# For cost calculation: (ActionType.COST_CALCULATION, function, best_value, worst_value)
# For modification: (ActionType.MODIFICATION, function)
ACTIONS = [
    (ActionType.MODIFICATION, set_design), 
    (ActionType.COST_CALCULATION, max_dB_S11_S22, -20, -2),
    (ActionType.COST_CALCULATION, min_db_RealizedGainTheta, 8, -10),
    (ActionType.COST_CALCULATION, min_db_AxialRatio, 30, 3),
    (ActionType.COST_CALCULATION, max_db_DirTotal, -10, 10),
]

# Helper functions with improved processing
def get_cost_calculation_actions():
    """Get a list of cost calculation actions."""
    result = []
    for action in ACTIONS:
        if action[0] == ActionType.COST_CALCULATION:
            # Unpack the action tuple correctly
            _, func, best, worst = action
            result.append((func, DESIGN_NAME, best, worst))
    return result

def get_modification_actions():
    """Get a list of modification actions."""
    result = []
    for action in ACTIONS:
        if action[0] == ActionType.MODIFICATION:
            # Unpack the action tuple correctly
            _, func = action
            result.append((func, DESIGN_NAME))
    return result

def get_data_extraction_function_names():
    """Get a list of all data extraction function names."""
    return [func.__name__ for action_type, func, *_ in ACTIONS 
            if action_type == ActionType.COST_CALCULATION]

def get_number_of_objectives():
    """Get the number of objectives (cost calculation actions)."""
    return len([action for action in ACTIONS 
                if action[0] == ActionType.COST_CALCULATION])

def generate_csv_headers(parameter_names):
    """Generate headers for CSV export."""
    return parameter_names + ["startTime", "endTime"] + get_data_extraction_function_names() + ["fileName"]
