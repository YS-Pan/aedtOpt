"""
Configuration file for HFSS optimization
This file centralizes all parameters and data extraction functions
"""
import numpy as np

# Project and design configuration
PROJECT_NAME = '20251004 hp helix.aedt'
DESIGN_NAME = '1 model'

# Simulation configuration
TIMEOUT = 3600  # Timeout for each HFSS simulation in seconds
HFSS_LAUNCH_INTERVAL = 10.0
TOLERANCE = 1e-3  # Tolerance for floating-point comparisons, in GHz

# Optimization algorithm configuration
P = 10  # Number of reference points per objective
NGEN = 50  # Number of generations
CXPB = 1.0  # Crossover probability
MUTPB = 1.0  # Mutation probability
NUM_PROCESSES = 6  # Number of parallel processes

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
        expression = "max(dB(S(1,1)),dB(S(t1,1))*1.5)"
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


def min_db_RealizedGainPhi(hfssApp, designName):
    """
    Steps:
    1) For each (Freq, Theta), compute across all Phi: min*0.8 + avg*0.2.
       This yields a Gain matrix vs (Freq, Theta).
    2) For each Freq, take the max across Theta within [80, 100] deg.
       This yields Gain vs Freq.
    3) Across all Freq, compute min*0.8 + avg*0.2 and return the scalar.
    Best result: higher is better.
    """
    try:
        hfssApp.set_active_design(designName)
        
        expression = "dB(RealizedGainPhi)"
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
            print("Failed to retrieve dB(RealizedGainPhi) data.")
            return False
            
        # Process the data
        result1 = sData.full_matrix_real_imag
        result2 = sData.intrinsics
        
        # Extract Theta, Phi, Freq points
        try:
            ThetaPoints = [float(theta) for theta in result2['Theta']]
            PhiPoints = [float(phi) for phi in result2['Phi']]
            FreqPoints = [float(freq) for freq in result2['Freq']]
        except Exception:
            print("Failed to parse Theta/Phi/Freq points.")
            return False
        
        if not ThetaPoints or not PhiPoints or not FreqPoints:
            print("Theta, Phi, or Freq points are missing.")
            return False
        
        # Create and populate the data array: shape (Freq, Phi, Theta)
        data = np.zeros((len(FreqPoints), len(PhiPoints), len(ThetaPoints)), dtype=float)
        expr_data = result1[0][expression]
        for (freq, phi, theta), value in expr_data.items():
            try:
                f = float(freq); p = float(phi); t = float(theta)
                freq_idx = next(i for i, fv in enumerate(FreqPoints) if abs(fv - f) < TOLERANCE)
                phi_idx = next(i for i, pv in enumerate(PhiPoints) if abs(pv - p) < TOLERANCE)
                theta_idx = next(i for i, tv in enumerate(ThetaPoints) if abs(tv - t) < TOLERANCE)
                data[freq_idx, phi_idx, theta_idx] = float(value)
            except (StopIteration, ValueError, TypeError):
                continue
        
        # Step 1: Across all Phi -> min*0.8 + avg*0.2, per (Freq, Theta)
        min_phi = np.min(data, axis=1)   # shape: (Freq, Theta)
        avg_phi = np.mean(data, axis=1)  # shape: (Freq, Theta)
        gain_vs_freq_theta = 0.8 * min_phi + 0.2 * avg_phi  # shape: (Freq, Theta)
        
        # Step 2: For each Freq, max across Theta within [80, 100] deg
        theta_indices = [i for i, th in enumerate(ThetaPoints) if 88.0 <= th <= 92.0]
        if not theta_indices:
            print("No Theta points found within 80deg to 100deg.")
            return False
        gain_vs_freq = np.max(gain_vs_freq_theta[:, theta_indices], axis=1)  # shape: (Freq,)
        
        if gain_vs_freq.size == 0:
            print("No data available after Theta-range reduction.")
            return False
        
        # Step 3: Across all Freq -> min*0.8 + avg*0.2
        final_min = float(np.min(gain_vs_freq))
        final_avg = float(np.mean(gain_vs_freq))
        return 0.8 * final_min + 0.2 * final_avg

    except Exception as e:
        print(f"Error in min_db_RealizedGainPhi: {e}")
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
        theta_indices = [i for i, theta in enumerate(ThetaPoints) if (0.0 <= theta <= 60.0) or (120.0 <= theta <= 180.0)]
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
    (ActionType.COST_CALCULATION, max_dB_S11_S22, -15, -2),
    (ActionType.COST_CALCULATION, min_db_RealizedGainPhi, 8, 0),
    (ActionType.COST_CALCULATION, min_db_AxialRatio, 25, 3),
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