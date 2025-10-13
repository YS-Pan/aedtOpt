import contextlib
import os
import sys
import shutil
import time
import random
import math
import csv
import ast
import re
import numpy as np
import func_timeout
from functools import partial

# Import configuration from optConfig
from optConfig import (
    PROJECT_NAME, DESIGN_NAME, TOLERANCE, COST_ERROR,
    get_cost_calculation_actions, get_modification_actions, generate_csv_headers
)

@contextlib.contextmanager
def suppress_output():
    """Suppresses stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def parse_value_with_unit(val_str):
    """
    Splits a string like '14mm' or '-30deg' into (14.0, 'mm') or (-30.0, 'deg').
    """
    match = re.match(r'^([-+]?\d*\.?\d+)([a-zA-Z]*)$', val_str.replace(' ', ''))
    if match:
        num, unit = match.groups()
        try:
            return float(num), unit
        except ValueError:
            return None, unit
    else:
        # If it doesn't match, try casting to float, set unit as empty
        try:
            return float(val_str), ''
        except:
            return None, ''

def find_nearest_index(lst, value, tol=TOLERANCE):
    """
    Finds the index of the element in lst that is closest to value within a tolerance.
    Raises ValueError if no such element is found.
    """
    for i, v in enumerate(lst):
        if abs(v - value) < tol:
            return i
    raise ValueError(f"No value within {tol} of {value} found in list.")

def get_opt_parameters():
    """
    Retrieve a list of optimization parameter names along with their minimum and maximum values and units from the HFSS design.
    Avoid modifying or deleting this function.
    """
    from ansys.aedt.core import Hfss  # Imported inside suppress_output to avoid global import in main.py

    with suppress_output():
        try:
            hfss = Hfss(
                project=PROJECT_NAME,
                design=DESIGN_NAME,
                new_desktop=True,
                non_graphical=False,
                close_on_exit=True,
                remove_lock=True
            )
        except Exception as e:
            sys.exit(f"Failed to open HFSS project '{PROJECT_NAME}', design '{DESIGN_NAME}': {e}")

    # Get all variables
    variables = hfss.variable_manager.variables

    # Collect variables enabled for optimization along with their min and max ranges and units
    var_names = []
    var_mins = []
    var_maxs = []
    var_units = []

    for var_name, var_obj in variables.items():
        try:
            if var_obj.is_optimization_enabled:
                min_val_str = var_obj.optimization_min_value
                max_val_str = var_obj.optimization_max_value

                min_val, unit_min = parse_value_with_unit(min_val_str)
                max_val, unit_max = parse_value_with_unit(max_val_str)

                if min_val is None or max_val is None:
                    continue

                if unit_min != unit_max:
                    continue
                unit = unit_min  # Assuming same unit

                var_names.append(var_name)
                var_mins.append(min_val)
                var_maxs.append(max_val)
                var_units.append(unit)
        except AttributeError:
            # Variable does not have optimization properties
            continue
        except Exception:
            continue

    hfss.release_desktop()
    return var_names, var_mins, var_maxs, var_units

def hfss_init(non_graphical=True):
    """
    Initialize the HFSS environment by copying the project and opening the specified design.
    Avoid modifying this function.
    """
    from ansys.aedt.core import Hfss  # Imported inside suppress_output to avoid global import in main.py

    file_name, file_extension = os.path.splitext(PROJECT_NAME)

    dest_folder = "optCopy"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Create a unique copy of the project file to avoid conflicts
    new_file = os.path.join(dest_folder, f"{file_name}_{int(time.time())}{file_extension}")
    shutil.copyfile(PROJECT_NAME, new_file)

    for attempt in range(3):
        with suppress_output():
            try:
                hfssApp = Hfss(
                    project=new_file,
                    design=DESIGN_NAME,
                    new_desktop=True,
                    non_graphical=non_graphical,
                    close_on_exit=True,
                    remove_lock=True
                )
                return [hfssApp, new_file]
            except:
                time.sleep(random.uniform(1, 3))
    return ["hfss_init failed", new_file]

def hfss_setPara(hfss, nameList, valList):
    """
    Set the parameter values in the HFSS design.
    Avoid modifying this function.
    """
    listOfChange = ["NAME:ChangedProps"]
    for name, val in zip(nameList, valList):
        listOfChange += [["NAME:" + name, "Value:=", str(val)]]
    hfss._odesign.ChangeProperty([
        "NAME:AllTabs",
        [
            "NAME:LocalVariableTab",
            [
                "NAME:PropServers",
                "LocalVariables"
            ],
            listOfChange
        ]
    ])

def get_3d_solution_data(hfssApp, expressions, setup_sweep_name='opt'):
    """
    Fetch and organize 3D solution data from HFSS for multiple expressions.

    Parameters:
    - expressions: list of expressions to fetch.
    - setup_sweep_name: name of the sweep setup.

    Returns:
    - data_dict: dictionary with expression as keys and data matrices as values.
    - ThetaPoints, PhiPoints, FreqPoints: lists of theta, phi, and frequency points.

    Avoid modifying this function.
    """
    # Ensure expressions is a list
    if isinstance(expressions, str):
        expressions = [expressions]

    # Fetch the solution data object for the given expressions and sweep
    sData = hfssApp.post.get_solution_data(
        expressions=expressions,
        setup_sweep_name=setup_sweep_name,
        report_category="Far Fields",
        context="3D",
        variations={
            "Theta": ["All"],
            "Phi": ["All"],
            "Freq": ["All"]
        },
    )

    if not sData:
        return None, None, None, None

    # Extract the result matrices and the intrinsics (points for freq, phi, theta)
    result1 = sData.full_matrix_real_imag
    result2 = sData.intrinsics

    # Extract Theta, Phi, Freq points from result2
    ThetaPoints = [float(theta) for theta in result2['Theta']]
    PhiPoints = [float(phi) for phi in result2['Phi']]
    FreqPoints = [float(freq) for freq in result2['Freq']]

    # Initialize a dictionary to store the sorted data for each expression
    data_dict = {}

    for expr in expressions:
        data_dict[expr] = np.zeros((len(FreqPoints), len(PhiPoints), len(ThetaPoints)))

    # Iterate over each expression and populate the data matrices
    for expr in expressions:
        expr_data = result1[0][expr]
        for (freq, phi, theta), value in expr_data.items():
            freq = float(freq)
            phi = float(phi)
            theta = float(theta)
            try:
                freq_idx = find_nearest_index(FreqPoints, freq)
                phi_idx = find_nearest_index(PhiPoints, phi)
                theta_idx = find_nearest_index(ThetaPoints, theta)
                data_dict[expr][freq_idx, phi_idx, theta_idx] = value
            except ValueError:
                continue

    return data_dict, ThetaPoints, PhiPoints, FreqPoints

def calcCost(result, best, worst):
    """
    Calculate the normalized cost based on the result.
    """
    if result is False:
        return 1.0  # Maximum cost if an error occurred
    else:
        return (math.tanh(4 / (worst - best) * result + 4 * worst / (best - worst) + 2) + 1) / 2

def uniform(low, up, size=None):
    """
    Generate a list of random numbers within specified bounds.
    """
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low]*size, [up]*size)]

def hfss_cost(
        hfssInitResult,         # list with two elements [hfssApp, fileName]
        inVals: list,           # value scaled to [0,1]
        inNames: list,
        var_units: list,
        var_mins: list,
        var_maxs: list,
        question="none"
) -> list:
    """
    Compute the cost function by setting parameters, running the simulation, and extracting results.
    """
    try:
        # Get actions from configuration
        cost_actions = get_cost_calculation_actions()
        modification_actions = get_modification_actions()
        
        if question == "number of objective":
            return len(cost_actions)

        costValList = [0] * len(cost_actions)
        costError = COST_ERROR  # Cost when an error occurs

        # Record the start time
        optTime = [0] * 2
        optTime[0] = time.strftime("%H:%M:%S", time.localtime())

        hfssApp, fileName = hfssInitResult

        if hfssApp == "hfss_init failed":
            costValList = [costError] * len(cost_actions)
            hfss_export([0] * len(inVals), costValList, optTime, f"{fileName}_initFail")
            return costValList

        # Scale input values from [0,1] to [min, max]
        try:
            scaledVals = [
                min_val + val * (max_val - min_val)
                for val, min_val, max_val in zip(inVals, var_mins, var_maxs)
            ]
        except Exception as e:
            print(f"Error during scaling: {e}")
            costValList = [costError] * len(cost_actions)
            hfss_export([0] * len(inVals), costValList, optTime, f"{fileName}_scalingFail")
            try:
                hfssApp.save_project()
                hfssApp.release_desktop()
            except:
                pass
            return costValList

        scaledVals_with_units = [f"{val}{unit}" for val, unit in zip(scaledVals, var_units)]

        # Set the scaled parameters
        try:
            hfss_setPara(hfssApp, inNames, scaledVals_with_units)
        except Exception as e:
            print(f"Error setting parameters: {e}")
            costValList = [costError] * len(cost_actions)
            hfss_export([0]*len(scaledVals_with_units), costValList, optTime, f"{fileName}_setParaFail")
            try:
                hfssApp.save_project()
                hfssApp.release_desktop()
            except:
                pass
            return costValList

        # Run the simulation
        try:
            success = hfssApp.analyze_setup(hfssApp.setup_names[0])
            if not success:
                raise Exception("Simulation failed to run successfully.")
        except Exception as e:
            print(f"Simulation error: {e}")
            costValList = [costError] * len(cost_actions)
            hfss_export(scaledVals_with_units, costValList, optTime, f"{fileName}_solveFail")
            try:
                hfssApp.save_project()
                hfssApp.release_desktop()
            except:
                pass
            return costValList

        # Execute modification actions
        for mod_func, design_name in modification_actions:
            try:
                mod_func(hfssApp, design_name)
            except Exception as e:
                print(f"Error in modification action: {e}")
                # We continue even if a modification action fails

        # Extract and calculate costs
        for idx, (cost_func, design_name, best, worst) in enumerate(cost_actions):
            try:
                resultVal = cost_func(hfssApp, design_name)
                costValList[idx] = calcCost(resultVal, best, worst)
            except Exception as e:
                print(f"Error in cost function {cost_func.__name__}: {e}")
                costValList[idx] = costError

        # Record the end time
        optTime[1] = time.strftime("%H:%M:%S", time.localtime())

        # Export results
        hfss_export(scaledVals_with_units, costValList, optTime, fileName)

        # Save and release
        try:
            hfssApp.save_project()
        except Exception as e:
            print(f"Error saving project: {e}")
        try:
            hfssApp.release_desktop()
        except Exception as e:
            print(f"Error releasing desktop: {e}")

        # Delete the solution folder (.aedtresults)
        try:
            project_filename = os.path.splitext(os.path.basename(fileName))[0]
            solution_folder = os.path.join("optCopy", f"{project_filename}.aedtresults")
            if os.path.exists(solution_folder):
                shutil.rmtree(solution_folder)
                print(f"Deleted solution folder: {solution_folder}")
            else:
                print(f"Solution folder {solution_folder} does not exist.")
        except Exception as e:
            print(f"Error deleting solution folder: {e}")

        # Delete the solution folder (.pyaedt)
        try:
            pyaedt_folder_name = f"{project_filename.replace(' ', '_')}.pyaedt"
            pyaedt_folder = os.path.join("optCopy", pyaedt_folder_name)
            if os.path.exists(pyaedt_folder):
                shutil.rmtree(pyaedt_folder)
                print(f"Deleted pyaedt folder: {pyaedt_folder}")
            else:
                print(f"Pyaedt folder {pyaedt_folder} does not exist.")
        except Exception as e:
            print(f"Error deleting pyaedt folder: {e}")

        return costValList
    except Exception as e:
        print(f"Critical error in hfss_cost: {e}")
        # Fallback to 3 objectives if we can't get cost_actions
        costValList = [COST_ERROR] * 3
        return costValList

def hfss_export(paraVal: list, cost: list, optTime: list, fileName: str):
    """
    Export the optimization results to a CSV file.
    """
    try:
        with open('output.csv', 'a', newline='') as resultFile:
            writer = csv.writer(resultFile)
            writer.writerow(paraVal + [optTime[0], optTime[1]] + cost + [fileName])
    except Exception as e:
        print(f"Error exporting results to CSV: {e}")
    return 0

def hfss_exportInit(paraName):
    """
    Initialize the CSV file with headers.
    """
    headers = generate_csv_headers(paraName)
    try:
        with open('output.csv', 'w', newline='') as resultFile:
            writer = csv.writer(resultFile)
            writer.writerow(headers)
    except Exception as e:
        print(f"Error initializing CSV file: {e}")
    return 0


if __name__ == "__main__":
    # Add debug line
    with open('hfssOpt_debug.log', 'a') as f:
        f.write(f"==== START ====\nArgs: {sys.argv}\nTime: {time.strftime('%H:%M:%S', time.localtime())}\n")
    
    with suppress_output():
        try:
            inVals = ast.literal_eval(sys.argv[1])          # List of normalized [0,1] values
            inNames = ast.literal_eval(sys.argv[2])         # List of parameter names
            timeOut = ast.literal_eval(sys.argv[3])         # Timeout in seconds
            var_units = ast.literal_eval(sys.argv[4]) if len(sys.argv) > 4 else []
            var_mins = ast.literal_eval(sys.argv[5]) if len(sys.argv) > 5 else []
            var_maxs = ast.literal_eval(sys.argv[6]) if len(sys.argv) > 6 else []
        except Exception as e:
            # Log the error
            with open('hfssOpt_debug.log', 'a') as f:
                f.write(f"Args parsing error: {e}\n")
            sys.exit(f"Error parsing command-line arguments: {e}")

        hfssInitResult = hfss_init()

        try:
            result = func_timeout.func_timeout(
                timeOut, hfss_cost, args=[hfssInitResult, inVals, inNames, var_units, var_mins, var_maxs]
            )
        except func_timeout.FunctionTimedOut:
            try:
                if hasattr(hfssInitResult[0], 'save_project'):
                    hfssInitResult[0].save_project()
                if hasattr(hfssInitResult[0], 'release_desktop'):
                    hfssInitResult[0].release_desktop()
            except:
                pass
            hfss_export([f"0{unit}" for unit in var_units],
                       [COST_ERROR]*len(get_cost_calculation_actions()),
                       ["timeout_start", "timeout_end"],
                       "time_out")
            result = "time out"
        except Exception as e:
            # Log the error
            with open('hfssOpt_debug.log', 'a') as f:
                f.write(f"Error during execution: {e}\n")
            hfss_export([f"0{unit}" for unit in var_units],
                       [COST_ERROR]*len(get_cost_calculation_actions()),
                       ["error_start", "error_end"],
                       f"Exception: {e}")
            result = "error"

    # Log the result
    with open('hfssOpt_debug.log', 'a') as f:
        f.write(f"Result: {result}\n==== END ====\n\n")
    
    print(result)