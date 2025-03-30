import hfssOpt
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from deap import algorithms, base, creator, tools
from filelock import FileLock
from math import factorial
import numpy as np
import func_timeout
import subprocess
import ast
import os
import random
import sys
import time
import json  # **Added Import**

# **Import the new function from readLegacy**
from readLegacy import generate_first_generation

random.seed("seed1")
os.chdir(os.path.dirname(__file__))

# Problem definition will be set after fetching parameters
NOBJ = 5  # Number of objectives: max_dB_S11 and db_Gain_Tot

# Algorithm parameters
P = 6  # Number of reference points per objective
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
MU = int(H + (4 - H % 4))  # Population size, adjusted to be a multiple of 4
"""
NOBJ = 5, P = 6, MU = 212
NOBJ = 5, P = 5, MU = 128
NOBJ = 4, P = 8, MU = 168
NOBJ = 3, P = 15, MU = 140
NOBJ = 2, P = 100, MU = 104
"""
NGEN = 50  # Number of generations
CXPB = 1.0  # Crossover probability
MUTPB = 1.0  # Mutation probability

COST_WHEN_ERROR = [1.1] * NOBJ  # Cost values assigned when an error occurs

# Create uniform reference points for NSGA-III
ref_points = tools.uniform_reference_points(NOBJ, P)

TIMEOUT = 3600  # Timeout for each HFSS simulation in seconds
HFSS_LAUNCH_INTERVAL = 15.0

# **Added Parameter File Path**
PARAM_FILE = 'parameters.json'

####################################################################################

def sub_withTimeout(inVals):
    """
    Wrapper function to execute the HFSS cost evaluation with a timeout.
    Reads parameter information from parameters.json each time.
    """
    try:
        # Read parameter info from parameters.json
        with open(PARAM_FILE, 'r') as f:
            params = json.load(f)
        inNames = params['optVarNames']
        var_mins = params['var_mins']
        var_maxs = params['var_maxs']
        units = params['var_units']

        return func_timeout.func_timeout(TIMEOUT + 30, hfssCost_subprocess, args=[inVals, inNames, units, var_mins, var_maxs])
    except func_timeout.FunctionTimedOut:
        hfssOpt.hfss_export([f"0{unit}" for unit in units],
                            COST_WHEN_ERROR,
                            ["timeout_start", "timeout_end"],
                            "subprocess timeout")
        return COST_WHEN_ERROR
    except Exception as e:
        hfssOpt.hfss_export([f"0{unit}" for unit in units],
                            COST_WHEN_ERROR,
                            ["error_start", "error_end"],
                            f"subprocess failed: {e}")
        return COST_WHEN_ERROR

def hfssCost_subprocess(inVals: list, inNames: list, units: list, var_mins: list, var_maxs: list) -> list:
    """
    Subprocess function to call hfssOpt.py and retrieve cost values.
    Implements file locking to prevent concurrent AEDT launches.
    """
    lockfile = 'hfss_lockfile.lock'
    timestamp_file = 'hfss_last_execution_time.txt'
    lock = FileLock(lockfile, timeout=500)

    try:
        with lock:
            # Read the last execution time
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    last_execution_time = float(f.read())
            else:
                last_execution_time = 0.0

            current_time = time.time()
            elapsed_time = current_time - last_execution_time

            if elapsed_time < HFSS_LAUNCH_INTERVAL:
                sleep_time = HFSS_LAUNCH_INTERVAL - elapsed_time
                time.sleep(sleep_time)
                current_time = time.time()

            # Update the last execution time
            with open(timestamp_file, 'w') as f:
                f.write(str(current_time))

        # Now proceed to call subprocess.run
        subOutput = subprocess.run(
            ['python', 'hfssOpt.py', str(inVals), str(inNames), str(TIMEOUT), str(units), str(var_mins), str(var_maxs)],
            capture_output=True,
            timeout=TIMEOUT + 30
        )
    except Exception as e:
        hfssOpt.hfss_export(
            [0]*len(units), COST_WHEN_ERROR,
            ["subprocess_lock_error_start", "subprocess_lock_error_end"],
            f"subprocess failed: {e}"
        )
        return COST_WHEN_ERROR

    # Parse the output from hfssOpt.py
    try:
        costList = ast.literal_eval(subOutput.stdout.decode())
    except Exception as e:
        hfssOpt.hfss_export([0]*len(units),
                            COST_WHEN_ERROR,
                            ["subprocess_parse_error_start", "subprocess_parse_error_end"],
                            "costList cannot be decoded")
        return COST_WHEN_ERROR

    if isinstance(costList, list) and all(isinstance(x, (float, int)) for x in costList):
        return costList
    else:
        if costList == "time out":
            hfssOpt.hfss_export([0]*len(units),
                                COST_WHEN_ERROR,
                                ["timeout_start", "timeout_end"],
                                "time out")
        else:
            hfssOpt.hfss_export([0]*len(units),
                                COST_WHEN_ERROR,
                                ["error_start", "error_end"],
                                "costList is not a list of float")
        return COST_WHEN_ERROR

####################################################################################
# Create DEAP classes for multi-objective minimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize the DEAP toolbox
toolbox = base.Toolbox()

# Global variables will be set inside main()
optVarNames = []
var_mins = []
var_maxs = []
var_units = []
evalFunc = None

####################################################################################
def main():
    """
    Main function to execute the NSGA-III optimization.
    """
    global optVarNames, var_mins, var_maxs, var_units, evalFunc, toolbox
    print(f"NOBJ (Number of objectives): {NOBJ}")
    print(f"P (Number of reference points per objective): {P}")
    print(f"MU (Population size): {MU}")
    
    # Check if the parameter file exists
    if os.path.exists(PARAM_FILE):
        try:
            with open(PARAM_FILE, 'r') as f:
                params = json.load(f)
            optVarNames = params['optVarNames']
            var_mins = params['var_mins']
            var_maxs = params['var_maxs']
            var_units = params['var_units']
            print("Loaded optimization parameters from parameters.json.")
        except Exception as e:
            sys.exit(f"Error reading {PARAM_FILE}: {e}")
    else:
        # Retrieve optimization parameters and save to the parameter file
        with hfssOpt.suppress_output():
            optVarNames, var_mins, var_maxs, var_units = hfssOpt.get_opt_parameters()
        params = {
            'optVarNames': optVarNames,
            'var_mins': var_mins,
            'var_maxs': var_maxs,
            'var_units': var_units
        }
        try:
            with open(PARAM_FILE, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Saved optimization parameters to {PARAM_FILE}.")
        except Exception as e:
            sys.exit(f"Error writing to {PARAM_FILE}: {e}")

    NDIM = len(optVarNames)  # Number of optimization parameters

    if NDIM == 0:
        sys.exit("No optimization parameters found. Exiting.")

    # Create variable boundaries normalized to [0,1]
    BOUND_LOW, BOUND_UP = [0.0] * NDIM, [1.0] * NDIM

    # Register DEAP toolbox components with correct dimensions and bounds
    toolbox.register("attr_float", hfssOpt.uniform, 0.0, 1.0, NDIM)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=30.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    # Initialize evaluation function
    evalFunc = sub_withTimeout
    toolbox.register("evaluate", evalFunc)

    # Initialize the CSV with headers
    hfssOpt.hfss_exportInit(optVarNames)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "avg", "std", "min", "max"]

    # =========================================================================
    # Generate the initial population, possibly from legacy.csv if it exists
    # =========================================================================
    if os.path.exists('legacy.csv'):
        try:
            legacy_init = generate_first_generation(
                MU, param_file=PARAM_FILE, legacy_file='legacy.csv',legacy_ratio=0.5
            )
            if legacy_init is not None and len(legacy_init) == MU:
                # Convert each list of floats into a DEAP Individual
                pop = [creator.Individual(ind) for ind in legacy_init]
                print("Initial population generated from legacy.csv")
            else:
                # Fallback if generation fails or does not return the correct quantity
                pop = toolbox.population(n=MU)
                print("Failed to use legacy.csv for initial population. Using random population.")
        except Exception as e:
            pop = toolbox.population(n=MU)
            print(f"Exception while using legacy.csv: {e}")
    else:
        # Generate population as usual
        pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind),
                   avg=record["avg"], std=record["std"],
                   min=record["min"], max=record["max"])
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN + 1):
        # Generate offspring
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind),
                       avg=record["avg"], std=record["std"],
                       min=record["min"], max=record["max"])
        print(logbook.stream)

    return pop, logbook

if __name__ == "__main__":
    # Initialize multiprocessing pool
    pool = Pool(6)
    toolbox.register("map", pool.map)

    # Run the optimization
    pop, log = main()

    # Extract and print fitness values of the final population
    pop_fit = np.array([ind.fitness.values for ind in pop])
    print(pop_fit)