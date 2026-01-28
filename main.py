# main.py
# An distributed optimization controller.
# This controller connects Ansys HFSS (as solver), DEAP (as optimizor), and HTCondor (for distributed computation).
# There are three key files:
# main.py
# batch_eval.py
# workflow.py
# main.py runs on the submit node, almost purely optimization algorithm. It should only be aware of the DEAP, and not aware of HFSS and HTCondor.
# The batch_eval.py runs on the submit node. It takes the data from main, synthersize condor jobs, send them to execute node in the condor cluster, and send the cost back to main.py. batch_eval.py should only be aware of HTCondor, and not aware of HFSS and DEAP.
# workflow.py runs on the execute node. It is launched by the htcondor. It runs the simulation with HFSS, save and process the resulting data. It should only be aware of HFSS, and not aware of HTCondor and DEAP.

from __future__ import annotations

import copy
import os
import random
import time
from typing import List, Tuple

from deap import algorithms, base, creator, tools

from misc import population_size
from optConfig import (
    CXPB,
    ETA_CX,
    ETA_MUT,
    EXPECT_POP_SIZE,
    MAX_GENERATION,
    MUTPB,
    NUMBER_OF_OBJECTIVES,
    P_CAP,
    RANDOM_SEED,
)


def validate_costs(costs: List[List[float]], pop_size: int, nobj: int) -> None:
    if not isinstance(costs, list) or len(costs) != pop_size:
        raise ValueError(
            f"costs shape mismatch: expected {pop_size}, got {len(costs) if isinstance(costs, list) else type(costs)}"
        )
    for i, row in enumerate(costs):
        if not isinstance(row, (list, tuple)) or len(row) != nobj:
            raise ValueError(f"costs[{i}] length mismatch: expected {nobj}, got {row}")
        for j, v in enumerate(row):
            x = float(v)
            if x < 0.0 or x > 1.0:
                raise ValueError(f"costs[{i}][{j}]={x} not in [0,1]")


def evaluate_population(pop, nobj: int) -> List[List[float]]:
    # main.py stays DEAP-centric; HTCondor is behind this one function call.
    from batch_eval import batch_evaluate

    normed_pop = [list(ind) for ind in pop]
    costs = batch_evaluate(normed_pop)
    validate_costs(costs, pop_size=len(pop), nobj=nobj)

    for ind, c in zip(pop, costs):
        ind.fitness.values = tuple(float(x) for x in c)
    return costs


def main() -> Tuple[List[List[float]], List[List[float]]]:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)

    # Parameter dimension (NDIM) is part of the optimization problem definition.
    from parameters_constraints import PARAMETERS

    ndim = len(PARAMETERS)
    nobj = int(NUMBER_OF_OBJECTIVES)

    pop_size, P = population_size(EXPECT_POP_SIZE, nobj, p_cap=P_CAP)
    ref_points = tools.uniform_reference_points(nobj, P)
    if len(ref_points) != pop_size:
        raise RuntimeError(f"NSGA-III mismatch: len(ref_points)={len(ref_points)} != pop_size={pop_size}")

    if RANDOM_SEED is not None:
        random.seed(int(RANDOM_SEED))

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * nobj)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("individual", tools.initRepeat, creator.Individual, random.random, n=ndim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=float(ETA_CX))
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=0.0,
        up=1.0,
        eta=float(ETA_MUT),
        indpb=1.0 / ndim,
    )
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    print(f"NDIM={ndim}, NOBJ={nobj}, POP_SIZE={pop_size}, MAX_GENERATION={int(MAX_GENERATION)}")

    pop = toolbox.population(n=pop_size)

    # --- NEW: smoke-test one individual before submitting the full first batch (gen=0)
    t_smoke = time.time()
    evaluate_population([pop[0]], nobj=nobj)
    print(
        f"[smoke-test] evaluated 1 individual in {time.time() - t_smoke:.1f}s "
        f"fitness={list(pop[0].fitness.values)}"
    )

    # Now evaluate the rest of generation 0 (avoids re-evaluating pop[0])
    t0 = time.time()
    if pop_size > 1:
        evaluate_population(pop[1:], nobj=nobj)
    print(f"[gen=0] evaluated in {time.time() - t0:.1f}s")

    for gen in range(1, int(MAX_GENERATION) + 1):
        parents = tools.selRandom(pop, pop_size)
        offspring = algorithms.varAnd(parents, toolbox, cxpb=float(CXPB), mutpb=float(MUTPB))

        t1 = time.time()
        evaluate_population(offspring, nobj=nobj)
        print(f"[gen={gen}] evaluated in {time.time() - t1:.1f}s")

        pop = toolbox.select(pop + offspring, pop_size)

    final_normed_pop = [list(ind) for ind in pop]
    final_costs = [list(ind.fitness.values) for ind in pop]
    return final_normed_pop, final_costs


if __name__ == "__main__":
    main()