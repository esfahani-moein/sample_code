import random
import numpy as np
import pygad
# from tqdm import tqdm
#import matplotlib.pyplot as plt
#import subprocess
#import json
#import sys
import time
#import subprocess
#from tqdm import tqdm

FNAME = [
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.1_s_100",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.0001_s_100",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.0001_s_300",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.01_s_100",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.01_s_200",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.01_s_300",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.1_s_200",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0.1_s_300",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0_s_100",
    "n_1000_c_10000000000_g_10_f_0.1_eps_0_s_200"
]
PACK_OPTIONS = [True, False]
LAST_BEST = 0
LAST_BEST_SOLUTION = None
N_SAME = 0
# JUMP_TYPE = "None"
def on_generation(ga_instance: pygad.GA):
    solutions = ga_instance.population
    all_fit_vals = ga_instance.last_generation_fitness
    sorted_solutions = [x for _, x in sorted(zip(all_fit_vals, solutions), key=lambda pair: pair[0])]
    ga_instance.population = np.array(sorted_solutions)
    ga_instance.last_generation_fitness = ga_instance.cal_pop_fitness()

def fitness_function(ga_instance: pygad.GA, solution,
                     solution_idx):  # if ga_instance.generations_completed == 0:
    total_profit = 0
    total_weight = 0
    if pack:
        solution = feasi_max(solution)
        ga_instance.population[solution_idx] = solution
    # Decode the binary solution into a list of selected item indices
    selected_items = np.where(solution == 1)[0]
    # Calculate the total profit and total weight of the selected items
    total_profit = np.sum(profits[selected_items])
    total_weight = np.sum(weights[selected_items])
    # Check if the total weight exceeds the knapsack capacity
    if total_weight > knapsack_capacity:
        # Penalize solutions that exceed the capacity by setting fitness to a very low value
        fitness = -1
    else:
        fitness = total_profit
    return fitness
def feasi_max(solution):
    items_0 = []
    items_1 = []
    for idx, item in enumerate(solution):
        if item == 1:
            items_1.append(idx)
        else:
            items_0.append(idx)
    random.shuffle(items_0)
    random.shuffle(items_1)
    items = items_1 + items_0
    total_profit = 0
    total_weight = 0
    for idx, item in enumerate(items):
        if idx < len(items_1):
            if weights[item] <= knapsack_capacity - total_weight:
                total_profit += profits[item]
                total_weight += weights[item]
            else:
                solution[item] = 0
        else:
            if weights[item] <= knapsack_capacity - total_weight:
                total_profit += profits[item]
                total_weight += weights[item]
                solution[item] = 1
    return solution
def on_fitness(ga_instance: pygad.GA, _):
    global LAST_BEST
    global LAST_BEST_SOLUTION
    global N_SAME

    best_fitness = ga_instance.best_solution()[1]
    if best_fitness <= LAST_BEST:
        N_SAME += 1
    else:
        N_SAME = 0
        LAST_BEST = best_fitness
        LAST_BEST_SOLUTION = ga_instance.best_solution()[0]

  # print(optimal_fitness - best_fitness, optimal_fitness - LAST_BEST, N_SAME)

for pack in PACK_OPTIONS:
    for fn in FNAME:
        INFILE = f"/alina-data0/Hafsa/Genetic_Algorithm/IN_{fn}.txt"
        OUTFILE =  f"/alina-data0/Hafsa/Genetic_Algorithm/OUT_{fn}.txt"
        print("Running with pack: " + str(pack) + " and File_Name: " + str(fn))

        population_size = 1000
            processes.append(process)
            print("Running with pack: " + str(pack) + " and File_Name: " + str(fn))
            # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
        population_size = 1000
        num_generations = 5
        mutation_type = "inversion"
        num_parents_mating = 500
        parent_selection_type = "tournament"
        crossover_type = "single_point"
        K_tournament = 3
        parallel_processing = 4
        mutation_probability = 0.02

        # Variables
        # to store the average differences in fitness
        average_differences = []
        # to store fitness values across generations for all runs
        all_fitness_values = []
        all_differences = []

        avg_runtime = 0
        for run in range(0, 3):
            print(f"Run: {run + 1}", end=" ")
            start_time = time.time()
            # Input file : Read input data from file
            with open(INFILE, "r") as file:
                input_data = file.read()
            data_lines = input_data.strip().split("\n")
            num_items = int(data_lines[0])
            knapsack_capacity = int(data_lines[-1])
            item_data = [list(map(int, line.split())) for line in data_lines[1:-1]]
            item_ids, profits, weights = zip(*item_data)
            item_ids = np.array(item_ids)
            profits = np.array(profits)
            weights = np.array(weights)
            # Output file : Read input data from file
            with open(OUTFILE, "r") as out_file:
                output_data = out_file.read()
            data_lines_out = output_data.strip().split("\n")
            optimal_fitness = int(data_lines_out[0])
            optimal_data = [list(map(int, line.split())) for line in data_lines_out[1:]]
            opt_profits, opt_weights = zip(*optimal_data)
            # Create an initial population ensuring the total weight does not exceed the knapsack capacity
            initial_population = []
            for i in range(population_size):
                individual = np.zeros(num_items)
                total_weight = 0
                for _ in range(num_items):
                    item_idx = np.random.randint(num_items)
                    if total_weight + weights[item_idx] <= knapsack_capacity:
                        individual[item_idx] = 1
                        total_weight += weights[item_idx]
                    else:
                        break  # Stop selecting items if adding the current item exceeds the capacity
                initial_population.append(individual)
            initial_population = np.array(initial_population)

            ga_instance = pygad.GA(
                num_generations=num_generations,
                parent_selection_type=parent_selection_type,
                crossover_type=crossover_type,
                num_parents_mating=num_parents_mating,
                fitness_func=fitness_function,
                initial_population=initial_population,
                mutation_type=mutation_type,
                parallel_processing=parallel_processing,
                on_generation=on_generation,
                on_fitness=on_fitness,
                mutation_probability=mutation_probability,
                K_tournament=K_tournament,
            # stop_criteria=["saturate_500"],
                keep_elitism=1,
            )
            # Run the optimization

                all_fitness_values.append(best_fitness)
                all_differences.append(optimal_fitness - best_fitness)
                avg_runtime += time_diff

        with open(f"Summary_all_K3.txt", "a") as o:
            max_f = max(all_fitness_values)
            min_diff = min(all_differences)
            avg_diff = sum(all_differences)/len(all_differences)
            o.write(f"Pack_{pack}_{fn}: {max_f},{min_diff}, {avg_diff}, {avg_runtime/10}\n")
