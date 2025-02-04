import random
import numpy as np
import pygad
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import json
import sys
import time


# INFILE = "INSTANCES/IN_n_1000_c_10000000000_g_10_f_0.1_eps_0_s_100.txt"
# OUTFILE = "INSTANCES/OUT_n_1000_c_10000000000_g_10_f_0.1_eps_0_s_100.txt"

# python runGA -i filename -o optimalFile -f name_of_output_file -j "CliqueSNV_globalBest" -p True
INFILE = sys.argv[sys.argv.index("-i") + 1]
#OUTFILE = sys.argv[sys.argv.index("-o") + 1]
#FNAME = sys.argv[sys.argv.index("-f") + 1]

N_RUNS = 3

population_size = 1000
num_generations = 100
mutation_type = "inversion"
num_parents_mating = 500
parent_selection_type = "tournament"
crossover_type = "single_point"
K_tournament = 25
parallel_processing = 4
mutation_probability = 0.02

# Variables
average_differences = []

LAST_BEST_FITNESS = 0
LAST_BEST_SOLUTION = None
N_SAME = 0
JUMP_THRESHOLD = 10
MIN_GEN_WAIT = 20

# Options: CliqueSNV_globalBest, CliqueSNV_currentBest, reintroduce_globalBest, None
JUMP_TYPE = sys.argv[sys.argv.index("-j") + 1]
PACK_SOLUTIONS = bool(sys.argv[sys.argv.index("-p") + 1])
if JUMP_TYPE == "None":
    N_RUNS = 1

# CSNV Parameters
CSNV_TF = 0.01
CSNV_TF_INCREMENT = 0.01
CSNV_FILE_NAME = "CSNV_input"
CSNV_FASTA_PATH = CSNV_FILE_NAME + ".fas"
open(CSNV_FASTA_PATH, "w").close()
CSNV_OUT_PATH = "csnv_out/"
CSNV_TIMEOUT = 120
CSNV_EDGE_LIMIT = 1000
# CSNV_CLQ_LIMIT = 2000
CSNV_MEMORY = 100  # GB

# statistics
# NUM_CSNV_COMPLETE = 0
# NUM_CSNV_FAILED = 0
# NUM_CSNV_TIMEOUT = 0
# NUM_CSNV_EDGE_LIMIT = 0
# NUM_CSNV_EDGE_ZERO = 0
STUCK_TIMES = []
IMPROVEMENTS = []
STATS = []
NUM_CSNV_HELPED = 0
NUM_CSNV_NOT_HELPED = 0


print("recieved args: ", sys.argv)


def fitness_function(ga_instance: pygad.GA, solution, solution_idx):  #   if ga_instance.generations_completed == 0:
    total_profit = 0
    total_weight = 0
    if PACK_SOLUTIONS:
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
    global LAST_BEST_FITNESS
    global LAST_BEST_SOLUTION
    global N_SAME
    global STUCK_TIMES
    global IMPROVEMENTS
    global MIN_GEN_WAIT
    global NUM_CSNV_HELPED
    global NUM_CSNV_NOT_HELPED

    best = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    best_solution = best[0]
    best_fitness = best[1]

    if best_fitness > LAST_BEST_FITNESS:
        STUCK_TIMES.append(N_SAME)
        IMPROVEMENTS.append(best_fitness - LAST_BEST_FITNESS)
        N_SAME = 0
        LAST_BEST_FITNESS = best_fitness
        LAST_BEST_SOLUTION = best_solution
    else:
        N_SAME += 1

    print( best_fitness, LAST_BEST_FITNESS, N_SAME)

    if N_SAME >= JUMP_THRESHOLD and ga_instance.generations_completed > MIN_GEN_WAIT:
        MIN_GEN_WAIT = ga_instance.generations_completed + JUMP_THRESHOLD
        solutions = ga_instance.population
        fitnesses = ga_instance.last_generation_fitness
        sorted_solutions = [x for _, x in sorted(zip(fitnesses, solutions), key=lambda pair: pair[0])]

        if JUMP_TYPE == "cliqueSNV_globalBest":
            new_solutions = jump(solutions, LAST_BEST_SOLUTION)
        elif JUMP_TYPE == "cliqueSNV_currentBest":
            new_solutions = jump(solutions, ga_instance.best_solution()[0])
        elif JUMP_TYPE == "reintroduce_globalBest":
            new_solutions = sorted_solutions[1:]
            new_solutions.append(LAST_BEST_SOLUTION)
        elif JUMP_TYPE == "None":
            new_solutions = []
        else:
            raise Exception(f"Unknown jump type{JUMP_TYPE}")

        if len(new_solutions) > 0:
            sorted_solutions[: len(new_solutions)] = new_solutions
            ga_instance.population = np.array(sorted_solutions)
            ga_instance.last_generation_fitness = ga_instance.cal_pop_fitness()
            print("Created", len(new_solutions), "new solutions")
           # print("#\tDifference from optimal\tHamming distance from current best")
            # get them from last_generation_fitness, they are the first len(new_solutions) elements
            csnv_helped = False
            for i in range(len(new_solutions)):
                fit = ga_instance.last_generation_fitness[i]
                ham = np.sum(np.abs(new_solutions[i] - LAST_BEST_SOLUTION))
            #    print(f"{i}\t{optimal_fitness - fit}\t{ham}")
                if fit > LAST_BEST_FITNESS:
                    csnv_helped = True
            if csnv_helped:
                NUM_CSNV_HELPED += 1
            else:
                NUM_CSNV_NOT_HELPED += 1

def jump(population, best_solution):
    haplotypes = run_cliqueSNV()
    if haplotypes == None:
        return []
    new_solutions = []
    for haplotype in haplotypes:
        clique_solution = np.array([0 if nucl == "A" else 1 for nucl in haplotype["haplotype"]])
        solution = create_solution(clique_solution, best_solution)
        new_solutions.append(solution)
    return new_solutions
def create_solution(clique_solution, best_solution):
    solution = add_solution_to_clique(clique_solution, best_solution)
    # compare solution with best solution
    # diff = solution - best_solution
    # diff = np.sum(np.abs(diff))
    # print("diff", diff)
    return solution
def add_solution_to_clique(clique_solution, original_solution):
    items_0 = []
    items_1 = []
    for idx, item in enumerate(original_solution):
        if item == 1:
            items_1.append(idx)
        else:
            items_0.append(idx)
    random.shuffle(items_0)
    random.shuffle(items_1)
    items = items_1 + items_0
    current_weight = 0
    for idx, item in enumerate(clique_solution):
        if item == 1:
            current_weight += weights[idx]
    for idx, item in enumerate(items):
        if weights[item] <= knapsack_capacity - current_weight:
            if clique_solution[item] == 0:
                clique_solution[item] = 1
                current_weight += weights[item]
    return clique_solution
# def build_fasta(population):
#     with open(CSNV_FASTA_PATH, "w") as f:
#         for idx, solution in enumerate(population):
#             f.write(f">seq{idx}\n")
#             for character in solution:
#                 if character == 0:
#                     f.write("A")
#                 elif character == 1:
#                     f.write("C")
#                 else:
#                     raise Exception("Invalid character in solution")
#             f.write("\n")

SEQ_COUNTER = 0
def update_all_solutions_fasta(population):
    global SEQ_COUNTER
    with open(CSNV_FASTA_PATH, "a") as f:
        for idx, solution in enumerate(population):
            f.write(f">seq{SEQ_COUNTER}\n")
            for character in solution:
                if character == 0:
                    f.write("A")
                elif character == 1:
                    f.write("C")
                else:
                    raise Exception("Invalid character in solution")
            f.write("\n")
            SEQ_COUNTER += 1
def run_cliqueSNV():
    global STATS
    did_complete = False
    did_kill = False
    stop = False
    start_time = time.time()
    tf = CSNV_TF
    while tf < 0.2:
        cmd = f"java -Xmx{CSNV_MEMORY}G -jar clique-snv.jar -in {CSNV_FASTA_PATH} -outDir {CSNV_OUT_PATH} -m snv-pacbio -log -tf {tf}".strip().split()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Read CSNV realtime output line by line until we get to the number of edges
        while True:
            line = proc.stdout.readline()
            line = line.decode("utf-8")
            if line.startswith("Edges found"):
                num_edges = line.split(" ")[-1].strip()
                num_edges = int(num_edges)
                if num_edges > CSNV_EDGE_LIMIT:
                    print("CSNV Too many edges:", num_edges, "with tf:", tf)
                    proc.kill()
                    did_kill = True
                    STATS.append(
                        {
                            "completed": False,
                            "fail_type": "edge_limit",
                            "tf": tf,
                        }
                    )

                elif num_edges == 0:
                    print("Stopping CSNV for 0 edges with tf:", tf)
                    proc.kill()
                    stop = True
                    STATS.append(
                        {
                            "completed": False,
                            "fail_type": "edge_zero",
                            "tf": tf,
                        }
                    )
                else:
                    print("Edges:", num_edges, "with tf:", tf)
                break  # don't need to keep reading
        if stop:
            print("Stopping CSNV")
            break
        if did_kill:
            did_kill = False
            tf += CSNV_TF_INCREMENT
            continue # to next tf
        # wait for process to finish
        print("Waiting for CSNV to finish")
        try:
            out, err = proc.communicate(timeout=CSNV_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("CSNV timed out")
            STATS.append(
                {
                    "completed": False,
                    "fail_type": "timeout",
                    "tf": tf,
                }
            )
            tf += CSNV_TF_INCREMENT
            continue
        if err == b"":
            did_complete = True
            STATS.append(
                {
                    "completed": True,
                    "fail_type": None,
                    "tf": tf,
                }
            )
            end_time = time.time()
            print("CSNV completed in", end_time - start_time, "seconds")
            return parse_csnv_out()
        else:
            print(err)
            print("CSNV failed")

    if not did_complete:
        STATS.append(
            {
                "completed": False,
                "fail_type": "timeout",
                "tf": tf,
            }
        )
        return None
    return parse_csnv_out()

def run_cliqueSNV_with_retries(tf=CSNV_TF):
    print("Running CSNV")
    new_tf = tf
    while new_tf < 1:
        did_kill = False
        cmd = f"java -Xmx{CSNV_MEMORY}G -jar clique-snv.jar -in {CSNV_FASTA_PATH} -outDir {CSNV_OUT_PATH} -m snv-pacbio -log -tf {new_tf}".strip().split()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            line = proc.stdout.readline()
            line = line.decode("utf-8")
            if line.startswith("Edges found"):
                num_edges = int(line.split(" ")[-1].strip())
                if num_edges > CSNV_EDGE_LIMIT:
                    print("CSNV Too many edges:", num_edges, "with tf:", new_tf)
                    new_tf += CSNV_TF_INCREMENT
                    proc.kill()
                    did_kill = True
                    break
                elif num_edges == 0:
                    print("Stopping CSNV for 0 edges")
                    return
                else:
                    print("Edges:", num_edges, "with tf:", new_tf)
                print(line.strip())
                # break

        if did_kill:
            continue
        # wait for process to finish
        print("Waiting for CSNV to finish")
        out, err = proc.communicate()
        if err != b"":
            print(err)
            print("CSNV failed")
            return
        return parse_csnv_out()

    print("CSNV failed: No suitable tf found")
    return

def parse_csnv_out():
    with open(CSNV_OUT_PATH + CSNV_FILE_NAME + ".json", "r") as f:
        out_dict = json.load(f)
        if out_dict["errorCode"] == 5:
            print("All haplotypes too low frequency:", out_dict["settings"]["-tf"])
            return None
        return out_dict["haplotypes"]
# avg_runtime = 0
for run in range(N_RUNS):
    print(f"Run: {run+1}", end=" ")

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
    #with open(OUTFILE, "r") as out_file:
    #    output_data = out_file.read()
   # data_lines_out = output_data.strip().split("\n")
   # optimal_fitness = int(data_lines_out[0])
   # optimal_data = [list(map(int, line.split())) for line in data_lines_out[1:]]
   # opt_profits, opt_weights = zip(*optimal_data)
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
    update_all_solutions_fasta(initial_population)
    with tqdm(total=num_generations) as pbar:

        def on_generation(ga_instance):
            pbar.update(1)
            update_all_solutions_fasta(ga_instance.population)

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
            stop_criteria=["saturate_500"],
            keep_elitism=1,
        )
        # Run the optimization
        ga_instance.run()
        # Retrieve the best solution and its fitness value
        best_solution = ga_instance.best_solution()
        best_fitness = ga_instance.best_solution()[1]
    end_time = time.time()
    #add latest N_SAME to STUCK_TIMES
    STUCK_TIMES.append(N_SAME)
    # Write run report
    with open(f"Test_terminal_Res_CSNV_Unsolved_{INFILE}.txt", "a") as o:
        o.write(
            f"{INFILE}: {best_fitness}, runtime: {end_time - start_time}, mean_stuck: {sum(STUCK_TIMES)/len(STUCK_TIMES)}, max_stuck: {max(STUCK_TIMES)}, IMPROVEMENTS: {IMPROVEMENTS}, NUM_CSNV_HELPED: {NUM_CSNV_HELPED}, NUM_NOT_HELPED: {NUM_CSNV_NOT_HELPED}, {STATS}\n"
        )
        print(
            f"best: {best_fitness}, runtime: {end_time - start_time}, mean_stuck: {sum(STUCK_TIMES)/len(STUCK_TIMES)}, max_stuck: {max(STUCK_TIMES)}, IMPROVEMENTS: {IMPROVEMENTS}, NUM_CSNV_HELPED: {NUM_CSNV_HELPED}, NUM_NOT_HELPED: {NUM_CSNV_NOT_HELPED}, {STATS}\n"
        )
    N_SAME = 0
    LAST_BEST_FITNESS = 0
    LAST_BEST_SOLUTION = None
    STUCK_TIMES = []
    IMPROVEMENTS = []
    STATS = []
    open(CSNV_FASTA_PATH, "w").close()
    SEQ_COUNTER = 0
    NUM_CSNV_HELPED = 0
    NUM_CSNV_NOT_HELPED = 0
    MIN_GEN_WAIT = 20
# with open(f"BEST_{FNAME}.txt", "a") as o:
#     o.write(f"Average runtime: {avg_runtime/1} \n")
