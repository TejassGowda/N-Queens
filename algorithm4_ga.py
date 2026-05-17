"""
=============================================================
  N-Queens Solver
  Algorithm 4: Genetic Algorithm (GA)
=============================================================
  Course  : AI & Optimization
  Student : [Your Name] | [Matriculation Number]

  HOW TO RUN:
    python algorithm4_ga.py
    Then enter a value for N when prompted.
=============================================================
"""

import time
import random
import tracemalloc


def count_conflicts(queens):
    """Count total diagonal conflicts across all queen pairs."""
    n = len(queens)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(queens[i] - queens[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def fitness(queens, n):
    """
    Fitness = maximum possible non-attacking pairs minus conflicts.
    A perfect solution has fitness = N*(N-1)/2.
    """
    max_fit = n * (n - 1) // 2
    return max_fit - count_conflicts(queens)


def random_individual(n):
    """Create a random permutation — one queen per row AND per column."""
    ind = list(range(n))
    random.shuffle(ind)
    return ind


def tournament_selection(population, fitnesses, k=5):
    """
    Tournament Selection:
    Pick k random individuals, return the one with highest fitness.
    """
    best_idx  = None
    best_fit  = -1
    chosen    = random.sample(range(len(population)), k)
    for idx in chosen:
        if fitnesses[idx] > best_fit:
            best_fit  = fitnesses[idx]
            best_idx  = idx
    return population[best_idx][:]


def order_crossover(parent1, parent2, n):
    """
    Order Crossover (OX):
    Copies a random segment from parent1.
    Fills remaining positions using parent2's order,
    skipping values already placed.
    Guarantees the child is always a valid permutation.
    """
    start = random.randint(0, n - 1)
    end   = random.randint(start + 1, n)

    child = [-1] * n
    child[start:end] = parent1[start:end]

    fill_values = [v for v in parent2 if v not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_values[idx]
            idx      += 1
    return child


def swap_mutation(individual, n):
    """
    Swap Mutation:
    Randomly swap two positions in the permutation.
    Keeps the individual a valid permutation.
    """
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 2)
    if j >= i:
        j += 1
    individual[i], individual[j] = individual[j], individual[i]
    return individual


def solve_ga(n):
    """
    Solve N-Queens using a Genetic Algorithm.

    Representation:
      Each individual is a PERMUTATION of [0 .. N-1].
      individual[col] = row.
      Because it is a permutation, row and column conflicts
      are impossible — only diagonal conflicts need resolving.

    Each generation:
      1. Evaluate fitness for all individuals
      2. Copy top ELITE_RATE% unchanged (elitism)
      3. Fill the rest using:
           - Tournament selection (pick best of 5 random)
           - Order Crossover (OX) to produce child
           - Swap mutation at MUTATION_RATE probability
      4. Repeat until 0 conflicts or max generations reached

    Time complexity  : O(generations x population x N^2)
    Space complexity : O(population x N)
    """

    # ── Parameters ────────────────────────────────────────────
    pop_size    = min(max(50, 2 * n), 300)
    elite_count = max(2, int(pop_size * 0.05))
    mut_rate    = 0.08
    if   n <= 50:   max_gen = 2_000
    elif n <= 200:  max_gen = 5_000
    else:           max_gen = 10_000
    max_fit     = n * (n - 1) // 2
    # ──────────────────────────────────────────────────────────

    # Initialise population
    population = [random_individual(n) for _ in range(pop_size)]
    fitnesses  = [fitness(ind, n) for ind in population]

    best_idx  = max(range(pop_size), key=lambda i: fitnesses[i])
    best_sol  = population[best_idx][:]
    best_fit  = fitnesses[best_idx]

    gen = 0
    for gen in range(max_gen):
        if best_fit == max_fit:
            break   # perfect solution found

        # Sort by fitness descending for elitism
        sorted_idx = sorted(range(pop_size),
                            key=lambda i: fitnesses[i], reverse=True)

        new_pop = []

        # Elitism: carry top individuals unchanged
        for e in range(elite_count):
            new_pop.append(population[sorted_idx[e]][:])

        # Fill rest with crossover + mutation
        while len(new_pop) < pop_size:
            p1    = tournament_selection(population, fitnesses)
            p2    = tournament_selection(population, fitnesses)
            child = order_crossover(p1, p2, n)
            if random.random() < mut_rate:
                child = swap_mutation(child, n)
            new_pop.append(child)

        population = new_pop
        fitnesses  = [fitness(ind, n) for ind in population]

        # Track best solution
        bi = max(range(pop_size), key=lambda i: fitnesses[i])
        if fitnesses[bi] > best_fit:
            best_fit = fitnesses[bi]
            best_sol = population[bi][:]

    conflicts = max_fit - best_fit

    return {
        "solution"         : best_sol,
        "conflicts"        : conflicts,
        "generations_used" : gen + 1,
        "population_size"  : pop_size,
        "exact"            : conflicts == 0,
    }


def estimate_memory(n, pop_size):
    """Estimate memory in KB — population matrix dominates."""
    pop_kb  = (pop_size * n * 4) / 1024
    fit_kb  = (pop_size * 4) / 1024
    return round(pop_kb + fit_kb, 2)


def print_board(queens, n):
    """Print a text chessboard."""
    print()
    for row in range(n):
        line = ""
        for col in range(n):
            if queens[col] == row:
                line += " Q "
            else:
                line += " . "
        print("  " + line)
    print()


def print_separator():
    print("=" * 60)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print_separator()
    print("  N-Queens  |  Algorithm 4: Genetic Algorithm")
    print("  Order Crossover + Tournament Selection + Elitism")
    print_separator()

    # Ask user for N
    while True:
        try:
            n = int(input("\n  Enter board size N: "))
            if n < 1:
                print("  Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("  Invalid input. Please enter an integer.")

    pop_size = min(max(50, 2 * n), 300)
    if   n <= 50:  max_gen_label = "2,000"
    elif n <= 200: max_gen_label = "5,000"
    else:          max_gen_label = "10,000"

    print(f"\n  Running Genetic Algorithm on N = {n} ...")
    print(f"  Population size : {pop_size}")
    print(f"  Max generations : {max_gen_label}")
    print(f"  Mutation rate   : 8%")
    print(f"  Elitism         : Top 5%")
    print()

    tracemalloc.start()
    t0     = time.perf_counter()
    result = solve_ga(n)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_kb  = estimate_memory(n, result["population_size"])
    peak_kb = round(peak_mem / 1024, 2)

    print_separator()
    print(f"  RESULTS  —  N = {n}")
    print_separator()
    print(f"  Time taken       : {elapsed_ms:.3f} ms")
    print(f"  Memory (est.)    : {mem_kb} KB")
    print(f"  Memory (peak)    : {peak_kb} KB")
    print(f"  Generations used : {result['generations_used']:,}")
    print(f"  Population size  : {result['population_size']}")
    print(f"  Conflicts left   : {result['conflicts']}")

    if result["exact"]:
        print(f"  STATUS           : EXACT SOLUTION  (0 conflicts)")
    else:
        print(f"  STATUS           : APPROXIMATE  ({result['conflicts']} conflicts remain)")
        print(f"  NOTE: Premature convergence at large N without diversity")
        print(f"        preservation. Try running again or use SA (algo 3).")

    if n <= 20 and result["solution"]:
        print(f"\n  Board visualisation (N={n}):")
        print_board(result["solution"], n)
        print(f"  Queen positions (col -> row):")
        for col, row in enumerate(result["solution"]):
            print(f"    Column {col+1:>3}  ->  Row {row+1}")
    elif result["solution"]:
        print(f"\n  First 10 queen positions (col -> row):")
        for col in range(min(10, n)):
            print(f"    Column {col+1:>3}  ->  Row {result['solution'][col]+1}")
        print(f"    ... ({n - 10} more columns)")

    print_separator()
    print()
