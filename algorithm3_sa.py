"""
=============================================================
  N-Queens Solver
  Algorithm 3: Simulated Annealing (SA)
=============================================================
  Course  : AI & Optimization
  Student : [Your Name] | [Matriculation Number]

  HOW TO RUN:
    python algorithm3_sa.py
    Then enter a value for N when prompted.
=============================================================
"""

import time
import math
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


def solve_sa(n):
    """
    Solve N-Queens using Simulated Annealing.

    Representation:
      queens[] is a PERMUTATION of [0 .. N-1].
      queens[col] = row.
      Because it is a permutation, row conflicts and column
      conflicts are IMPOSSIBLE by construction.
      The algorithm only needs to eliminate DIAGONAL conflicts.

    How it works:
      1. Start from a random permutation
      2. Propose a random swap of two queen positions
      3. If the swap reduces conflicts -> always accept
      4. If the swap increases conflicts -> accept with
         probability  exp(-delta / T)   (Metropolis criterion)
      5. Cool temperature T by factor alpha each step
      6. Stop when T is tiny, max steps reached, or 0 conflicts

    Time complexity  : O(max_steps x N^2)
    Space complexity : O(N)
    """

    # ── Tuning parameters ─────────────────────────────────────
    T_start  = max(2.0, 0.5 * n)
    T_min    = 0.01
    alpha    = 0.9995 if n <= 100 else 0.99998
    if   n <= 50:   max_steps = 200_000
    elif n <= 200:  max_steps = 800_000
    else:           max_steps = 3_000_000
    # ──────────────────────────────────────────────────────────

    # Start with a random permutation (one queen per row AND col)
    current = list(range(n))
    random.shuffle(current)

    cur_conflicts  = count_conflicts(current)
    best           = current[:]
    best_conflicts = cur_conflicts

    T              = T_start
    steps          = 0
    accepted_worse = 0

    while T > T_min and steps < max_steps and best_conflicts > 0:
        # Pick two random column indices to swap
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 2)
        if j >= i:
            j += 1

        # Perform swap
        current[i], current[j] = current[j], current[i]
        new_conflicts = count_conflicts(current)
        delta         = new_conflicts - cur_conflicts

        if delta < 0:
            # Improvement: always accept
            cur_conflicts = new_conflicts
            if cur_conflicts < best_conflicts:
                best_conflicts = cur_conflicts
                best           = current[:]
        elif random.random() < math.exp(-delta / T):
            # Worse: accept with Metropolis probability
            cur_conflicts  = new_conflicts
            accepted_worse += 1
        else:
            # Reject: undo swap
            current[i], current[j] = current[j], current[i]

        T     *= alpha
        steps += 1

    return {
        "solution"       : best,
        "conflicts"      : best_conflicts,
        "steps"          : steps,
        "accepted_worse" : accepted_worse,
        "final_temp"     : round(T, 6),
        "exact"          : best_conflicts == 0,
    }


def estimate_memory(n):
    """Estimate memory in KB — SA uses only 2 arrays of size N."""
    return round((n * 2 * 4) / 1024 + 0.5, 2)


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
    print("  N-Queens  |  Algorithm 3: Simulated Annealing")
    print("  Metropolis Criterion with Geometric Cooling Schedule")
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

    if   n <= 50:  max_steps_label = "200,000"
    elif n <= 200: max_steps_label = "800,000"
    else:          max_steps_label = "3,000,000"

    alpha = 0.9995 if n <= 100 else 0.99998
    T0    = max(2.0, 0.5 * n)

    print(f"\n  Running Simulated Annealing on N = {n} ...")
    print(f"  Initial temperature : {T0:.2f}")
    print(f"  Cooling factor      : {alpha}")
    print(f"  Max steps           : {max_steps_label}")
    print()

    tracemalloc.start()
    t0     = time.perf_counter()
    result = solve_sa(n)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_kb  = estimate_memory(n)
    peak_kb = round(peak_mem / 1024, 2)

    print_separator()
    print(f"  RESULTS  —  N = {n}")
    print_separator()
    print(f"  Time taken       : {elapsed_ms:.3f} ms")
    print(f"  Memory (est.)    : {mem_kb} KB")
    print(f"  Memory (peak)    : {peak_kb} KB")
    print(f"  Steps taken      : {result['steps']:,}")
    print(f"  Worse moves acc. : {result['accepted_worse']:,}")
    print(f"  Final temperature: {result['final_temp']}")
    print(f"  Conflicts left   : {result['conflicts']}")

    if result["exact"]:
        print(f"  STATUS           : EXACT SOLUTION  (0 conflicts)")
    else:
        print(f"  STATUS           : APPROXIMATE  ({result['conflicts']} conflicts remain)")
        print(f"  TIP: Run again — SA is stochastic and may find 0 conflicts next time.")

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
