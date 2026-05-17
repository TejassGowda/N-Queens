"""
=============================================================
  N-Queens Solver
  Algorithm 2: Greedy Hill-Climbing (Local Search)
=============================================================
  Course  : AI & Optimization
  Student : [Your Name] | [Matriculation Number]

  HOW TO RUN:
    python algorithm2_greedy.py
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


def count_conflicts_for_col(queens, col):
    """Count diagonal conflicts for one specific column only."""
    n = len(queens)
    c = 0
    for j in range(n):
        if j == col:
            continue
        if abs(queens[j] - queens[col]) == abs(j - col):
            c += 1
    return c


def solve_greedy(n, max_restarts=200):
    """
    Solve N-Queens using Greedy Hill-Climbing with Random Restarts.

    How it works:
      1. Start from a random queen arrangement (one per column)
      2. For each column, try all N rows and pick the one with
         fewest diagonal conflicts  (minimum-conflict heuristic)
      3. Repeat until no improvement or max passes reached
      4. If still conflicts remain -> restart from fresh random board
      5. Repeat up to max_restarts times

    Time complexity  : O(restarts x passes x N^2)
    Space complexity : O(N)
    """

    best_solution  = None
    best_conflicts = float("inf")
    total_passes   = 0

    for restart in range(max_restarts):

        # Random starting arrangement (one queen per column, random row)
        queens = [random.randint(0, n - 1) for _ in range(n)]

        improved = True
        passes   = 0

        while improved and passes < n * 3:
            improved = False
            passes  += 1
            total_passes += 1

            for col in range(n):
                current_row      = queens[col]
                min_conflicts    = count_conflicts_for_col(queens, col)
                best_row         = current_row

                # Try every other row for this column
                for row in range(n):
                    if row == current_row:
                        continue
                    queens[col] = row
                    c = count_conflicts_for_col(queens, col)
                    if c < min_conflicts:
                        min_conflicts = c
                        best_row      = row

                # Move queen to the row with fewest conflicts
                if best_row != current_row:
                    queens[col] = best_row
                    improved    = True
                else:
                    queens[col] = current_row

        # Check total conflicts after this restart
        conf = count_conflicts(queens)
        if conf < best_conflicts:
            best_conflicts = conf
            best_solution  = queens[:]

        # Stop if we found a perfect solution
        if best_conflicts == 0:
            break

    return {
        "solution"       : best_solution,
        "conflicts"      : best_conflicts,
        "total_passes"   : total_passes,
        "restarts_used"  : restart + 1,
        "exact"          : best_conflicts == 0,
    }


def estimate_memory(n, total_passes):
    """Estimate memory in KB from data structure sizes."""
    arrays_kb  = (n * 2 * 4) / 1024   # current + best solution arrays
    passes_kb  = (total_passes * 8) / 1024
    return round(max(arrays_kb + passes_kb, 0.1), 2)


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
    print("  N-Queens  |  Algorithm 2: Greedy Hill-Climbing")
    print("  Minimum-Conflict Heuristic with Random Restarts")
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

    print(f"\n  Running Greedy Hill-Climbing on N = {n} ...")
    print(f"  (Max restarts: 200  |  Max passes per restart: {n * 3})")
    print()

    tracemalloc.start()
    t0     = time.perf_counter()
    result = solve_greedy(n, max_restarts=200)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_kb  = estimate_memory(n, result["total_passes"])
    peak_kb = round(peak_mem / 1024, 2)

    print_separator()
    print(f"  RESULTS  —  N = {n}")
    print_separator()
    print(f"  Time taken      : {elapsed_ms:.3f} ms")
    print(f"  Memory (est.)   : {mem_kb} KB")
    print(f"  Memory (peak)   : {peak_kb} KB")
    print(f"  Total passes    : {result['total_passes']:,}")
    print(f"  Restarts used   : {result['restarts_used']} / 200")
    print(f"  Conflicts left  : {result['conflicts']}")

    if result["exact"]:
        print(f"  STATUS          : EXACT SOLUTION  (0 conflicts)")
    else:
        print(f"  STATUS          : APPROXIMATE  ({result['conflicts']} conflicts remain)")
        print(f"  NOTE: Greedy gets stuck in local optima for large N.")
        print(f"        Try Simulated Annealing (algo 3) for better accuracy.")

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
