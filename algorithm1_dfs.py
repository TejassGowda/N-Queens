"""
=============================================================
  N-Queens Solver
  Algorithm 1: Exhaustive Depth-First Search (DFS)
=============================================================
  Course  : AI & Optimization
  Student : [Your Name] | [Matriculation Number]

  HOW TO RUN:
    python algorithm1_dfs.py
    Then enter a value for N when prompted.
=============================================================
"""

import time
import tracemalloc


def count_conflicts(queens):
    """Count diagonal conflicts in a queen placement."""
    n = len(queens)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(queens[i] - queens[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def solve_dfs(n, timeout_sec=5.0):
    """
    Solve N-Queens using exhaustive Depth-First Search
    with bitmask pruning.

    - queens[col] = row  means a queen sits in that column at that row
    - Three bitmask arrays give O(1) conflict checking per step
    - Backtracks as soon as a conflict is detected
    - Returns the FIRST valid solution found (not all solutions)

    Time complexity  : O(N!)  worst case
    Space complexity : O(N)   recursion depth + bitmasks
    """

    queens   = [-1] * n        # queens[col] = row
    row_used = [False] * n     # which rows are occupied
    diag1    = [False] * (2 * n)   # row - col  diagonal
    diag2    = [False] * (2 * n)   # row + col  diagonal

    solution = [None]
    nodes    = [0]
    start    = time.perf_counter()

    def backtrack(col):
        # All columns filled -> valid solution found
        if col == n:
            solution[0] = queens[:]
            return

        # Wall-clock timeout check
        if (time.perf_counter() - start) > timeout_sec:
            return

        for row in range(n):
            d1 = row - col + n
            d2 = row + col

            # Skip if row or either diagonal is already occupied
            if row_used[row] or diag1[d1] or diag2[d2]:
                continue

            # Place queen
            queens[col]    = row
            row_used[row]  = True
            diag1[d1]      = True
            diag2[d2]      = True
            nodes[0]      += 1

            backtrack(col + 1)

            # If solution found, stop immediately
            if solution[0] is not None:
                return

            # Remove queen (backtrack)
            queens[col]    = -1
            row_used[row]  = False
            diag1[d1]      = False
            diag2[d2]      = False

    backtrack(0)

    elapsed_ms = (time.perf_counter() - start) * 1000
    timed_out  = (solution[0] is None)

    return {
        "solution"  : solution[0],
        "time_ms"   : round(elapsed_ms, 3),
        "nodes"     : nodes[0],
        "conflicts" : 0 if solution[0] else -1,
        "timed_out" : timed_out,
    }


def estimate_memory(n, nodes):
    """Estimate memory usage in KB from data structure sizes."""
    # queens array + row_used + diag1 + diag2 arrays
    array_bytes  = n * 4          # queens[]
    array_bytes += n * 1          # row_used[]
    array_bytes += 2 * n * 1      # diag1[] + diag2[]
    # Each node explored uses roughly one stack frame
    stack_bytes  = nodes * 48     # approximate stack frame size
    total_kb     = (array_bytes + stack_bytes) / 1024
    return round(max(total_kb, n * 0.004), 2)


def print_board(queens, n):
    """Print a text chessboard showing queen positions."""
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
    print("  N-Queens  |  Algorithm 1: Exhaustive DFS")
    print("  Depth-First Search with Backtracking & Bitmask Pruning")
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

    # Set timeout based on board size
    if n <= 20:
        timeout = 10.0
    elif n <= 30:
        timeout = 8.0
    else:
        timeout = 5.0

    print(f"\n  Running DFS on N = {n}  (timeout = {timeout}s) ...")
    print()

    tracemalloc.start()
    result = solve_dfs(n, timeout_sec=timeout)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_kb   = estimate_memory(n, result["nodes"])
    peak_kb  = round(peak_mem / 1024, 2)

    print_separator()
    print(f"  RESULTS  —  N = {n}")
    print_separator()
    print(f"  Time taken      : {result['time_ms']:.3f} ms")
    print(f"  Memory (est.)   : {mem_kb} KB")
    print(f"  Memory (peak)   : {peak_kb} KB")
    print(f"  Nodes explored  : {result['nodes']:,}")

    if result["timed_out"]:
        print(f"\n  STATUS          : TIMEOUT — no solution found within {timeout}s")
        print(f"  This is expected for large N (DFS is O(N!) in the worst case)")
    else:
        print(f"  Conflicts       : {result['conflicts']}")
        print(f"  STATUS          : EXACT SOLUTION FOUND")

        # Show board only for small N (too big to print otherwise)
        if n <= 20:
            print(f"\n  Board visualisation (N={n}):")
            print_board(result["solution"], n)
            print(f"  Queen positions (col -> row):")
            for col, row in enumerate(result["solution"]):
                print(f"    Column {col+1:>3}  ->  Row {row+1}")
        else:
            print(f"\n  First 10 queen positions (col -> row):")
            for col in range(min(10, n)):
                print(f"    Column {col+1:>3}  ->  Row {result['solution'][col]+1}")
            print(f"    ... ({n - 10} more columns)")

    print_separator()
    print()
