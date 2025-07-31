# ─────────────────────────────────────────────────────────────────────────
#  Optimal “largest-solvable-size” search that re-uses *balancedtest* infra
# ─────────────────────────────────────────────────────────────────────────
import boolbench
import math, re, random, tempfile, argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List
import sys

number_of_seeds = 10

# ------------------------------------------------------------------------
# 1.  Parse balanced_seeds.txt   →   {(n_vars, n_clauses): [seed, seed…]}
# ------------------------------------------------------------------------
def load_balanced_seeds(path: str = "balanced_seeds.txt"
                        ) -> Dict[Tuple[int, int], List[int]]:
    """
    Returns { (num_vars, num_clauses) : [all SAT+UNSAT seeds in order] }.
    """
    blocks = defaultdict(list)
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("num_vars"):
            m = re.match(r"num_vars:\s*(\d+)\s+num_clauses:\s*(\d+)", lines[i])
            num_vars, num_clauses = map(int, m.groups())
            i += 1
            sat_seeds, unsat_seeds = [], []
            while i < len(lines) and (lines[i].startswith("SAT") or lines[i].startswith("UNSAT")):
                seeds_here = [int(s) for s in re.findall(r"-?\d+", lines[i])]
                if lines[i].startswith("SAT"):
                    sat_seeds.extend(seeds_here)
                else:
                    unsat_seeds.extend(seeds_here)
                i += 1

            # Pick equal number of SAT and UNSAT seeds, up to number_of_seeds total
            n_each = min(len(sat_seeds), len(unsat_seeds), number_of_seeds // 2)
            picked_sat = sat_seeds[:n_each]
            picked_unsat = unsat_seeds[:n_each]
            blocks[(num_vars, num_clauses)].extend(picked_sat + picked_unsat)
            print(f"Picked {len(picked_sat)} SAT and {len(picked_unsat)} UNSAT seeds for ({num_vars}, {num_clauses})")
        else:
            i += 1
    return {k: v for k, v in blocks.items() if v}  # remove empties just in case


# ------------------------------------------------------------------------
# 2.  Single-seed evaluation  (expensive call we want to minimise)
# ------------------------------------------------------------------------
def _llm_correct(num_vars: int, num_clauses: int, seed: int) -> bool:
    """
    Returns True ⟺ the LLM's answer for that *one* SAT instance is correct.
    Any ambiguity counts as a miss.
    """
    # 2.1  Re-create the exact CNF instance
    random.seed(seed)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cnf")
    cnf_path = tmp.name
    tmp.close()  # the generator writes to the same path
    boolbench.generate_3sat_problem(num_vars, num_clauses, file_path=cnf_path)

    # 2.2  Query the remote model (your existing function)
    llm_raw = boolbench.query_llm_for_solution(cnf_path)

    # 2.3  Ground-truth via the local solver
    is_sat, _ = boolbench.solve_locally(cnf_path)

    # 2.4  Decide whether the model’s claim matches ground truth
    lines = [ln.upper() for ln in llm_raw.strip().splitlines()]
    says_sat   = any("SATISFIABLE"   in ln and "UNSATISFIABLE" not in ln for ln in lines)
    says_unsat = any("UNSATISFIABLE" in ln                         for ln in lines)

    if says_sat and not says_unsat:
        claim = True
    elif says_unsat and not says_sat:
        claim = False
    else:
        return False            # ambiguous → count as wrong

    return claim == is_sat


# Convenience alias so that the search code below matches the earlier
# “query_model(problem_size, seed)” signature.
def query_model(problem_size: Tuple[int, int], seed: int) -> bool:
    return _llm_correct(problem_size[0], problem_size[1], seed)


# ------------------------------------------------------------------------
# 3.  Early-stopping accuracy test for one <num_vars, num_clauses> block
# ------------------------------------------------------------------------
def passes_threshold(problem_size: Tuple[int, int],
                     seeds: List[int],
                     threshold: float) -> bool:
    """
    Early-exit evaluation: keep calling the LLM only until the result
    is mathematically decided.
    """
    total, correct = len(seeds), 0
    for idx, s in enumerate(seeds, 1):
        if query_model(problem_size, s):
            correct += 1

        remaining = total - idx
        best_possible   = (correct + remaining) / total   # optimistic
        worst_possible  =  correct             / total   # pessimistic

        if best_possible  < threshold: return False
        if worst_possible >= threshold: return True
    return correct / total >= threshold


# ------------------------------------------------------------------------
# 4.  Exponential + binary search over all blocks
# ------------------------------------------------------------------------
def largest_solvable(blocks: Dict[Tuple[int, int], List[int]],
                     threshold: float
                     ) -> Tuple[int, int]:
    """
    Returns the *largest* (num_vars, num_clauses)   for which accuracy ≥ threshold.
    “Largest” is ordered by (num_vars, num_clauses) lexicographically.
    """
    sizes = sorted(blocks.keys())           # natural hardness proxy
    if not sizes:
        raise ValueError("No blocks found in balanced_seeds.txt")

    # 4.1  Exponential growth until first failure
    lo, hi = -1, 0
    print("Starting exponential search...")
    while hi < len(sizes) and passes_threshold(sizes[hi], blocks[sizes[hi]], threshold):
        print(f"  ✓ Passed: {sizes[hi]}")
        lo, hi = hi, (hi * 2 + 1)
    if hi < len(sizes):
        print(f"  ✗ Failed: {sizes[hi]}")
    # If we never failed, the last block works.
    if hi >= len(sizes):
        print("All blocks passed threshold. Returning largest.")
        return sizes[-1]

    # 4.2  Binary search inside (lo, hi]
    print(f"Starting binary search between {sizes[lo]} and {sizes[hi]}")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        print(f"  Testing: {sizes[mid]}...", end="")
        if passes_threshold(sizes[mid], blocks[sizes[mid]], threshold):
            print(" ✓ Passed")
            lo = mid
        else:
            print(" ✗ Failed")
            hi = mid
    print(f"Best found: {sizes[lo]}")
    return sizes[lo]


# ------------------------------------------------------------------------
# 5.  CLI hook (new “search” mode)
# ------------------------------------------------------------------------
def run_search(args):
    threshold = args.threshold
    blocks = load_balanced_seeds()
    best = largest_solvable(blocks, threshold)
    print(f"\n➜  Largest solvable block ≥ {threshold*100:.2f}% accuracy: "
          f"num_vars={best[0]}, num_clauses={best[1]}")


# Glue this into your existing argparse ‘main’…
def _augment_argparse_for_search(parser):
    sub = parser.add_parser("search", help="Find hardest size ≥ threshold")
    sub.add_argument("--threshold", type=float, default=0.841,
                     help="Required accuracy (default: 0.841)")
    sub.set_defaults(func=run_search)


# Only run if we’re executed as a script *and* “search” is requested
if __name__ == "__main__":
    # The original script already builds an ArgumentParser called `parser`.
    # We re-build it here only if none exists (stand-alone run) – otherwise
    # we patch the existing one in the global namespace.
    if "parser" not in globals():
        parser = argparse.ArgumentParser(description="BoolBench extended")
        subparsers = parser.add_subparsers(dest="mode")
        _augment_argparse_for_search(subparsers)
        # Add a default if no arguments are given (for VS Code Run/Debug)
        if len(sys.argv) == 1:
            sys.argv += ["search"]
        args = parser.parse_args()
        if hasattr(args, "func"):
            args.func(args)
        else:
            parser.print_help()
    else:
        _augment_argparse_for_search(globals()["subparsers"])
