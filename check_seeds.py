import re
import os
import random
from boolbench import generate_3sat_problem, solve_locally

INPUT_FILE = "balanced_seeds.txt"
TEMP_CNF_FILE = "temp_check.cnf"

# Helper to parse a line like 'SAT: 1, 2, 3, 4' into a list of ints
def parse_seed_line(line):
    return [int(s.strip()) for s in line.split(":", 1)[1].split(",") if s.strip()]

def main():
    with open(INPUT_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        # Find a block header
        if lines[i].startswith("num_vars:"):
            # Parse num_vars and num_clauses
            m = re.match(r"num_vars:\s*(\d+)\s+num_clauses:\s*(\d+)", lines[i])
            if not m:
                print(f"Could not parse header: {lines[i]}")
                i += 1
                continue
            num_vars = int(m.group(1))
            num_clauses = int(m.group(2))
            # Next lines: SAT and UNSAT
            sat_seeds = []
            unsat_seeds = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith("SAT:") or lines[j].startswith("UNSAT:")):
                if lines[j].startswith("SAT:"):
                    sat_seeds = parse_seed_line(lines[j])
                elif lines[j].startswith("UNSAT:"):
                    unsat_seeds = parse_seed_line(lines[j])
                j += 1
            # Check seeds
            sat_correct = 0
            sat_wrong = 0
            unsat_correct = 0
            unsat_wrong = 0
            for seed in sat_seeds:
                random.seed(seed)
                generate_3sat_problem(num_vars, num_clauses, TEMP_CNF_FILE)
                is_sat, _ = solve_locally(TEMP_CNF_FILE)
                if is_sat:
                    sat_correct += 1
                else:
                    sat_wrong += 1
            for seed in unsat_seeds:
                random.seed(seed)
                generate_3sat_problem(num_vars, num_clauses, TEMP_CNF_FILE)
                is_sat, _ = solve_locally(TEMP_CNF_FILE)
                if not is_sat:
                    unsat_correct += 1
                else:
                    unsat_wrong += 1
            print(f"num_vars: {num_vars} num_clauses: {num_clauses}")
            print(f"  SAT seeds:    {sat_correct} correct, {sat_wrong} wrong")
            print(f"  UNSAT seeds:  {unsat_correct} correct, {unsat_wrong} wrong\n")
            i = j
        else:
            i += 1
    # Clean up temp file
    if os.path.exists(TEMP_CNF_FILE):
        os.remove(TEMP_CNF_FILE)

if __name__ == "__main__":
    main() 