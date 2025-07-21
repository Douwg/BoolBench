import os
import random
import requests
import cnfgen
import json
from pysat.solvers import Glucose3
from dotenv import load_dotenv
import argparse
import csv
import re # Added for balancedtest mode

# global variables
model_string = "google/gemini-2.5-flash"#"google/gemini-2.5-pro"#"google/gemini-2.5-flash"#"mistralai/devstral-small-2505:free"

# --- Configuration ---
# You will need to set your OpenRouter API key as an environment variable.
# For example, in your terminal: export OPENROUTER_API_KEY='your_api_key_here'
load_dotenv()  # Load environment variables from .env file
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY is not set.")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- 1. Generate a 3-SAT Problem ---
def generate_3sat_problem(num_vars=8, num_clauses=5, file_path="problem.cnf"):
    """
    Generates a random 3-SAT problem using cnfgen and saves it to a file.

    Args:
        num_vars (int): The number of variables in the problem.
        num_clauses (int): The number of clauses in the problem.
        file_path (str): The path to save the generated .cnf file.

    Returns:
        str: The path to the saved CNF file.
    """
    #print(f"Generating a 3-SAT problem with {num_vars} variables and {num_clauses} clauses...")
    # k=3 for 3-SAT, i=num_clauses, n=num_vars
    problem = cnfgen.RandomKCNF(3, num_vars, num_clauses)
    # Write to a temporary file first
    temp_path = file_path + ".tmp"
    problem.to_file(temp_path)
    # Remove the first 6 lines (header) and save to the final file
    with open(temp_path, "r") as fin, open(file_path, "w") as fout:
        for i, line in enumerate(fin):
            if i >= 6:
                fout.write(line)
    os.remove(temp_path)
    #print(f"Problem saved to {file_path}")
    return file_path

# --- 2. Query the LLM for a Solution ---
def query_llm_for_solution(cnf_file_path):
    """
    Reads a CNF file, sends it to an LLM via OpenRouter, and asks for a solution.

    Args:
        cnf_file_path (str): Path to the DIMACS CNF file.

    Returns:
        str: The raw text response from the LLM.
    """
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY environment variable not set."

    print("Querying the LLM for a solution...")
    try:
        with open(cnf_file_path, 'r') as f:
            cnf_content = f.read()

        prompt = f"""
       You will evaluate a boolean satisfiability (SAT) problem presented in DIMACS CNF format.

        CNF FORMAT EXPLANATION:
        - Each line represents a clause (disjunction of literals)
        - Positive integers represent variables (e.g., 3 means variable x3)
        - Negative integers represent negated variables (e.g., -3 means NOT x3)
        - Each clause ends with 0
        - A clause like "1 -3 4 0" means (x1 OR NOT x3 OR x4)

        YOUR TASK:
        Determine if there exists an assignment of true/false values to all variables that makes every clause true.

        SOLUTION APPROACH:
        1. Analyze the clauses to understand the constraints
        2. Attempt to find a consistent assignment that satisfies all clauses
        3. Verify your assignment by checking each clause. 

        OUTPUT FORMAT:
        - If satisfiable: 
        Line 1: SATISFIABLE
        Line 2: [comma-separated list of literals]
        Example: SATISFIABLE
                [1, -2, 3, -4, 5]
        
        - If unsatisfiable:
        Line 1: UNSATISFIABLE

        IMPORTANT NOTES:
        - Your assignment must include ALL variables (either positive or negative)
        - Each variable appears exactly once in the assignment
        - Positive integer v means variable v is TRUE
        - Negative integer -v means variable v is FALSE
        - Do not include any explanation or additional text
        - Ensure your assignment actually satisfies all clauses before responding

        Problem:
        {cnf_content}
        """

        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_string, # You can choose any model on OpenRouter
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "tool_choice": "none"
            }
        )
        response.raise_for_status() # Raise an exception for bad status codes
        
        data = response.json()
        llm_response = data['choices'][0]['message']['content']
        print("LLM Response received")
        return llm_response

    except requests.exceptions.RequestException as e:
        return f"Error querying OpenRouter: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing LLM response: {e}"
    except FileNotFoundError:
        return f"Error: CNF file not found at {cnf_file_path}"


# --- 3. Run a Local SAT Solver ---
def solve_locally(cnf_file_path):
    """
    Solves the SAT problem using a local solver (py-sat).

    Args:
        cnf_file_path (str): Path to the DIMACS CNF file.

    Returns:
        tuple: A tuple containing (is_satisfiable, model), where model is the
               satisfying assignment if one exists.
    """
    #print("Solving the problem locally with py-sat...")
    solver = Glucose3()
    # Read the CNF file and add clauses to the solver
    with open(cnf_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("c") or line.startswith("p"):
                continue
            clause = [int(x) for x in line.split() if x != "0"]
            if clause:
                solver.add_clause(clause)
    
    is_satisfiable = solver.solve()
    model = solver.get_model() if is_satisfiable else None
    
    solver.delete()
    #print("Local solver finished.")
    return is_satisfiable, model

# --- 4. Verify the LLM's Solution ---
def verify_llm_solution(llm_response, local_is_sat, local_model):
    """
    Parses the LLM's response and verifies it against the local solver's result.

    Args:
        llm_response (str): The text response from the LLM.
        local_is_sat (bool): Whether the problem is satisfiable according to the local solver.
        local_model (list): The correct model from the local solver.
    """
    print("\n--- Verification ---")
    print(f"Local Solver Result: {'SATISFIABLE' if local_is_sat else 'UNSATISFIABLE'}")
    if local_is_sat:
        print(f"Local Solver Model: {local_model}")

    print(f"\nLLM Raw Response:\n---\n{llm_response}\n---")

    # Basic parsing of the LLM response
    lines = llm_response.strip().split('\n')

    sat_claim = any("SATISFIABLE" in line.upper() and not "UNSATISFIABLE" in line.upper() for line in lines)
    unsat_claim = any("UNSATISFIABLE" in line.upper() for line in lines)

    if sat_claim and not unsat_claim:
        llm_claims_sat = True
    elif unsat_claim and not sat_claim:
        llm_claims_sat = False
    else:
        print("\n⚠️ AMBIGUOUS RESPONSE: The LLM's response is unclear. It should clearly state either SATISFIABLE or UNSATISFIABLE.")
        return

    if not local_is_sat:
        if not llm_claims_sat:
            print("\n✅ CORRECT: The LLM correctly identified the problem as UNSATISFIABLE.")
        else:
            print("\n❌ INCORRECT: The LLM claimed the problem was SATISFIABLE, but it is UNSATISFIABLE.")
        return

    # If the problem is satisfiable
    if not llm_claims_sat:
        print("\n❌ INCORRECT: The LLM claimed the problem was UNSATISFIABLE, but it is SATISFIABLE.")
        return

    # Try to parse the model from the LLM response
    try:
        # Find the line with the model, which should contain integers
        model_line = ""
        for line in lines[1:]:
            if any(char.isdigit() for char in line):
                model_line = line
                break
        
        if not model_line:
             print("\n❌ INCORRECT: LLM claimed SATISFIABLE but did not provide a parseable model.")
             return

        # Clean up and parse the model
        model_str = model_line.replace('[', '').replace(']', '').replace(',', ' ')
        llm_model = [int(v) for v in model_str.split()]

        # Verification check
        # A valid model must satisfy all clauses. We can check this by creating a
        # temporary solver, adding the clauses, and then adding the LLM's model
        # as unit clauses to see if the formula is still solvable.
        
        verifier = Glucose3()
        #verifier.append_formula("problem.cnf")
        with open("problem.cnf", "r") as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("c") or line.startswith("p"):
                    continue
                clause = [int(x) for x in line.split() if x != "0"]
                if clause:
                    verifier.add_clause(clause)
        verifier.add_clause(llm_model) # Add the LLM's solution as a set of constraints

        if verifier.solve():
            print(f"\nLLM Parsed Model: {llm_model}")
            print("\n✅ CORRECT: The LLM found a valid satisfying assignment.")
        else:
            print(f"\nLLM Parsed Model: {llm_model}")
            print("\n❌ INCORRECT: The model provided by the LLM does not satisfy the formula.")
        
        verifier.delete()

    except (ValueError, IndexError) as e:
        print(f"\n⚠️ PARSING ERROR: Could not parse the model from the LLM's response. Error: {e}")
        print("The LLM claimed SATISFIABLE, but its formatting was incorrect.")

# --- Utility: Generate seeds for 50/50 SAT/UNSAT split ---
def generate_sat_unsat_seeds(num_vars, num_clauses, num_each, output_file):
    """
    Finds seeds that generate satisfiable and unsatisfiable problems for given num_vars and num_clauses.
    Writes the seeds to output_file in the specified format.
    """
    temp_cnf_file = "temp_problem.cnf"
    sat_seeds = []
    unsat_seeds = []
    seed = 0
    while len(sat_seeds) < num_each or len(unsat_seeds) < num_each:
        random.seed(seed)
        generate_3sat_problem(num_vars=num_vars, num_clauses=num_clauses, file_path=temp_cnf_file)
        is_sat, model = solve_locally(temp_cnf_file)
        if is_sat and len(sat_seeds) < num_each:
            sat_seeds.append(seed)
        elif not is_sat and len(unsat_seeds) < num_each:
            unsat_seeds.append(seed)
        seed += 1
    with open(output_file, "w") as f:
        f.write(f"num_vars: {num_vars} num_clauses: {num_clauses}\n")
        f.write("SAT: " + ", ".join(str(s) for s in sat_seeds) + "\n")
        f.write("UNSAT: " + ", ".join(str(s) for s in unsat_seeds) + "\n")
    print(f"Seeds written to {output_file}")

def run_single(args):
    cnf_file = generate_3sat_problem(num_vars=args.num_vars, num_clauses=args.num_clauses)
    llm_result = query_llm_for_solution(cnf_file)
    is_sat, model = solve_locally(cnf_file)
    print(f"Local solver result: {'SATISFIABLE' if is_sat else 'UNSATISFIABLE'}")
    if is_sat:
        print(f"Model: {model}")
    verify_llm_solution(llm_result, is_sat, model)

def run_seedfile(args):
    output_file = args.output_file or f"seeds_{args.num_vars}vars_{args.num_clauses}clauses.txt"
    generate_sat_unsat_seeds(num_vars=args.num_vars, num_clauses=args.num_clauses, num_each=args.num_each, output_file=output_file)

def run_seedfile_csv(args):
    with open(args.csv_file, newline='') as csvfile, open(args.output_file, "w") as outfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            try:
                n = int(row[0])
                m = int(row[1])
            except ValueError:
                print(f"Skipping invalid row: {row}")
                continue
            print(f"Generating seeds for num_vars={n}, num_clauses={m} ...")
            temp_cnf_file = "temp_problem.cnf"
            sat_seeds = []
            unsat_seeds = []
            seed = 0
            while len(sat_seeds) < args.num_each or len(unsat_seeds) < args.num_each:
                random.seed(seed)
                generate_3sat_problem(num_vars=n, num_clauses=m, file_path=temp_cnf_file)
                is_sat, model = solve_locally(temp_cnf_file)
                if is_sat and len(sat_seeds) < args.num_each:
                    sat_seeds.append(seed)
                elif not is_sat and len(unsat_seeds) < args.num_each:
                    unsat_seeds.append(seed)
                seed += 1
            outfile.write(f"num_vars: {n} num_clauses: {m}\n")
            outfile.write("SAT: " + ", ".join(str(s) for s in sat_seeds) + "\n")
            outfile.write("UNSAT: " + ", ".join(str(s) for s in unsat_seeds) + "\n")
            if idx != 0:
                outfile.write("\n")
            print(f"Seeds for num_vars={n}, num_clauses={m} written.")

def run_balancedtest(args):
    with open("balanced_seeds.txt", "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    found = False
    i = 0
    while i < len(lines):
        if lines[i].startswith("num_vars:"):
            m = re.match(r"num_vars:\s*(\d+)\s+num_clauses:\s*(\d+)", lines[i])
            if m and int(m.group(1)) == args.num_vars:
                num_clauses = int(m.group(2))
                found = True
                sat_seeds = []
                unsat_seeds = []
                j = i + 1
                while j < len(lines) and (lines[j].startswith("SAT:") or lines[j].startswith("UNSAT:")):
                    if lines[j].startswith("SAT:"):
                        sat_seeds = [int(s.strip()) for s in lines[j].split(":", 1)[1].split(",") if s.strip()]
                    elif lines[j].startswith("UNSAT:"):
                        unsat_seeds = [int(s.strip()) for s in lines[j].split(":", 1)[1].split(",") if s.strip()]
                    j += 1
                break
        i += 1
    if not found:
        print(f"No block found in balanced_seeds.txt for num_vars={args.num_vars}")
        exit(1)
    num_sat = len(sat_seeds)
    num_unsat = len(unsat_seeds)
    if args.sat_mode == "SAT":
        selected_seeds = sat_seeds[:args.num_problems]
        label = "SAT"
    elif args.sat_mode == "UNSAT":
        selected_seeds = unsat_seeds[:args.num_problems]
        label = "UNSAT"
    elif args.sat_mode == "balanced":
        n_each = args.num_problems // 2
        selected_seeds = []
        for k in range(n_each):
            if k < num_sat:
                selected_seeds.append((sat_seeds[k], "SAT"))
            if k < num_unsat:
                selected_seeds.append((unsat_seeds[k], "UNSAT"))
        if args.num_problems % 2 == 1:
            if n_each < num_sat:
                selected_seeds.append((sat_seeds[n_each], "SAT"))
            elif n_each < num_unsat:
                selected_seeds.append((unsat_seeds[n_each], "UNSAT"))
    else:
        print(f"Unknown sat_mode: {args.sat_mode}")
        exit(1)
    results = []
    if args.sat_mode == "balanced":
        for seed, label in selected_seeds:
            random.seed(seed)
            cnf_file = generate_3sat_problem(num_vars=args.num_vars, num_clauses=num_clauses, file_path="temp_balanced.cnf")
            llm_result = query_llm_for_solution(cnf_file)
            is_sat, model = solve_locally(cnf_file)
            print(f"\nSeed {seed} [{label}]:")
            verify_llm_solution(llm_result, is_sat, model)
            lines = llm_result.strip().split('\n')
            sat_claim = any("SATISFIABLE" in line.upper() and not "UNSATISFIABLE" in line.upper() for line in lines)
            unsat_claim = any("UNSATISFIABLE" in line.upper() for line in lines)
            if sat_claim and not unsat_claim:
                llm_claims_sat = True
            elif unsat_claim and not sat_claim:
                llm_claims_sat = False
            else:
                llm_claims_sat = None
            correct = (llm_claims_sat is not None) and ((llm_claims_sat and is_sat) or (not llm_claims_sat and not is_sat))
            results.append((seed, label, is_sat, llm_claims_sat, correct))
    else:
        for seed in selected_seeds:
            random.seed(seed)
            cnf_file = generate_3sat_problem(num_vars=args.num_vars, num_clauses=num_clauses, file_path="temp_balanced.cnf")
            llm_result = query_llm_for_solution(cnf_file)
            is_sat, model = solve_locally(cnf_file)
            print(f"\nSeed {seed} [{args.sat_mode}]:")
            verify_llm_solution(llm_result, is_sat, model)
            lines = llm_result.strip().split('\n')
            sat_claim = any("SATISFIABLE" in line.upper() and not "UNSATISFIABLE" in line.upper() for line in lines)
            unsat_claim = any("UNSATISFIABLE" in line.upper() for line in lines)
            if sat_claim and not unsat_claim:
                llm_claims_sat = True
            elif unsat_claim and not sat_claim:
                llm_claims_sat = False
            else:
                llm_claims_sat = None
            correct = (llm_claims_sat is not None) and ((llm_claims_sat and is_sat) or (not llm_claims_sat and not is_sat))
            results.append((seed, args.sat_mode, is_sat, llm_claims_sat, correct))
    n_correct = sum(1 for r in results if r[4])
    n_total = len(results)
    n_sat = sum(1 for r in results if r[2])
    n_unsat = n_total - n_sat
    n_sat_correct = sum(1 for r in results if r[2] and r[4])
    n_unsat_correct = sum(1 for r in results if not r[2] and r[4])
    print(f"\nSummary: LLM was correct {n_correct} / {n_total} times for num_vars={args.num_vars}, mode={args.sat_mode}")
    print(f"  SAT problems:     {n_sat_correct} / {n_sat} correct")
    print(f"  UNSAT problems:   {n_unsat_correct} / {n_unsat} correct")
    if os.path.exists("temp_balanced.cnf"):
        os.remove("temp_balanced.cnf")

def main():
    parser = argparse.ArgumentParser(description="BoolBench: SAT problem generator and benchmark tool.")
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    single_parser = subparsers.add_parser("single", help="Run the old single-problem benchmark")
    single_parser.add_argument("--num_vars", type=int, default=5, help="Number of variables")
    single_parser.add_argument("--num_clauses", type=int, default=20, help="Number of clauses")

    seedfile_parser = subparsers.add_parser("seedfile", help="Generate a text file with SAT/UNSAT seeds")
    seedfile_parser.add_argument("--num_vars", type=int, default=5, help="Number of variables")
    seedfile_parser.add_argument("--num_clauses", type=int, default=20, help="Number of clauses")
    seedfile_parser.add_argument("--num_each", type=int, default=500, help="Number of SAT and UNSAT seeds to find")
    seedfile_parser.add_argument("--output_file", type=str, default=None, help="Output file name (default: seeds_<num_vars>vars_<num_clauses>clauses.txt)")

    seedfile_csv_parser = subparsers.add_parser("seedfile_csv", help="Generate SAT/UNSAT seed files for each parameter set in a CSV file")
    seedfile_csv_parser.add_argument("--csv_file", type=str, required=True, help="CSV file with lines: num_vars,num_clauses")
    seedfile_csv_parser.add_argument("--num_each", type=int, default=500, help="Number of SAT and UNSAT seeds to find for each parameter set")
    seedfile_csv_parser.add_argument("--output_file", type=str, default="seeds_from_csv.txt", help="Output file name for all results")

    balancedtest_parser = subparsers.add_parser("balancedtest", help="Test using seeds from balanced_seeds.txt for a given num_vars")
    balancedtest_parser.add_argument("--num_vars", type=int, required=True, help="Number of variables (must match a block in balanced_seeds.txt)")
    balancedtest_parser.add_argument("--sat_mode", type=str, choices=["SAT", "UNSAT", "balanced"], required=True, help="Which type of problems to test: SAT, UNSAT, or balanced")
    balancedtest_parser.add_argument("--num_problems", type=int, default=10, help="Number of problems to test in this run")

    args = parser.parse_args()

    if args.mode == "single":
        run_single(args)
    elif args.mode == "seedfile":
        run_seedfile(args)
    elif args.mode == "seedfile_csv":
        run_seedfile_csv(args)
    elif args.mode == "balancedtest":
        run_balancedtest(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
    
