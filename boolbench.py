import os
import random
import requests
import cnfgen
import json
from pysat.solvers import Glucose3
from dotenv import load_dotenv

# global variables
model_string = "mistralai/devstral-small-2505:free"#"google/gemini-2.5-pro"#"google/gemini-2.5-flash"#"mistralai/devstral-small-2505:free"

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
    print(f"Generating a 3-SAT problem with {num_vars} variables and {num_clauses} clauses...")
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
    print(f"Problem saved to {file_path}")
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
        The following is a boolean satisfiability problem in DIMACS CNF format.
        Is this formula satisfiable? If it is, provide a satisfying assignment.
        A satisfying assignment is a list of integers where a positive integer 'v' means variable 'v' is true,
        and a negative integer '-v' means variable 'v' is false.
        For example: [1, -2, 3, -4, 5]

        Problem:
        {cnf_content}

        You must follow these rules for your response:
        If satisfiable, your answer should start with "SATISFIABLE" on the first line,
        followed by the satisfying assignment on the next line.
        If unsatisfiable, simply respond with "UNSATISFIABLE".
        No other text should be included in your response.
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
    print("Solving the problem locally with py-sat...")
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
    print("Local solver finished.")
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


# --- Main Execution ---
if __name__ == "__main__":
    # Generate the problem
    cnf_file = generate_3sat_problem(num_vars=10, num_clauses=43)
    
    # Get solutions
    llm_result = query_llm_for_solution(cnf_file)
    is_sat, model = solve_locally(cnf_file)
    
    # Verify
    verify_llm_solution(llm_result, is_sat, model)
    
