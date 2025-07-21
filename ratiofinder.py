import cnfgen
from pysat.solvers import Solver
import time

def is_satisfiable(cnf_formula):
    """
    Checks if a CNF formula is satisfiable using the PySAT library.

    Args:
        cnf_formula: A CNF formula object from the cnfgen library.

    Returns:
        bool: True if the formula is satisfiable, False otherwise.
    """
    # Convert the cnfgen formula to the list-of-lists format PySAT expects.
    clauses = list(cnf_formula.clauses())
    with Solver(bootstrap_with=clauses) as solver:
        return solver.solve()

def calculate_satisfiability_rate(k, n, m, num_samples=500):
    """
    Calculates the percentage of satisfiable random k-CNF problems for given parameters.

    Args:
        k (int): The number of literals per clause (e.g., 3 for 3-SAT).
        n (int): The number of variables.
        m (int): The number of clauses.
        num_samples (int): The number of random problems to generate and test.

    Returns:
        float: The fraction of problems that were satisfiable (from 0.0 to 1.0).
    """
    sat_count = 0
    for _ in range(num_samples):
        cnf = cnfgen.RandomKCNF(k, n, m)
        if is_satisfiable(cnf):
            sat_count += 1
    return sat_count / num_samples

def find_optimal_m(n, k=3, target_rate=0.50, tolerance=0.02):
    """
    Finds the optimal number of clauses (m) for a given number of variables (n)
    to achieve a target satisfiability rate.

    This is done by searching for m around the known phase transition point for k-SAT.

    Args:
        n (int): The number of variables.
        k (int): The number of literals per clause.
        target_rate (float): The desired satisfiability rate (e.g., 0.5 for 50%).
        tolerance (float): The acceptable deviation from the target rate (e.g., 0.02 for ±2%).

    Returns:
        int: The optimized number of clauses (m) that meets the criteria.
    """
    # Start with a good initial guess based on the known 3-SAT phase transition ratio (~4.26)
    m = int(4.26 * n)
    
    print(f"Optimizing m for n={n}...")
    
    # Keep track of tried m values to prevent getting stuck in a two-value loop
    tried_m = set()

    # Iteratively adjust m until the satisfiability rate is within the target range
    while True:
        # If we have already tried this m value, we are in a loop.
        # Widen the search by taking a slightly larger step.
        if m in tried_m:
             print(f"   -> Stuck between values. Adjusting step.")
             m += 1 # or handle as preferred

        tried_m.add(m)

        rate = calculate_satisfiability_rate(k, n, m)
        print(f"   Trying m = {m}... Satisfiability rate: {rate:.2%}")

        if target_rate - tolerance <= rate <= target_rate + tolerance:
            print(f"✅ Success! Optimal m for n={n} is {m}")
            return m
        elif rate > (target_rate + tolerance):
            # Too easy, problems are too satisfiable. Increase constraints.
            m += 1
        else:
            # Too hard, problems are not satisfiable enough. Decrease constraints.
            m -= 1
        
        # Safety break for edge cases
        if m <= 0:
            print(f"❌ Error: Could not find an optimal m for n={n}.")
            return None


def main():
    """
    Main function to run the optimization for a range of n values and save results.
    """
    k = 3
    x_range = 31 # Corresponds to x=0 to 30
    output_filename = "optimal_pairs.csv"
    results = []
    
    total_start_time = time.time()

    for i, x in enumerate(range(x_range)):
        n = 10 + 2 * x
        
        print("\n" + "="*40)
        print(f"Processing Run {i + 1} of {x_range} | Current n = {n}")
        print("="*40)
        
        optimal_m = find_optimal_m(n, k)
        
        if optimal_m is not None:
            results.append((n, optimal_m))

    print("\n\n" + "*"*40)
    print("      🎉 All runs completed! 🎉")
    print("*"*40)

    # Save the results to a file
    try:
        with open(output_filename, 'w') as f:
            # Write a header
            f.write("n,m\n")
            for n_val, m_val in results:
                f.write(f"{n_val},{m_val}\n")
        print(f"\nSuccessfully saved {len(results)} (n, m) pairs to '{output_filename}'")
    except IOError as e:
        print(f"\nError writing to file: {e}")

    total_end_time = time.time()
    print(f"Total execution time: {(total_end_time - total_start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()