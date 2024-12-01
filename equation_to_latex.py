import pandas as pd
import re
from sympy import symbols, sqrt, exp, latex
from sympy.parsing.sympy_parser import parse_expr
import tkinter as tk
from tkinter import filedialog

# Function to extract unique variables dynamically from equations
def extract_variables(equations):
    variables = set()
    for eq in equations:
        # Find all occurrences of variable patterns like x0, x1, etc.
        matches = re.findall(r"x\d+", eq)
        variables.update(matches)
    return {var: symbols(var) for var in variables}

# Function to preprocess equations for SymPy parsing
def preprocess_equation(eq):
    # Replace `square(...)` with `Pow(..., 2)` explicitly
    eq = re.sub(r"square\((.*?)\)", r"Pow(\1, 2)", eq)
    eq = eq.replace("^2", "**2")  # Adjust for power notation
    return eq

# Request file upload
def upload_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")],
    )
    return file_path

# Main script
if __name__ == "__main__":
    print("Please upload a CSV file containing equations.")
    input_file_path = upload_file()

    if not input_file_path:
        print("No file selected. Exiting...")
        exit()

    # Load the CSV file
    data = pd.read_csv(input_file_path)

    # Check if the file contains an 'Equation' column
    if "Equation" not in data.columns:
        raise ValueError("The uploaded file does not contain an 'Equation' column.")

    # Extract equations and dynamically find variables
    equations = data["Equation"]
    variables = extract_variables(equations)

    # Create a dictionary for known functions and map them to SymPy equivalents
    known_functions = {"Pow": lambda x, y: x**y, "sqrt": sqrt, "exp": exp}

    # Convert equations to LaTeX
    latex_equations = []
    for eq in equations:
        try:
            parsed_eq = preprocess_equation(eq)
            sympy_expr = parse_expr(parsed_eq, local_dict={**variables, **known_functions})
            latex_equations.append(latex(sympy_expr))
        except Exception as e:
            latex_equations.append(f"Error parsing equation: {eq}. Error: {str(e)}")

    # Save the LaTeX equations to a new file
    output_file_path = input_file_path.replace(".csv", "_latex_equations_fixed.txt")
    with open(output_file_path, 'w') as file:
        file.write("\n".join(latex_equations))

    print(f"LaTeX equations have been saved to {output_file_path}")
