import tkinter as tk
from tkinter import filedialog

# Prompt user to select a dataset file
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a Dataset File",
        filetypes=[("Dataset files", "*.tsv *.csv"), ("All files", "*.*")],
    )
    # Raise an error if no file is selected
    if not file_path:
        raise ValueError("No file selected. Please select a valid dataset file.")
    return file_path

print("Please upload a data file.")
dataset_path = select_file()

# Prompt the user for sample size, number of iterations, and populations
def get_user_inputs():
    try:
        # Get inputs for sample size, iterations, populations, and output file name
        sample_size = int(input("Enter the sample size for the subset of data: "))
        niterations = int(input("Enter the number of iterations for the PySR regressor: "))
        populations = int(input("Enter the number of populations for the PySR regressor: "))
        file_name = input("Enter the name of the output file: ")
        return sample_size, niterations, populations, file_name
    except ValueError:
        raise ValueError("Invalid input. Please enter numeric values for sample size, iterations, and populations.")

# Request inputs from the user
sample_size, iterations_count, populations_count, file_name = get_user_inputs()

print("Importing PySR libraries...")

# Import required libraries
import time
import os
import pandas as pd
# import numpy
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

for i in range(5):
    # Read the dataset
    df = pd.read_csv(dataset_path, sep='\s+', header=0, on_bad_lines='warn')
    # Randomly sample a subset of the data
    df_subset = df.sample(n=sample_size, random_state=23)
    # Determine the number of features (last column is the target)
    feature_count = df_subset.shape[1] - 1
    X = df_subset.iloc[:, :feature_count].values  # Input features
    y = df_subset.iloc[:, feature_count].values   # Target variable
    output_file_name = f"{file_name}_{i+1}"       # Unique output file name for each run

    # Initialize the symbolic regression model
    model = PySRRegressor(
        equation_file=f"{output_file_name}.csv",  # Output file for equations
        niterations=iterations_count,            # Number of iterations
        binary_operators=["+", "-", "*", "/"],   # Allowed binary operators
        unary_operators=["neg", "exp", "sqrt", "square"],  # Allowed unary operators
        verbosity=1,                             # Verbosity level
        populations=populations_count,           # Number of populations
        model_selection="best"                   # Selection criteria for the best model
    )

    # Measure the time taken for model training
    elapsed_time = time.perf_counter()
    model.fit(X, y)  # Train the model
    elapsed_time = time.perf_counter() - elapsed_time

    # LaTeX output file for documenting results
    latex_file_name = f"{output_file_name}.tex"

    # Create a LaTeX file to document the equations and results
    with open(latex_file_name, 'w') as tex_file:
        tex_file.write(r"\documentclass{article}" + '\n')
        tex_file.write(r"\usepackage{amsmath}" + '\n')
        tex_file.write(r"\begin{document}" + '\n')
        
        # Write the time elapsed and experiment parameters
        tex_file.write(r"\section*{Model Training Time}" + '\n')
        tex_file.write(f"Time elapsed: {elapsed_time:.2f} seconds." + '\n\n')
        tex_file.write(f"Sample size: {sample_size}." + '\n')
        tex_file.write(f"Number of iterations: {iterations_count}." + '\n')
        tex_file.write(f"Number of populations: {populations_count}." + '\n\n')

        # Document the Pareto Front equations
        tex_file.write(r"\section*{Pareto Front Equations}" + '\n')
        tex_file.write(r"\begin{align*}" + '\n')
        for i in range(len(model.equations_)):  # Loop through the Pareto front
            equation = model.latex(i)          # Get LaTeX representation of the equation
            tex_file.write(equation + r" \\" + '\n')
        tex_file.write(r"\end{align*}" + '\n\n')
        
        # Write the selected equation
        tex_file.write(r"\section*{Selected Equation}" + '\n')
        tex_file.write(r"\[ " + model.latex() + r" \]" + '\n')
        
        # End the LaTeX document
        tex_file.write(r"\end{document}" + '\n')

    print(f"LaTeX file '{latex_file_name}' has been created.")

# Clean up temporary files
os.system('del /Q *.bkup')
os.system('del /Q *.pkl')
