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
    if not file_path:
        raise ValueError("No file selected. Please select a valid dataset file.")
    return file_path

print("Please upload a data file.")
dataset_path = select_file()

# Prompt the user for sample size, number of iterations, and populations
def get_user_inputs():
    try:
        sample_size = int(input("Enter the sample size for the subset of data: "))
        niterations = int(input("Enter the number of iterations for the PySR regressor: "))
        populations = int(input("Enter the number of populations for the PySR regressor: "))
        output_name = input("Enter the name of the output file: ")
        return sample_size, niterations, populations, output_name
    except ValueError:
        raise ValueError("Invalid input. Please enter numeric values for sample size, iterations, and populations.")

# Request inputs
sample_size, iterations_count, populations_count, output_file_name = get_user_inputs()

print("Importing PySR libraries...")


# import pysr
import time
import sympy
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv(dataset_path, sep='\s+', header=0, on_bad_lines='warn')
df_subset = df.sample(n=sample_size, random_state=23)
feature_count = df_subset.shape[1] - 1
X = df_subset.iloc[:, :feature_count].values
y = df_subset.iloc[:, feature_count].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=23)

default_pysr_params = dict(
    populations=populations_count,
    model_selection="best",
)

# Learn equations
model = PySRRegressor(
    equation_file=f"{output_file_name}.csv",
    niterations=iterations_count,
    maxsize=30,
    nested_constraints={"sin": {"sin": 0}, "sqrt": {"sqrt": 0}},
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["neg", "exp", "sqrt", "square", "sin", "cos", "log"],
    verbosity=1,

    tempdir=None,               # Use in-memory storage for temporary directory
    temp_equation_file=None,    # Use in-memory storage for equation file
    delete_tempfiles=True,      # Ensure any residual temp files are deleted

    **default_pysr_params,
)

elapsed_time = time.perf_counter()
model.fit(X_train, y_train)
elapsed_time = time.perf_counter() - elapsed_time

time.sleep(3)

print(f"Time elapsed: {elapsed_time} seconds\n")

print("Pareto Front  \n$")
for i in range(len(model.equations_)):
  print(model.latex(i) + " \\\\")
print("$\n")

print("Selected \\ \n$")
print(model.latex())
print("$")


# LaTeX output
# After your model has been trained and you've obtained the results

# Define the name of your LaTeX file
latex_file_name = f"{output_file_name}.tex"

# Open the LaTeX file for writing
with open(latex_file_name, 'w') as tex_file:
    # Write the LaTeX document preamble
    tex_file.write(r"\documentclass{article}" + '\n')
    tex_file.write(r"\usepackage{amsmath}" + '\n')
    tex_file.write(r"\begin{document}" + '\n')
    
    # Write the time elapsed
    tex_file.write(r"\section*{Model Training Time}" + '\n')
    tex_file.write(f"Time elapsed: {elapsed_time:.2f} seconds." + '\n\n')
    
    # Write the Pareto Front equations
    tex_file.write(r"\section*{Pareto Front Equations}" + '\n')
    tex_file.write(r"\begin{align*}" + '\n')
    for i in range(len(model.equations_)):
        equation = model.latex(i)
        tex_file.write(equation + r" \\" + '\n')
    tex_file.write(r"\end{align*}" + '\n\n')
    
    # Write the selected equation
    tex_file.write(r"\section*{Selected Equation}" + '\n')
    tex_file.write(r"\[ " + model.latex() + r" \]" + '\n')
    
    # End the document
    tex_file.write(r"\end{document}" + '\n')

print(f"LaTeX file '{latex_file_name}' has been created.")


os.system('del /Q *.bkup')
os.system('del /Q *.pkl')
