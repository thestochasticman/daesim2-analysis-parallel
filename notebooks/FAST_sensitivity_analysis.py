# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
## Parameters
parameter_modulepath = ["PlantCH2O.CanopyGasExchange.Leaf", "PlantCH2O.CanopyGasExchange.Leaf", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantDev"]
parameter_module = ["Leaf", "Leaf", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantDev"]
parameter_names  = ["Vcmax_opt", "g1", "SLA", "maxLAI", "ksr_coeff", "gdd_requirements"]
parameter_units  = ["mol CO2 m-2 s-1", "kPa^0.5", "m2 g d.wt-1", "m2 m-2", "g d.wt-1 m-1", "deg C d"]
parameter_init   = [60e-6, 3, 0.03, 6, 1000, 900]
parameter_min    = [30e-6, 1, 0.015, 5, 300, 600]
parameter_max    = [120e-6, 7, 0.035, 7, 5000, 1800]
parameter_phase_specific = [False, False, False, False, False, True]
parameter_phase = [None, None, None, None, None, "vegetative"] 

# Check if all parameter vectors have the same length
lengths = [len(parameter_modulepath), len(parameter_module), len(parameter_names), len(parameter_units), len(parameter_init), len(parameter_min), len(parameter_max)]
# Print result of the length check
if all(length == lengths[0] for length in lengths):
    print("All parameter vectors are of the same length.")
else:
    print("The parameter vectors are not of the same length. Lengths found:", lengths)
    
# Create a dataframe to combine the parameter information into one data structure
parameters_df = pd.DataFrame({
    "Module Path": parameter_modulepath,
    "Module": parameter_module,
    "Phase Specific": parameter_phase_specific,
    "Phase": parameter_phase,
    "Name": parameter_names,
    "Units": parameter_units,
    "Initial Value": parameter_init,
    "Min": parameter_min,
    "Max": parameter_max
})

# %%
