# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FAST Sensitivity Analysis
#
# Efficiency: 
#
# FAST uses a frequency-based sampling approach that transforms the sensitivity analysis into a spectral analysis problem. This allows it to estimate first-order and total-order sensitivity indices with fewer model evaluations compared to Sobol’ sequences, which rely on a large number of Monte Carlo-style simulations to achieve similar results.
#
# - Mechanism: FAST converts the multi-dimensional sensitivity problem into a one-dimensional function using a sinusoidal transformation, making it easier to extract sensitivity indices via Fourier transformations.
# - Sample size: FAST typically requires far fewer model runs than Sobol’, especially as the number of parameters increases. For example, FAST might require on the order of hundreds of model evaluations, while Sobol’ sequences may require thousands to achieve the same level of accuracy.
# - Higher-order interactions: While FAST is efficient at capturing first-order and total-order effects, its ability to handle higher-order interactions (i.e., interactions between more than two parameters) is more limited compared to Sobol’s method, which decomposes variance at multiple levels.
#
# Advantages:
#
# - Computational efficiency: FAST often requires fewer model runs to converge on reliable sensitivity indices, making it faster for large-scale models.
# - Non-linearity: It is well-suited for models with strong non-linearities.
# - Smaller sample size: Achieves good coverage of the parameter space with fewer samples compared to Sobol’.
#
# Disadvantages:
#
# - Handling interactions: While FAST can capture total-order sensitivity (including interactions), it may not fully distinguish between individual higher-order interactions (e.g., interactions between three or more parameters).
# - Periodic inputs: FAST is better suited to continuous and periodic inputs, which can be a limitation if your model’s parameters are non-periodic or have unusual distributions.
#
# FAST is typically more computationally efficient than Sobol’ sequences for sensitivity analysis, especially when the focus is on first-order and total-order sensitivity indices. Sobol' sequences are more robust for detailed interaction analysis but are more resource-intensive.
#
# ### Step 1.  Install Required Libraries
#
# The main Python package for performing FAST is SALib (Sensitivity Analysis Library). If you don’t have it installed, you can install it via pip or conda. 

# %%
from SALib.sample import fast_sampler
from SALib.analyze import fast

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## DAESIM2
from daesim.plant_1000 import PlantModuleCalculator
from daesim.climate import *
from daesim.plantgrowthphases import PlantGrowthPhases
from daesim.management import ManagementModule
from daesim.soillayers import SoilLayers
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.boundarylayer import BoundaryLayerModule
from daesim.leafgasexchange import LeafGasExchangeModule
from daesim.leafgasexchange2 import LeafGasExchangeModule2
from daesim.canopygasexchange import CanopyGasExchange
from daesim.plantcarbonwater import PlantModel as PlantCH2O
from daesim.plantallocoptimal import PlantOptimalAllocation
from daesim.utils import ODEModelSolver

## DAESIM2 Analysis
from daesim2_analysis import fast_sensitivity as fastsa

## Argument Parsing
import __main__
from argparse import ArgumentParser

## Parallelisation
from typing_extensions import Callable
from typing_extensions import Self
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional
from typing import List
from os import makedirs
from tqdm import tqdm
from time import time

### Parse Arguments if running as python file
## Argument Parsing
def is_interactive():
    """
    Check if the script is running in an interactive environment.
    """
    try:
        return hasattr(__main__, '__file__') == False
    except AttributeError:
        return True

IS_INTERACTIVE = is_interactive()

if not IS_INTERACTIVE:
  parser = ArgumentParser()
  group1 = parser.add_argument_group('Optimisation Arguments')
  group1.add_argument(
      '--n_processes',
      type=int,
      required=True,
      help='number of processes to use for FAST SA'
  )
  group1.add_argument(
      '--n_samples',
      type=int,
      required=True,
      help='number of samples to generate for FAST SA'
  )

  group2 = parser.add_argument_group('File Arguments')
  group1.add_argument(
      '--dir_results',
      type=str,
      required=True,
      help="Path to the directory for storing FAST SA results"
  )
  group2.add_argument(
      '--xsite',
      type=str,
      required=True,
      help='name of site'
  )
  group2.add_argument(
      '--paths_df_forcing',
      nargs='+',
      required=True,
      help='list of paths to files containing forcing data'
  )

  args = parser.parse_args()

  n_processes     : int = args.n_processes
  n_samples       : int = args.n_samples
  dir_results     : str = args.dir_results
  xsite           : str = args.xsite
  paths_df_forcing: str = args.paths_df_forcing
  n_samples       : int = args.n_samples
else:
  n_processes     : int = 4
  n_samples       : int = 600
  dir_results     : str = '/g/data/xe2/ya6227/DAESIM/results/'
  xsite           : str = 'Milgadara_2018_test'
  paths_df_forcing: str = [
                            '/g/data/xe2/ya6227/DAESIM/example_dfs/DAESim_forcing_Milgadara_2018.csv',
                            '/g/data/xe2/ya6227/DAESIM/example_dfs/DAESim_forcing_Milgadara_2019.csv'
                          ]

dir_xsite_results = ''.join([dir_results, xsite])
filepath_write = ''.join([dir_xsite_results, 'parameters/'])
makedirs(filepath_write, exist_ok=True)
path_mpx = ''.join([filepath_write, xsite + '_Mpx'])
print(path_mpx)
# # %% [markdown]
# # ### Step 2. Define the model problem
# #
# # Need to define the number of parameters, their names, and the ranges (bounds) over which the sensitivity analysis will be performed. For example, if your model has three parameters (e.g., param1, param2, param3), you would define the problem in the following format:
# #
# # This includes both model parameters (e.g. here this includes Vcmax_opt, Jmax_opt, g1) and the forcing data ranges (e.g. here this includes T, Q, fgsw).

# # %%
# ## Parameters
# # parameter_modulepath = ["PlantCH2O.CanopyGasExchange.Leaf", "PlantCH2O.CanopyGasExchange.Leaf", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantDev", "PlantDev", "", ""]
# # parameter_module = ["Leaf", "Leaf", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantDev", "PlantDev", "", ""]
# # parameter_names  = ["Vcmax_opt", "g1", "SLA", "maxLAI", "ksr_coeff", "Psi_f", "gdd_requirements", "gdd_requirements", "GY_FE", "GY_SDW_50"]
# # parameter_units  = ["mol CO2 m-2 s-1", "kPa^0.5", "m2 g d.wt-1", "m2 m-2", "g d.wt-1 m-1", "MPa", "deg C d", "deg C d", "thsnd grains g d.wt spike-1", "g d.wt m-2"]
# # parameter_init   = [60e-6, 3, 0.03, 6, 1000, -1.5, 900, 650, 0.1, 100]
# # parameter_min    = [30e-6, 1, 0.015, 5, 300, -4.0, 600, 350, 0.08, 80]
# # parameter_max    = [120e-6, 6, 0.035, 7, 5000, -1.0, 1800, 700, 0.21, 150]
# # parameter_phase_specific = [False, False, False, False, False, False, True, True, False, False]
# # parameter_phase = [None, None, None, None, None, None, "vegetative", "grainfill", None, None]

# parameter_modulepath = ["PlantCH2O.CanopyGasExchange.Leaf", "PlantCH2O.CanopyGasExchange.Leaf", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantDev", "PlantDev", "", "", "", ""]
# parameter_module = ["Leaf", "Leaf", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantCH2O", "PlantDev", "PlantDev", "", "", "", ""]
# parameter_names  = ["Vcmax_opt", "g1", "SLA", "maxLAI", "ksr_coeff", "Psi_f", "sf", "gdd_requirements", "gdd_requirements", "GY_FE", "GY_SDW_50", "CI", "d_r_max"]
# parameter_units  = ["mol CO2 m-2 s-1", "kPa^0.5", "m2 g d.wt-1", "m2 m-2", "g d.wt-1 m-1", "MPa", "MPa-1", "deg C d", "deg C d", "thsnd grains g d.wt spike-1", "g d.wt m-2", "-", "m"]
# parameter_init   = [60e-6, 3, 0.03, 6, 1000, -3.5, 3.5, 900, 650, 0.1, 100, 0.75, 0.5]
# parameter_min    = [30e-6, 1, 0.015, 5, 300, -8.0, 1.5, 600, 350, 0.08, 80, 0.5, 0.15]
# parameter_max    = [120e-6, 6, 0.035, 7, 5000, -1.0, 7.0, 1800, 700, 0.21, 150, 1.0, 0.66]
# parameter_phase_specific = [False, False, False, False, False, False, False, True, True, False, False, False, False]
# parameter_phase = [None, None, None, None, None, None, None, "vegetative", "grainfill", None, None, None, None]

# # %%
# # Check if all parameter vectors have the same length
# lengths = [len(parameter_modulepath), len(parameter_module), len(parameter_names), len(parameter_units), len(parameter_init), len(parameter_min), len(parameter_max)]
# # Print result of the length check
# if all(length == lengths[0] for length in lengths):
#     print("All parameter vectors are of the same length.")
# else:
#     print("The parameter vectors are not of the same length. Lengths found:", lengths)
    
# # Create a dataframe to combine the parameter information into one data structure
# parameters_df = pd.DataFrame({
#     "Module Path": parameter_modulepath,
#     "Module": parameter_module,
#     "Phase Specific": parameter_phase_specific,
#     "Phase": parameter_phase,
#     "Name": parameter_names,
#     "Units": parameter_units,
#     "Initial Value": parameter_init,
#     "Min": parameter_min,
#     "Max": parameter_max
# })

# # Dictionary required as input to SALib FAST sampler
# problem = {
#     "num_vars": len(parameters_df),    # Number of input parameters
#     "names": parameters_df["Name"].values,   # Parameter names
#     "bounds": [[row['Min'], row['Max']] for _, row in parameters_df.iterrows()],    # Parameter ranges
# }

# # %% [markdown]
# # ### Step 3. Generate the FAST Samples
# #
# # To apply the FAST method, you need to generate the sampling points for the input parameters using the SALib.sample.fast function. Here, you specify the total number of samples you want (it is important to choose a sufficiently large number to ensure the analysis is robust). 
# #
# # N.B. For a typical FAST setup:
# #
# # - num_samples refers to the number of frequencies used to sample each parameter.
# # - The actual number of parameter sets generated depends on num_vars and the sampling process.
# #
# # The total number of samples is typically num_samples * num_vars, and thus, Y should have this size. You can adjust the setup to reflect this. 

# # %%
# # Generate samples using the FAST method
# # ### Step 3. Generate the FAST Samples
# #
# # To apply the FAST method, you need to generate the sampling points for the input parameters using the SALib.sample.fast function. Here, you specify the total number of samples you want (it is important to choose a sufficiently large number to ensure the analysis is robust). 
# #
# # N.B. For a typical FAST setup:
# #
# # - num_samples refers to the number of frequencies used to sample each parameter.
# # - The actual number of parameter sets generated depends on num_vars and the sampling process.
# #
# # The total number of samples is typically num_samples * num_vars, and thus, Y should have this size. You can adjust the setup to reflect this.
 
# # %%
# # Generate samples using the FAST method
# param_values = fast_sampler.sample(problem, n_samples, seed=0)

# # param_values will contain the sampled input sets for the parameters
# print(param_values.shape)

# # %% [markdown]
# # ### Step 4. Initialise the study site and import forcing data

# # %%
# def load_df_forcing(paths_df_forcing: List[str])->pd.DataFrame:
#     dfs: List[pd.DataFrame] = []
#     for path_df_forcing in paths_df_forcing:
#         df = pd.read_csv(path_df_forcing)
#         df['Date'] = pd.to_datetime(df['Date'])
#         dfs += [df]
#     df_forcing = pd.concat(dfs)
#     df_forcing = df_forcing.drop_duplicates(subset='Date')
#     df_forcing = df_forcing.sort_values(by='Date')
#     df_forcing = df_forcing.reset_index(drop=True)
#     df_forcing['DOY'] = df_forcing['Date'].dt.dayofyear
#     df_forcing['Year'] = df_forcing['Date'].dt.year
#     return df_forcing


# df_forcing = load_df_forcing(paths_df_forcing=paths_df_forcing)

# # df_forcing1 = pd.read_csv(paths_df_forcing[0])
# # df_forcing2 = pd.read_csv(paths_df_forcing[1])

# # df_forcing1['Date'] = pd.to_datetime(df_forcing1['Date'])
# # df_forcing2['Date'] = pd.to_datetime(df_forcing2['Date'])

# # # Append the new dataframe and drop duplicates based on the "Date" column
# # df_forcing = pd.concat([df_forcing1, df_forcing2]).drop_duplicates(subset="Date").sort_values(by="Date")

# # # Reset the index for the combined dataframe (optional)
# # df_forcing.reset_index(drop=True, inplace=True)

# # # Add ordinal day of year (DOY) and Year variables
# # df_forcing["DOY"] = df_forcing["Date"].dt.dayofyear
# # df_forcing["Year"] = df_forcing["Date"].dt.year


# # %%
# print(df_forcing)
# # %%
# ## Interpolate discrete soil moisture data
# # df_forcing["Soil moisture interp"] = df_forcing["Soil moisture"].interpolate('quadratic')

# # ## Assume that the forcing data (units: mm) can be equated to relative changes in volumetric 
# # # between two arbitrary minimum and maximum values
# # f_soilTheta_min = 0.25
# # f_soilTheta_max = 0.40

# # f_soilTheta_min_mm = df_forcing["Soil moisture interp"].min()
# # f_soilTheta_max_mm = df_forcing["Soil moisture interp"].max()

# # f_soilTheta_norm_mm = (df_forcing["Soil moisture interp"].values - f_soilTheta_min_mm)/(f_soilTheta_max_mm - f_soilTheta_min_mm)
# # f_soilTheta_norm = f_soilTheta_min + f_soilTheta_norm_mm * (f_soilTheta_max - f_soilTheta_min)


# # ## Milgadara site location-34.38904277303204, 148.46949938279096
# # SiteX = ClimateModule(CLatDeg=-34.389,CLonDeg=148.469,timezone=10)
# # start_doy = df_forcing["DOY"].values[0]    # 1.0
# # start_year = df_forcing["Year"].values[0]   # 2021
# # nrundays = df_forcing.index.size

# # ## Time discretisation
# # time_nday, time_doy, time_year = SiteX.time_discretisation(start_doy, start_year, nrundays=nrundays)
# # ## Adjust daily time-step to represent midday on each day
# # time_doy = [time_doy[i]+0.5 for i in range(len(time_doy))]

# # %%
# SiteX = ClimateModule(CLatDeg=-36.05,CLonDeg=146.5,timezone=10)
# start_doy_f = df_forcing["DOY"].values[0]
# start_year_f = df_forcing["Year"].values[0]
# nrundays_f = df_forcing.index.size

# ## Time discretisation
# time_nday_f, time_doy_f, time_year_f = SiteX.time_discretisation(start_doy_f, start_year_f, nrundays=nrundays_f)
# ## Adjust daily time-step to represent midday on each day
# time_doy_f = [time_doy_f[i]+0.5 for i in range(len(time_doy_f))]

# ## Time discretization for forcing data
# time_index_f = pd.to_datetime(df_forcing["Date"].values)

# # Define lists of sowing and harvest dates
# sowing_dates = [
#     pd.Timestamp(year=1971, month=5, day=11)
# ]

# harvest_dates = [
#     pd.Timestamp(year=1971, month=12, day=23)
# ]

# _soilTheta_z = np.column_stack((
#     df_forcing["Soil moisture 5 cm"].values,
#     df_forcing["Soil moisture 8 cm"].values,
#     df_forcing["Soil moisture 14 cm"].values,
#     df_forcing["Soil moisture 22 cm"].values,
#     df_forcing["Soil moisture 34 cm"].values,
#     df_forcing["Soil moisture 52 cm"].values,))

# Climate_doy_f = interp_forcing(time_nday_f, time_doy_f, kind="pconst") #, fill_value=(time_doy[0],time_doy[-1]))
# Climate_year_f = interp_forcing(time_nday_f, time_year_f, kind="pconst") #, fill_value=(time_year[0],time_year[-1]))
# Climate_airTempCMin_f = interp1d(time_nday_f, df_forcing["Minimum temperature"].values)
# Climate_airTempCMax_f = interp1d(time_nday_f, df_forcing["Maximum temperature"].values)
# Climate_airTempC_f = interp1d(time_nday_f, (df_forcing["Minimum temperature"].values+df_forcing["Maximum temperature"].values)/2)
# Climate_solRadswskyb_f = interp1d(time_nday_f, 10*(df_forcing["Global Radiation"].values-df_forcing["Diffuse Radiation"].values))
# Climate_solRadswskyd_f = interp1d(time_nday_f, 10*df_forcing["Diffuse Radiation"].values)
# Climate_airPressure_f = interp1d(time_nday_f, 100*df_forcing["Pressure"].values)
# Climate_airRH_f = interp1d(time_nday_f, df_forcing["Relative Humidity"].values)
# Climate_airU_f = interp1d(time_nday_f, df_forcing["Uavg"].values)
# Climate_airCO2_f = interp1d(time_nday_f, df_forcing["Atmospheric CO2 Concentration (bar)"].values)
# Climate_airO2_f = interp1d(time_nday_f, df_forcing["Atmospheric O2 Concentration (bar)"].values)
# Climate_soilTheta_z_f = interp1d(time_nday_f, _soilTheta_z, axis=0)  # Interpolates across timesteps, handles all soil layers at once
# Climate_nday_f = interp1d(time_nday_f, time_nday_f)   ## nday represents the ordinal day-of-year plus each simulation day (e.g. a model run starting on Jan 30 and going for 2 years will have nday=30+np.arange(2*365))

# # %% [markdown]
# # #### b. Rutherglen Site 1971

# # %%

# SiteX = ClimateModule(CLatDeg=-36.05,CLonDeg=146.5,timezone=10)
# start_doy_f = df_forcing["DOY"].values[0]
# start_year_f = df_forcing["Year"].values[0]
# nrundays_f = df_forcing.index.size

# ## Time discretisation
# time_nday_f, time_doy_f, time_year_f = SiteX.time_discretisation(start_doy_f, start_year_f, nrundays=nrundays_f)
# ## Adjust daily time-step to represent midday on each day
# time_doy_f = [time_doy_f[i]+0.5 for i in range(len(time_doy_f))]

# ## Time discretization for forcing data
# time_index_f = pd.to_datetime(df_forcing["Date"].values)

# # Define lists of sowing and harvest dates
# sowing_dates = [
#     pd.Timestamp(year=1971, month=5, day=11)
# ]

# harvest_dates = [
#     pd.Timestamp(year=1971, month=12, day=23)
# ]

# _soilTheta_z = np.column_stack((
#     df_forcing["Soil moisture 5 cm"].values,
#     df_forcing["Soil moisture 8 cm"].values,
#     df_forcing["Soil moisture 14 cm"].values,
#     df_forcing["Soil moisture 22 cm"].values,
#     df_forcing["Soil moisture 34 cm"].values,
#     df_forcing["Soil moisture 52 cm"].values,))

# Climate_doy_f = interp_forcing(time_nday_f, time_doy_f, kind="pconst") #, fill_value=(time_doy[0],time_doy[-1]))
# Climate_year_f = interp_forcing(time_nday_f, time_year_f, kind="pconst") #, fill_value=(time_year[0],time_year[-1]))
# Climate_airTempCMin_f = interp1d(time_nday_f, df_forcing["Minimum temperature"].values)
# Climate_airTempCMax_f = interp1d(time_nday_f, df_forcing["Maximum temperature"].values)
# Climate_airTempC_f = interp1d(time_nday_f, (df_forcing["Minimum temperature"].values+df_forcing["Maximum temperature"].values)/2)
# Climate_solRadswskyb_f = interp1d(time_nday_f, 10*(df_forcing["Global Radiation"].values-df_forcing["Diffuse Radiation"].values))
# Climate_solRadswskyd_f = interp1d(time_nday_f, 10*df_forcing["Diffuse Radiation"].values)
# Climate_airPressure_f = interp1d(time_nday_f, 100*df_forcing["Pressure"].values)
# Climate_airRH_f = interp1d(time_nday_f, df_forcing["Relative Humidity"].values)
# Climate_airU_f = interp1d(time_nday_f, df_forcing["Uavg"].values)
# Climate_airCO2_f = interp1d(time_nday_f, df_forcing["Atmospheric CO2 Concentration (bar)"].values)
# Climate_airO2_f = interp1d(time_nday_f, df_forcing["Atmospheric O2 Concentration (bar)"].values)
# Climate_soilTheta_z_f = interp1d(time_nday_f, _soilTheta_z, axis=0)  # Interpolates across timesteps, handles all soil layers at once
# Climate_nday_f = interp1d(time_nday_f, time_nday_f)   ## nday represents the ordinal day-of-year plus each simulation day (e.g. a model run starting on Jan 30 and going for 2 years will have nday=30+np.arange(2*365))

# # %% [markdown]
# # ### Step 4. Initialise the Model

# # %%
# ## Check that the first sowing date and last harvest date are within the available forcing data period
# # Apply validation to sowing and harvest dates
# SiteX.validate_event_dates(sowing_dates, time_index_f, event_name="Sowing")
# SiteX.validate_event_dates(harvest_dates, time_index_f, event_name="Harvest")

# # Find steps for all sowing and harvest dates in the forcing data
# sowing_steps_f = SiteX.find_event_steps(sowing_dates, time_index_f)
# harvest_steps_f = SiteX.find_event_steps(harvest_dates, time_index_f)

# # 3. Generate time axis for ODE Solver, which is set to start at the first sowing event and end just after the last harvest event
# time_axis = time_nday_f[sowing_steps_f[0]:harvest_steps_f[-1]+2]
# # Corresponding date time index
# time_index = time_index_f[sowing_steps_f[0]:harvest_steps_f[-1]+2]

# ## Determine the ODE solver reset days
# sowing_steps_itax = SiteX.find_event_steps(sowing_dates, time_index)
# harvest_steps_itax = SiteX.find_event_steps(harvest_dates, time_index)
# reset_days_itax = SiteX.find_event_steps(sowing_dates+harvest_dates, time_index)
# reset_days_itax.sort()
# reset_days_tax = list(time_axis[reset_days_itax])

# ## Determine the sowing and harvest days and years for the DAESIM2 Management module
# sowingDays = np.floor(Climate_doy_f(time_axis[sowing_steps_itax]))
# sowingYears = np.floor(Climate_year_f(time_axis[sowing_steps_itax]))
# harvestDays = np.floor(Climate_doy_f(time_axis[harvest_steps_itax]))
# harvestYears = np.floor(Climate_year_f(time_axis[harvest_steps_itax]))

# # %%
# ManagementX = ManagementModule(cropType="Wheat",sowingDays=sowingDays,harvestDays=harvestDays,sowingYears=sowingYears,harvestYears=harvestYears)

# PlantDevX = PlantGrowthPhases(
#     phases=["germination", "vegetative", "spike", "anthesis", "grainfill", "maturity"],
#     # gdd_requirements=[120,800,250,200,300,200],
#     gdd_requirements=[50,800,280,150,300,300],
#     # vd_requirements=[0, 40, 0, 0, 0, 0],
#     vd_requirements=[0, 30, 0, 0, 0, 0],
#     allocation_coeffs = [
#         [0.2, 0.1, 0.7, 0.0, 0.0],
#         [0.5, 0.1, 0.4, 0.0, 0.0],
#         [0.30, 0.4, 0.30, 0.0, 0.0],
#         [0.30, 0.4, 0.30, 0.0, 0.0],
#         [0.1, 0.1, 0.1, 0.7, 0.0],
#         [0.1, 0.1, 0.1, 0.7, 0.0]
#     ],
#     turnover_rates = [[0.001,  0.001, 0.001, 0.0, 0.0],
#                       [0.01, 0.002, 0.008, 0.0, 0.0],
#                       [0.01, 0.002, 0.008, 0.0, 0.0],
#                       [0.01, 0.002, 0.008, 0.0, 0.0],
#                       [0.033, 0.016, 0.033, 0.0002, 0.0],
#                       [0.10, 0.033, 0.10, 0.0002, 0.0]])

# BoundLayerX = BoundaryLayerModule(Site=SiteX)
# LeafX = LeafGasExchangeModule2(Site=SiteX,Jmax_opt_rVcmax=0.89,Jmax_opt_rVcmax_method="log")
# CanopyX = CanopyLayers(nlevmlcan=3)
# CanopyRadX = CanopyRadiation(Canopy=CanopyX)
# CanopyGasExchangeX = CanopyGasExchange(Leaf=LeafX,Canopy=CanopyX,CanopyRad=CanopyRadX)
# # SoilLayersX = SoilLayers(nlevmlsoil=2,z_max=2.0)
# SoilLayersX = SoilLayers(nlevmlsoil=6,z_max=0.66,z_top=0.10,discretise_method="horizon",
#                          z_horizon=[0.06, 0.06, 0.06, 0.10, 0.10, 0.28],
#                         Psi_e=[-1.38E-03, -1.38E-03, -1.38E-03, -1.32E-03, -2.58E-03, -0.960E-03],
#                         b_soil = [4.74, 4.74, 4.74, 6.77, 8.17, 10.73],
#                          K_sat = [29.7, 29.7, 29.7, 25.2, 13.9, 40.9],
#                          soilThetaMax = [0.12, 0.12, 0.12, 0.20, 0.3, 0.4])
# # PlantCH2OX = PlantCH2O(Site=SiteX,SoilLayers=SoilLayersX,CanopyGasExchange=CanopyGasExchangeX,BoundaryLayer=BoundLayerX,maxLAI=6.0,ksr_coeff=1000,SLA=0.030)
# PlantCH2OX = PlantCH2O(Site=SiteX,SoilLayers=SoilLayersX,CanopyGasExchange=CanopyGasExchangeX,BoundaryLayer=BoundLayerX,maxLAI=6.5,ksr_coeff=1500,SLA=0.02,sf=1.0,Psi_f=-5.0)
# PlantAllocX = PlantOptimalAllocation(Plant=PlantCH2OX,dWL_factor=1.02,dWR_factor=1.02)
# PlantX = PlantModuleCalculator(
#     Site=SiteX,
#     Management=ManagementX,
#     PlantDev=PlantDevX,
#     PlantCH2O=PlantCH2OX,
#     PlantAlloc=PlantAllocX,
#     GDD_method="nonlinear",
#     GDD_Tbase=0.0,
#     GDD_Topt=22.5,
#     GDD_Tupp=35.0,
#     hc_max_GDDindex=sum(PlantDevX.gdd_requirements[0:2])/PlantDevX.totalgdd,
#     d_r_max=2.0,
#     Vmaxremob=3.0,
#     Kmremob=0.5,
#     remob_phase=["grainfill","maturity"],
#     specified_phase="spike",
#     grainfill_phase=["grainfill","maturity"],
# )

# # %%
# ## Define the callable calculator that defines the right-hand-side ODE function
# PlantXCalc = PlantX.calculate

# Model = ODEModelSolver(calculator=PlantXCalc, states_init=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_start=time_axis[0], log_diagnostics=True)

# forcing_inputs = [Climate_solRadswskyb_f,
#                   Climate_solRadswskyd_f,
#                   Climate_airTempCMin_f,
#                   Climate_airTempCMax_f,
#                   Climate_airPressure_f,
#                   Climate_airRH_f,
#                   Climate_airCO2_f,
#                   Climate_airO2_f,
#                   Climate_airU_f,
#                   Climate_soilTheta_z_f,
#                   Climate_doy_f,
#                   Climate_year_f]

# reset_days = reset_days_tax

# zero_crossing_indices = [4,5,6]

# # %% [markdown]
# # ### Step 5. Re-Initialise and Run the Model Using the Sampled Input Sets
# #
# # Once you have the samples, you can evaluate your model at each of the sampled input points. This typically involves running a loop where each set of sampled inputs is passed through the model, and the output (or outputs) is recorded.

# # %%
# # iparamsets_per_run = [list(range(i, i+n_processes)) for i in range(0, len(param_values), n_processes)]
# # iparamsets_per_run = [param_values[i * n_processes : (i + 1) * n_processes] for i in range(len(param_values) // n_processes)]
# iparamsets_per_run = [list(range(len(param_values)))[i * n_processes : (i + 1) * n_processes] for i in range(len(param_values) // n_processes + 1)]
# write_to_nc = True
# # # Create input_data for model run
# input_data = [ODEModelSolver, time_axis, forcing_inputs, reset_days, zero_crossing_indices]

# # # Create output array for target variables
# Mpx_column_headers = "nparamset,W_P_peakW,W_L_peakW,W_R_peakW,W_S_peakW,W_S_spike0,W_S_anth0,GPP_int_seas,NPP_int_seas,Rml_int_seas,Rmr_int_seas,Rg_int_seas,trflux_int_seas,FCstem2grain_int_seas,NPP2grain_int_seas,E_int_seas,LAI_peakW,W_spike_anth1,GY_mature,Sdpot_mature,GN_mature"   # to save as header in csv file. N.B. must match the custom output in the update_and_run_model function
# Mpx = []
# n_runs = len(iparamsets_per_run)
# makedirs(filepath_write, exist_ok=True)

# def evaluate_paramset(iparamset: int)->np.ndarray:
#     nparamset = iparamset + 1
#     paramset = param_values[iparamset]
#     model_output = fastsa.update_and_run_model(
#         paramset,
#         PlantX,
#         [ODEModelSolver, time_axis.copy(), forcing_inputs, reset_days, zero_crossing_indices],
#         parameters_df,
#         problem)
#     Mpxi, diagnostics = model_output[0], model_output[1]
#     if write_to_nc:
#         # Write diagnostics to file
#         nsigfigures = len(str(np.shape(param_values)[0]))   # number of significant figures of parameter sets to insert zero-padded nparamset into filename 
#         # File name for writing outputs to file
#         filename_write = f"FAST_results_{xsite}_paramset{nparamset:0{nsigfigures}}.nc"
#         paramset = param_values[iparamset]
#         print(filepath_write)
#         fastsa.write_diagnostics_to_nc(PlantX, diagnostics, filepath_write, filename_write, time_axis, time_nday_f, time_year_f, time_doy_f, problem, paramset)
#     return np.insert(Mpxi, 0, nparamset)
# # %%
# print(n_runs)
# iparamsets_per_run = list(reversed(iparamsets_per_run))
# print(iparamsets_per_run[0])
# start = time()
# for n_run in tqdm(range(n_runs), desc="Processing"):
#     with Pool(processes=n_processes) as pool:
#         results = pool.map(evaluate_paramset, iparamsets_per_run[n_run])
#     Mpx += results
# # %%
# end = time()

# np.save(path_mpx, np.array(Mpx))
# print(path_mpx)
# # # %%
# # dir_time_log = ''.join([filepath_write, 'time_log'])