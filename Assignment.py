#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
import rasterio as rio
import matplotlib.colors
from matplotlib.patches import Patch
import re

# Read climate data
grh = pd.read_csv("data_raw/climate_GRH_data.txt", sep="\s+")
grh.index = pd.to_datetime(grh["Year"].astype(str) + '-' + grh['Month'].astype(str).str.zfill(2), format='%Y-%m')
grh = grh.drop(columns=["Year", "Month"])
grh = grh.loc[(grh.index >= "1980-10-01") & (grh.index <= "2009-10-01")]

# Read raster data
with rio.open('data_raw/dem_fiescher_1980.asc') as src1:
    dem_1980 = src1.read(1)

with rio.open('data_raw/dem_fiescher_2009.asc') as src2:
    dem_2009 = src2.read(1)

with rio.open('data_raw/glacier_mask_fiescher_1980.asc') as src3:
    mask_1980 = src3.read(1)

with rio.open('data_raw/glacier_mask_fiescher_2009.asc') as src4:
    mask_2009 = src4.read(1)

# Parameters
elevation_grh = 1980  # m
dt_dz = -6.5 / 1000  # °C/m
rain_snow_thresh = 2  # °C
melt_thresh = 0  # °C
days_in_month = 30.42  # days
total_mass_balance = glacier_mass_loss / average_cells_number / cell_surface / rho_water * 1000  # mm w.e.

# Compute DEM with temperature difference between glacier and station
# (assignment says that the glacier should be described with dem_2009 for this)
dem_elev_diff = dem_2009 - elevation_grh
dem_temp_diff = dem_elev_diff * dt_dz  # °C

empty_df = []

for datetime in grh.index:
    # print("Processing", datetime)
    # Creating temperature DEMs for land and glacier
    temp_grh = float(grh["Temperature"].loc[grh.index == datetime])
    prec_grh = float(grh["Precipitation"].loc[grh.index == datetime])

    dem_temp = (dem_temp_diff + temp_grh) * mask_2009
    dem_snow = (np.where(dem_temp <= rain_snow_thresh, 1, 0)) * mask_2009
    dem_accumulation = prec_grh * dem_snow
    dem_melt = (np.where(dem_temp > melt_thresh, 1, 0)) * mask_2009

    empty_df.append([datetime, dem_temp, dem_snow, dem_accumulation, dem_melt])

df = pd.DataFrame(empty_df)
df.columns = ["datetime", "dem_temperature", "dem_snow", "dem_accumulation", "dem_melt"]
df = df.set_index("datetime")

fig, axs = plt.subplots(nrows=2, ncols=4, sharey=True, sharex=True, figsize=(12, 6))
axs[0, 0].imshow(df["dem_accumulation"].loc[df.index == "1990-02-01"][0], cmap="Blues")
axs[0, 0].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[0, 1].imshow(df["dem_accumulation"].loc[df.index == "1990-06-01"][0], cmap="Blues")
axs[0, 1].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[0, 2].imshow(df["dem_accumulation"].loc[df.index == "1990-07-01"][0], cmap="Blues")
axs[0, 2].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[0, 3].imshow(df["dem_accumulation"].loc[df.index == "1990-08-01"][0], cmap="Blues")
axs[0, 3].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[1, 0].imshow(df["dem_melt"].loc[df.index == "1990-02-01"][0], cmap="Reds")
axs[1, 0].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[1, 1].imshow(df["dem_melt"].loc[df.index == "1990-06-01"][0], cmap="Reds")
axs[1, 1].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[1, 2].imshow(df["dem_melt"].loc[df.index == "1990-07-01"][0], cmap="Reds")
axs[1, 2].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[1, 3].imshow(df["dem_melt"].loc[df.index == "1990-08-01"][0], cmap="Reds")
axs[1, 3].contour(mask_2009, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[0, 0].set_title("Accumulation \n 1990-02")
axs[0, 1].set_title("Accumulation \n 1990-06")
axs[0, 2].set_title("Accumulation \n 1990-07")
axs[0, 3].set_title("Accumulation \n 1990-08")
axs[1, 0].set_title("Melt \n 1990-02")
axs[1, 1].set_title("Melt \n 1990-06")
axs[1, 2].set_title("Melt \n 1990-07")
axs[1, 3].set_title("Melt \n 1990-08")
axs[0, 0].set_ylabel("# Cells")
axs[1, 0].set_ylabel("# Cells")
axs[1, 0].set_xlabel("# Cells")
axs[1, 1].set_xlabel("# Cells")
axs[1, 2].set_xlabel("# Cells")
axs[1, 3].set_xlabel("# Cells")
plt.tight_layout()
plt.savefig("plots/ddf_modelling_part_3.png", dpi=300)
plt.show()

df["perc_melting_cells"] = 0

for date in df.index:
    df["perc_melting_cells"].loc[date] = df["dem_melt"].loc[date].sum()

df["temperature_grh"] = grh["Temperature"]
df["degrees_per_day"] = df["temperature_grh"] * df["perc_melting_cells"] * days_in_month

ddf = ((df["dem_accumulation"].sum().sum() - total_mass_balance) / (df["degrees_per_day"].sum()))

print("-" * 10, "RESULTS", "-" * 10)
print("DDF:", round(ddf, 3), "[mm °C-1 d-1]")
print("-" * 30)