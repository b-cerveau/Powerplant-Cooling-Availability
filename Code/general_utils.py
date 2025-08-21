import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from netCDF4 import Dataset

import pandas as pd
import datetime


################################################
# CSV utils for plant data

def filterPlantList(inputpath, outputpath):
    '''
    Filters data CSVs from the JRC plant database to include only currently commissioned plants
    with freshwater cooling, excluding hydropower plants. Adds a 'region' column set to 'Germany'.
    Saves the resulting cleaned CSV to `outputpath`.
    '''

    df = pd.read_csv(inputpath, sep=',') 

    # Filter: freshwater, commissioned, non-hydro, in Germany
    filtered_df = df[
        (df['water_type'] == 'Freshwater') &
        (df['status_g'] == 'COMMISSIONED') &
        ~df['type_g'].isin([
            'Hydro Pumped Storage',
            'Hydro Run-of-river and poundage',
            'Hydro Water Reservoir'
        ]) &
        (df['country'] == 'Germany')
    ].copy()

    # Add the 'region' column
    filtered_df.loc[:, 'region'] = 'Germany'

    # Write output
    filtered_df.to_csv(outputpath, index=False)

def appendEfficiencyData(plantpath, efficiencypath, outputpath):
    '''
    For a CSV file of powerplants (path `plantpath`) containing database key attributes `eic_p` and `eic_g`, 
    completes it with efficiency information by looking up a plant efficiency database (csv under `efficiencypath`), 
    and saves the result to `outputpath`
    '''
    plant_df = pd.read_csv(plantpath, sep = ';')
    efficiency_df = pd.read_csv(efficiencypath, sep= ',')

    # Merge the two dataframes on eic_p and eic_g
    merged_df = pd.merge(
        plant_df,
        efficiency_df[['eic_p', 'eic_g', 'eff', 'best_source']],
        on=['eic_p', 'eic_g'],
        how='left'  # Use 'left' join to keep all rows from the plant list
    )

    merged_df.to_csv(outputpath, index=False)

    print(f'Merge complete. Output saved under {outputpath}')

def getListofCoordinates(files, outputpath = 'unique_coordinates.csv'):
    ''' Used as a query for meteorological services.'''

    # Read all CSVs, keeping only lat/lon
    dfs = [
        pd.read_csv(f)[lambda df: df['country'].str.lower().isin(['germany','de'])][['lat', 'lon']]
        for f in files
    ]
    all_coords = pd.concat(dfs, ignore_index=True)
    unique_coords = all_coords.drop_duplicates().reset_index(drop=True)

    # Add an index/key column
    unique_coords.insert(0, "id", range(1, len(unique_coords) + 1))
    unique_coords.to_csv(outputpath, index=False)


################################################
# NetCDF utils

def previewDataset(ncDataset):
    """
    Shows various relevant information about a netCDF file (`ncDataset` is an instance of the NetCDF4-Python `Dataset` class).
    """
    print(ncDataset)
    
    # List global attributes
    print("\nGlobal Attributes:")
    print(ncDataset.ncattrs())

    # List dimensions (e.g., time, latitude, longitude)
    print("\nDimensions:")
    for dim in ncDataset.dimensions.values():
        print(dim)

    # List variables
    print("\nVariables:")
    for var_name, var in ncDataset.variables.items():
       print(f"{var_name}: {var}")
    print('Done')

def showGriddedDischarge(ncDataset, disVar, lonVar, latVar, timestamp, savename=None):
    """
    Given a gridded river discharge dataset (netCDF variable with format time,lat,lon) at a specific time, returns the discharge map in logscale.

    `dischargeVar`, `lonVar`, `latVar` = discharge/longitude/latitude NetCDF variable names (`str`)
    `timestamp` = time index at which discharge should be plotted
    `savename` = name under which to save file if wanted. No save by default.
    """

    lon = ncDataset.variables[lonVar][:]
    lat = ncDataset.variables[latVar][:]
    dis = ncDataset.variables[disVar][:]  

    # Prepare discharge data
    dis_plot = dis[timestamp, :, :]
    dis_plot[dis_plot <= 0] = 1e-3  # to avoid log(0)

    vmin = 1e-3
    vmax = np.nanmax(dis_plot)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)

    # Meshgrid for plotting
    lons, lats = np.meshgrid(lon, lat)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8),
                       subplot_kw={'projection': ccrs.PlateCarree()})

    # Set extent [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent([-13, 42, 33, 72], crs=ccrs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='linen')
    ax.add_feature(cfeature.OCEAN, facecolor='#CCFFFF')

    # Gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Plot with log scale
    pcm = ax.pcolormesh(lons, lats, dis_plot,
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    cmap='viridis', shading='auto',
                    transform=ccrs.PlateCarree())

    # Colorbar with log ticks
    cb = plt.colorbar(pcm, orientation='horizontal', pad=0.05, shrink=0.8)
    log_ticks = np.logspace(np.floor(np.log10(vmin)),
                        np.ceil(np.log10(vmax)),
                        int(np.ceil(np.log10(vmax)) - np.floor(np.log10(vmin))) + 1)
    cb.set_ticks([t for t in log_ticks if t <= vmax])
    cb.ax.xaxis.set_major_formatter(LogFormatter())
    cb.set_label('Discharge (m³/s)')

    # Title
    plt.title('Discharge')

    plt.tight_layout()
    plt.show()

    if savename:
        plt.savefig(savename)
    
    print('Done')

def check_Value_in_NetCDF_file(path):
    '''
    Used to look up specific values in a NetCDF file, to manually check for missing data or inconsistencies

    `path` : path to netCDF file

    Here the tool is written to lookup a specific streamflow value from a file
    '''
    ds = Dataset(path)

    # Extract relevant arrays
    object_ids = ds.variables['OBJECTID_1'][:]
    julian_days = ds.variables['Julian_day'][:]
    Tw = ds.variables['Tw']

    # Find index of OBJECTID = 49720
    obj_index = np.where(object_ids == 49720)[0]
    if len(obj_index) == 0:
        raise ValueError("OBJECTID 49720 not found in file")
    obj_index = obj_index[0]

    # Convert 2027-11-06 to Julian day (starting from Jan 1)
    target_date = datetime.date(2027, 11, 6)
    start_date = datetime.date(2027, 1, 1)
    julian_index = (target_date - start_date).days  # zero-based index

    if julian_index >= len(julian_days):
        raise ValueError("Julian index exceeds bounds of file")

    # Get temperature value
    tw_value = Tw[obj_index, julian_index]
    print(f"Water temperature on 2027-11-06 at reach 49720: {tw_value} °C")

    raw_value = Tw[obj_index, julian_index].data  
    print(f"Raw value at reach 49720 on 2027-11-06: {raw_value}")
    print(f"Raw bytes: {Tw[obj_index, julian_index].tobytes()}")

def plot_reach_timeseries(csv_file, reach_id, start_date=None, end_date=None, variable_label=None):
    """
    Plots the timeseries of a specific reach from a hydro CSV file matching the Loire nomenclature.

    Parameters:

    - `csv_file` (`str`): Path to the hydro CSV file (e.g., "Tw_CNRM-CM5-LR_ALADIN63_rcp26.csv").
    - `reach_id` (`int`): The OBJECTID of the reach to plot.
    - `start_date` (`str` or `datetime`): Optional. Start date for the time window (e.g., '2040-01-01').
    - `end_date` (`str` or `datetime`): Optional. End date for the time window (e.g., '2045-12-31').
    - `variable_label` (`str`): Optional. Label for the y-axis. Defaults to "Tw (°C)" or "Q (m³/s)" based on filename.
    """

    df = pd.read_csv(csv_file, parse_dates=['Gregorian_day'])
    reach_col = f"reach_{reach_id}"
    
    if reach_col not in df.columns:
        raise ValueError(f"Reach {reach_id} not found in the data.")

    # Filter date range
    if start_date:
        df = df[df['Gregorian_day'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Gregorian_day'] <= pd.to_datetime(end_date)]

    # Infer variable name from filename if no label provided
    if not variable_label:
        if 'Tw' in csv_file:
            variable_label = 'Water Temperature (°C)'
        elif 'Q' in csv_file:
            variable_label = 'Streamflow (m³/s)'
        else:
            variable_label = 'Value'

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(df['Gregorian_day'], df[reach_col], label=f"Reach {reach_id}", color='tab:blue')
    plt.xlabel('Date')
    plt.ylabel(variable_label)
    plt.title(f'{variable_label} for Reach {reach_id}')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

