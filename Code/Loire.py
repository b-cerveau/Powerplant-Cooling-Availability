import numpy as np
import pandas as pd
import os
import xarray as xr
import re
from pathlib import Path
from datetime import datetime, timedelta, date
from functools import partial

from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from Code.PowerPlantSimulation import PowerPlant, EnvironmentalReg

'''
This file supports the relevant plant initialisation and timeseries preprocessing steps for the Loire power plants in France.

Original hydrological dataset : https://doi.org/10.5194/essd-15-2827-2023

Atmospheric dataset : https://www.drias-climat.fr/commande
'''

## Input filepaths
PLANTS = 'Inputs/Loire/Plants/LoirePlants.csv'
ATM_RAW = 'Inputs/Loire/DRIAS_raw'
ATM_PROCESSED = 'Inputs/Loire/DRIAS_processed'
HYDRO_RAW = 'Inputs/Loire/TNET_raw'
HYDRO_PROCESSED = 'Inputs/Loire/TNET_processed'

## Valid model/scenarios
MODELS = {'CNRM-CM5-LR_ALADIN63', 'IPSL-CM5A-MR_WRF381P', 'HadGEM2_CCLM4-8-17'}
SCENARIOS = {'rcp26', 'rcp45', 'rcp85'}
VALID_COMBINATIONS = {
    (scenario, model)
    for scenario in SCENARIOS
    for model in (['CNRM-CM5-LR_ALADIN63'] if scenario == 'rcp26' else MODELS)
}

# Plant initialisation
def Civaux_environmental_checker(Q=np.inf,Tw=-np.inf, T_outlet=-np.inf, Q_wd =0, Q_evap=0, 
                                 T_regulation = True, Q_regulation = True, **kwargs):
    '''
    Full environmental conditions for the CIVAUX plant: used to build the PowerPlant object later on.

    **Input parameters:**

    - `Q`: upstream river flow (m³/s)  
    - `Tw`: upstream river temperature (°C)  
    - `T_outlet`: river temperature at plant outlet (°C)  
    - `Q_wd`: withdrawal flow rate (m³/s)  
    - `Q_evap`: consumption flow rate  

    All variables default to a neutral value if not indicated.

    **Output:**

    - `True` if the operating conditions are compliant with regulation, `False` otherwise
    '''
    if Q_regulation:
        if Q_wd > 6:
            return False
        if Q_evap > 1.7:
            return False
        if Q-Q_evap < 10 :
            return False
        
    if T_regulation:
        T_downstream = EnvironmentalReg.downstream_temp(Q, Tw, Q_wd, Q_evap, T_outlet)
        DeltaT = T_downstream - Tw

        if Tw < 25:
            if DeltaT > 2 or T_downstream >25:
                return False
        elif DeltaT > 1e-4 :
            return False
    return True

def Cordemais_environmental_checker(Q=np.inf,Tw=-np.inf, T_outlet=-np.inf, Q_wd =0, day = '', 
                                    T_regulation = True, Q_regulation = True, **kwargs):
    
    ''' Source : https://www.loire-atlantique.gouv.fr/contenu/telechargement/51838/336012/file/APC+CORDEMAIS.pdf'''

    if Q_regulation:
        if Q_wd > 46 or Q_wd >Q:
            return False
        
    if T_regulation:
        d = datetime.strptime(day, "%Y-%m-%d").date()
        Jun15 = date(d.year, 6, 15)  
        Oct10  = date(d.year, 10, 10)

        if T_outlet - Tw > 8:
            return False
        if Jun15 <= d <= Oct10:
            if T_outlet > 34:
                return False
        elif T_outlet > 30:
            return False

    return True

def initialiseLoirePlants(T_regulation = True, Q_regulation = True):
    '''
    Initialises all the modelled plants from the Loire river basin, with or without their temperature regulation:

    - Nuclear plants : CHINON, BELLEVILLE, DAMPIERRE, ST_LAURENT, CIVAUX
    - Coal plants : CORDEMAIS
    - Gas plants : BAYET (GA-MORANT-1)

    '''
    custom_regulations = {
        'BELLEVILLE' : EnvironmentalReg.loire_regulation(10.5, 
                                                         T_regulation= T_regulation, Q_regulation = Q_regulation),

        'CHINON' : EnvironmentalReg.loire_regulation(8.6, 
                                                     T_regulation=T_regulation, Q_regulation= Q_regulation),

        'DAMPIERRE' : EnvironmentalReg.loire_regulation(12.3, 
                                                        T_regulation = T_regulation, Q_regulation = Q_regulation),

        'ST_LAURENT' : EnvironmentalReg.loire_regulation(7, 
                                                         T_regulation = T_regulation, Q_regulation = Q_regulation),

        'CIVAUX': EnvironmentalReg(regulation_function=partial(Civaux_environmental_checker,
                                                                T_regulation=T_regulation, Q_regulation=Q_regulation)),

        'CORDEMAIS': EnvironmentalReg(regulation_function=partial(Cordemais_environmental_checker,
                                                                    T_regulation=T_regulation, Q_regulation=Q_regulation),
                                        Q_wd_max= 46 if Q_regulation else None, 
                                        Q_evap_max = np.inf if Q_regulation else None,
                                        Q_downstream_min= 0 if Q_regulation else None,
                                        gamma = 1 if Q_regulation else None)
    }

    plant_list = PowerPlant.loadFromCSV(PLANTS, sep=';')

    for plant in plant_list:
        for key in custom_regulations:
            if plant.name.upper().startswith(key.upper()):
                plant.environmental_reg = custom_regulations[key]
                break

    print('Loire plants initialised.')
    return plant_list
    # return [plant for plant in plant_list if plant.name.startswith('CORDEMAIS')]         # useful for testing purposes
    
# Timeseries preprocessing
def extractLoireHydroTimeseries(variable: str, gcm_rcm: str, scenario: str, target_reaches, rawdir : str, outputdir : str):
    '''
    For a given RCP scenario and climate model combination, extracts the hydrological simulation timeseries for an input list of river reaches.  
    Runs through the raw simulation folders and concatenates along the time dimension to extract coherent timeseries. A reach ID is an integer 
    between 1 and 52278; the geographical mapping can be seen by opening the `.shp` shapefile in the original hydrological download using GIS software like ArcGIS or QGIS.
    
    Missing values are handled by linear interpolation between the two closest defined values.  
    A log text file tracking changes is also output.

    **Input parameters:**

    - `variable`: desired variable, expected to be `'Q'` or `'Tw'`
    - `gcm_rcm`: climate model combination, expected to be a string from  
    `{'CNRM-CM5-LR_ALADIN63', 'IPSL-CM5A-MR_WRF381P', 'HadGEM2_CCLM4-8-17'}`
    - `scenario`: RCP pathway, expected to be a string from  
    `{'rcp26', 'rcp45', 'rcp85'}`
    - `target_reaches`: list of reach IDs to extract timeseries for.  
    
    - `rawdir`: filepath to the directory containing all raw downloaded simulation folders.  
    The simulation folder naming is expected to match the original format:  
    `'Projections_{variable}_{gcm_rcm}_{scenario}_2006_2100'`
    - `outputdir`: filepath to the directory where outputs will be stored

    **Outputs:**

    - A CSV file named ``{variable}_{gcm_rcm}_{scenario}.csv`` in `outputdir`.  
    This CSV contains two time dimension columns: `'Gregorian_day'` and `'Julian_day'` (where `Julian_day` is 1 on 01/01/2006 and increments thereafter).  
    It also contains one column per desired reach named `'reach_{reachID}'` listing the timeseries for that reach.  
    
        - *Example:* The file `Q_CNRM-CM5-LR_ALADIN63_rcp26.csv` contains timeseries for streamflow under RCP2.6 with the GCM-RCM combination `CNRM-CM5-LR_ALADIN63`.  
    If `49720` was in `target_reaches`, the column `'reach_49720'` contains the streamflow values at that reach.

    - A text file named ``{variable}_{gcm_rcm}_{scenario}_interpolation_log.txt`` tracking missing values and their handling.

    **Returns:**

    - `outputpath`: The path to the output CSV file
    '''

    os.makedirs(rawdir, exist_ok=True)
    os.makedirs(outputdir, exist_ok=True)

    # Validation
    
    if gcm_rcm not in MODELS:
        raise ValueError("Invalid GCM-RCM.")
    if scenario not in SCENARIOS:
        raise ValueError("Invalid scenario.")
    if variable not in {'Q', 'Tw'}:
        raise ValueError("Invalid variable. Use 'Q' or 'Tw'.")
    
    target_reaches = {rid for rid in target_reaches if rid >= 0}

    data_dir = os.path.join(rawdir, f'Projections_{variable}_{gcm_rcm}_{scenario}_2006_2100')
    nc_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")])

    reach_series = {rid: [] for rid in target_reaches}
    gregorian_days = []

    base_date = datetime(2006, 1, 1)

    for fpath in nc_files:
        ds = xr.open_dataset(fpath)

        filename = os.path.basename(fpath)
        try:
            start_date_str = filename.split('_')[-2]  # e.g., "20060101"
            start_date = datetime.strptime(start_date_str, "%Y%m%d")
        except (IndexError, ValueError):
            raise ValueError(f"Could not parse date from filename: {filename}")

        julian_vals = ds["Julian_day"].values
        if not (np.all(np.diff(julian_vals) == 1) and julian_vals[0] == 1):
            raise ValueError(f"Unexpected Julian_day values in file: {filename}")

        num_days = ds.sizes["Julian_day"]
        days = [start_date + timedelta(days=i) for i in range(num_days)]
        gregorian_days.extend(days)

        for rid in target_reaches:
            values = ds[variable].sel(OBJECTID_1=rid).values
            reach_series[rid].append(values)

        ds.close()

    final_data = {rid: np.concatenate(chunks) for rid, chunks in reach_series.items()}

    df = pd.DataFrame({
        "Gregorian_day": gregorian_days,
        "Julian_day": [(d - base_date).days + 1 for d in gregorian_days]
    })

    log_lines = []

    for rid in sorted(final_data.keys()):
        values = final_data[rid]
        series = pd.Series(values, index=gregorian_days)
        if series.isna().any():
            interpolated = series.interpolate(method="linear", limit_direction="both")
            nan_indices = series[series.isna()].index

            for day in nan_indices:
                log_lines.append(f"Reach {rid} | Date {day.strftime('%Y-%m-%d')} | Interpolated value: {interpolated[day]:.4f}")
            series = interpolated

        df[f"reach_{rid}"] = series.values

    # Save CSV
    output_path = os.path.join(outputdir, f'{variable}_{gcm_rcm}_{scenario}.csv')
    df.to_csv(output_path, index=False)

    # Save log
    log_path = os.path.join(outputdir, f'{variable}_{gcm_rcm}_{scenario}_interpolation_log.txt')
    with open(log_path, 'w') as f:
        if log_lines:
            f.write("\n".join(log_lines))
        else:
            f.write("No missing values found. No interpolation performed.\n")

    print(f'Hydro extraction for {variable} completed successfully.\nCSV saved at: {output_path}\nLog saved at: {log_path}')
    return output_path

def getLoireHydroPath(gcm_rcm, scenario, plant_list = None, runExtraction=True):
    '''
    To avoid re-extracting the subset of reach timeseries from large NetCDF files.  
    Runs the extraction using `extractLoireHydroTimeseries()` if `runExtraction` is True, and always returns the path where the resulting extraction files are stored.  

    Typical use case: plants have already been simulated before, so hydrological simulation files exist in the same path.

    **Input parameters:**

    - `gcm_rcm` (`str`): GCM RCM combination used, expected to be in  
    `{'CNRM-CM5-LR_ALADIN63', 'IPSL-CM5A-MR_WRF381P', 'HadGEM2_CCLM4-8-17'}`
    - `scenario` (`str`): RCP scenario used, expected to be in  
    `{'rcp26', 'rcp45', 'rcp85'}`
    - `plant_list`: list of instances of the `PowerPlant` class. Plants the hydrology has to be extracted for if `runExtraction` is `True`.
    - `runExtraction` (`bool`): `True` if extraction must be run for the given `plant_list`.

    **Output:**

    - `(Q_output_path, Tw_output_path)`: tuple containing paths to CSV files where the subset of extracted reach timeseries for `Q` and `Tw` are stored.
    '''

    
    if runExtraction :
        if plant_list is None :
            raise ValueError('No plant list given for the extraction of hydro time series')
        
        reaches = {plant.TNET_reach for plant in plant_list if plant.TNET_reach>=0}
        Q_output_path = extractLoireHydroTimeseries('Q', gcm_rcm, scenario, reaches, rawdir=HYDRO_RAW, outputdir=HYDRO_PROCESSED)
        Tw_output_path = extractLoireHydroTimeseries('Tw', gcm_rcm, scenario, reaches, rawdir=HYDRO_RAW, outputdir=HYDRO_PROCESSED)

    else :
        Q_output_path = os.path.join(HYDRO_PROCESSED, f'Q_{gcm_rcm}_{scenario}.csv')
        Tw_output_path = os.path.join(HYDRO_PROCESSED, f'Tw_{gcm_rcm}_{scenario}.csv')

    return Q_output_path,Tw_output_path

def prepareLoireAtmosphereTimeseries():
    '''
    Processes raw DRIAS2020 `.txt` atmospheric simulation files from given SAFRAN gridpoints into cleaned atmospheric timeseries.

    - Converts temperature from Kelvin to Celsius (`Tair`)
    - Converts specific humidity to relative humidity (`RH`) in %
    - Extracts SAFRAN gridpoint metadata: latitude, longitude, Lambert coordinates, altitude
    - Saves gridpoint metadata to `SAFRAN_metadata.csv`
    - Saves cleaned timeseries files using naming conventions consistent with the rest of the code:  
    `output_dir/scenario/model/PointID_scenario_model_atmtimeseries.csv`
    '''

    input_dir = ATM_RAW
    output_dir = ATM_PROCESSED

    model_map = {
        'CNRM-CERFACS-CNRM-CM5_CNRM-ALADIN63': 'CNRM-CM5-LR_ALADIN63',
        'MOHC-HadGEM2-ES_CLMcom-CCLM4-8-17': 'HadGEM2_CCLM4-8-17',
        'IPSL-IPSL-CM5A-MR_IPSL-WRF381P': 'IPSL-CM5A-MR_WRF381P'
    }
    scenario_map = {
        'rcp2': 'rcp26',
        'rcp4': 'rcp45',
        'rcp8': 'rcp85'
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safran_metadata = {}

    for dirname, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.txt'):
                continue

            print(f"Found file: {file}")

            models = {
                'CNRM-CERFACS-CNRM-CM5_CNRM-ALADIN63',
                'MOHC-HadGEM2-ES_CLMcom-CCLM4-8-17',
                'IPSL-IPSL-CM5A-MR_IPSL-WRF381P'
            }
            scenarios = {'rcp2.6', 'rcp4.5', 'rcp8.5'}

            model_pattern = '|'.join(re.escape(m) for m in models)
            scenario_pattern = '|'.join(re.escape(s) for s in scenarios)
            
            pattern = (
                r'^P(?P<point_ID>\d{5})_tasAdjusthussAdjust_France_'
                rf'(?P<model>{model_pattern})_'
                rf'(?P<scenario>{scenario_pattern})_METEO-FRANCE_ADAMONT-France_SAFRAN_day_'
                r'\d{8}-\d{8}\.txt$'
            )
            
            match = re.match(pattern, file)
            if not match:
                raise ValueError(f"Filename does not match expected pattern: {file}")
            
            point_id,model_raw,scenario_full = match.group('point_ID'), match.group('model'), match.group('scenario')

            model_std = model_map.get(model_raw)
            scenario_short = f"{scenario_full.split('.')[0]}"
            scenario_std = scenario_map.get(scenario_short)

            if not model_std or not scenario_std:
                print(f"⚠️ Unknown model or scenario in: {file}")
                continue

            subdir = os.path.join(output_dir, scenario_std, model_std)
            Path(subdir).mkdir(parents=True, exist_ok=True)
            out_file = os.path.join(subdir, f"{point_id}_{scenario_std}_{model_std}_atmtimeseries.csv")
            filepath = os.path.join(dirname, file)

            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            meta = {
                'point_id': point_id,
                'lat': None,
                'lon': None,
                'E': None,
                'N': None,
                'altitude_m': None
            }

            data_start = None
            for i, line in enumerate(lines):
                if 'Latitude' in line:
                    lat_match = re.search(r"Latitude\s+=\s+([0-9.\-]+)", line)
                    if lat_match:
                        meta['lat'] = float(lat_match.group(1))
                elif 'Longitude' in line:
                    lon_match = re.search(r"Longitude\s+=\s+([0-9.\-]+)", line)
                    if lon_match:
                        meta['lon'] = float(lon_match.group(1))
                elif re.search(r"E\s+=\s+\d+", line):
                    e_match = re.search(r"E\s+=\s+(\d+)", line)
                    if e_match:
                        meta['E'] = int(e_match.group(1))
                elif re.search(r"N\s+=\s+\d+", line):
                    n_match = re.search(r"N\s+=\s+(\d+)", line)
                    if n_match:
                        meta['N'] = int(n_match.group(1))
                elif 'altitude' in line:
                    alt_match = re.search(r"altitude\s+=\s+(\d+)", line)
                    if alt_match:
                        meta['altitude_m'] = int(alt_match.group(1))
                elif line.strip().startswith('200'):
                    data_start = i
                    break

            if all(meta[k] is not None for k in ['lat', 'lon', 'E', 'N', 'altitude_m']):
                safran_metadata[point_id] = meta
            else:
                print(f"⚠️ Incomplete metadata for point {point_id} in file: {file}")
                continue

            df = pd.read_csv(filepath, sep=';', skiprows=data_start, names=['date', 'tasAdjust', 'hussAdjust'])
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['Tair'] = (df['tasAdjust'] - 273.15).round(2)

            # Convert specific humidity → relative humidity
            sh = df['hussAdjust'].values * units('kg/kg')
            temp_k = df['tasAdjust'].values * units('K')
            pressure = 1013.25 * units.hPa
            rh = relative_humidity_from_specific_humidity(pressure, temp_k, sh).to('percent').magnitude
            df['RH'] = rh.round(2)

            df_out = df[['date', 'Tair', 'RH']].copy()
            df_out['date'] = df_out['date'].dt.strftime('%Y-%m-%d')
            df_out.to_csv(out_file, index=False)
            print(f"✅ Saved: {out_file}")

    # Write all collected metadata
    metadata_df = pd.DataFrame(list(safran_metadata.values()))
    metadata_path = os.path.join(output_dir, 'SAFRAN_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n📄 Metadata written to: {metadata_path}")

def buildPlantTimeSeries(plant_list, output_dir : str, 
                         scenario : str, gcm_rcm : str,
                         extractHydro : bool =True, extractAtm : bool = False,
                         Q_offset :float = 1, Tw_offset : float = 0, Tair_offset : float = 0, RH_offset : float = 0):
    '''
    Generates plant-specific climatic input timeseries CSV files for a given plant list.  
    Each file contains input hydroclimatic conditions `Q`, `Tw`, `Tair`, `RH` (where applicable) for the plant,  
    and two time dimensions: `Julian_day` and `Gregorian_day`.  

    - If no data is available for a specific plant (e.g., an air-cooled plant with no hydro input),  
    timeseries will still be generated with the available variables.  
    - If data is missing, output is truncated accordingly (e.g., if meteorological data ends on 2100-12-31 but hydro simulation ends on 2100-07-31, output will end on 2100-07-31).  
    - Output files are saved under:  
    `output_dir/{plant.name}_{scenario}_{gcm_rcm}_timeseries.csv`

    **Inputs:**  
    - `plant_list`: list of instances from the `PowerPlant` class, for which timeseries are to be extracted  
    - `Q_filepath`: path to a CSV file containing streamflow projections. Expected to contain columns `'Gregorian_day'`, `'Julian_day'`,  
    and a set of columns labeled `reach_{TNET reach ID}` containing Q projections for each reach  
    - `Tw_filepath`: path to a CSV file containing water temperature projections. Expected to contain columns `'Gregorian_day'`, `'Julian_day'`,  
    and a set of columns labeled `reach_{TNET reach ID}` containing Tw projections for each reach  
    - `atm_dir`: path to a directory containing atmospheric input data (`'Tair'`: air temperature in °C, `'RH'`: relative humidity in %)  
    Directory expected to be organized as output of `prepareLoireAtmosphereTimeseries()`, i.e.  
    `output_dir/scenario/model/PointID_scenario_model_atmtimeseries.csv`  
    - `output_dir`: directory where the timeseries CSV files will be stored  
    - `scenario`: RCP scenario, expected among `{'rcp26', 'rcp45', 'rcp85'}`  
    - `gcm_rcm`: GCM-RCM combination, expected among `{'CNRM-CM5-LR_ALADIN63', 'IPSL-CM5A-MR_WRF381P', 'HadGEM2_CCLM4-8-17'}`
    '''
    Q_filepath, Tw_filepath = getLoireHydroPath(gcm_rcm=gcm_rcm, scenario = scenario, plant_list=plant_list, runExtraction = extractHydro)
    
    if extractAtm:
        prepareLoireAtmosphereTimeseries()

    atm_dir = ATM_PROCESSED

    # Loads Q and Tw CSVs
    Q_df = pd.read_csv(Q_filepath, parse_dates=["Gregorian_day"])
    Tw_df = pd.read_csv(Tw_filepath)

    # Checks that time dimension is consistent
    if not Q_df['Julian_day'].equals(Tw_df['Julian_day']):
        raise ValueError("Julian_day column mismatch between Q and Tw files.")
        
    # Make sure the input and output directories exist
    os.makedirs(atm_dir,exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for plant in plant_list:
        tnet_reach = plant.TNET_reach
        q_column = f'reach_{tnet_reach}'
        tw_column = f'reach_{tnet_reach}'

        SAFRAN_point = plant.SAFRAN_point
        atm_filepath = os.path.join(atm_dir, scenario, gcm_rcm, f"{SAFRAN_point}_{scenario}_{gcm_rcm}_atmtimeseries.csv")

        q_exists = q_column in Q_df.columns
        tw_exists = tw_column in Tw_df.columns
        atm_exists = os.path.isfile(atm_filepath)

        print(f'For plant : {plant.name}, grid point {SAFRAN_point} \n Data found : {atm_exists}')

        # Load atmospheric data if existing
        if atm_exists:
            atm_df = pd.read_csv(atm_filepath, parse_dates=["date"])
            atm_df.rename(columns={'date': 'Gregorian_day'}, inplace=True)
        else:
            atm_df = None

        # Start assembling the timeseries DataFrame
        base_df = Q_df[['Julian_day', 'Gregorian_day']].copy()

        if q_exists:
            base_df['Q'] = Q_df[q_column]
        if tw_exists:
            base_df['Tw'] = Tw_df[tw_column]
        if atm_df is not None:
            base_df = base_df.merge(atm_df, on='Gregorian_day', how='left')

        # Drop rows with any missing values
        base_df.dropna(inplace=True)

        # If hydro data is not present, fallback to atmospheric data's date index
        if not q_exists and not tw_exists and atm_df is None :
                print(f"No valid data found for plant {plant.name}. Skipping.")
                continue  # Skip this plant entirely
        
        # Apply sensitivity offsets if applicable
        if 'Q' in base_df.columns:
            base_df['Q'] *= Q_offset
        if 'Tw' in base_df.columns:
            base_df['Tw'] += Tw_offset
        if 'Tair' in base_df.columns:
            base_df['Tair'] += Tair_offset
        if 'RH' in base_df.columns:
            base_df['RH'] += RH_offset

        # Output file naming and saving
        output_filename = f'{plant.name}_{scenario}_{gcm_rcm}_timeseries.csv'
        output_path = os.path.join(output_dir, output_filename)
        base_df.to_csv(output_path, index=False)
        print(f"Saved input environmental time series for plant {plant.name} to {output_path}")


