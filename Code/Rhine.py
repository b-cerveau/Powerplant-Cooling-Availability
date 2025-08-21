import os
import re
import numpy as np
import pandas as pd

from Code.Rheinkilometer import get_rhine_km_id
from Code.Elbekilometer import get_elbe_km
from Code.PowerPlantSimulation import PowerPlant, EnvironmentalReg

###########################################################################################################

##################### Input filepaths
Q_RAW = 'Inputs/Germany/Q_raw/Discharge_projections'
Q_PROCESSED = 'Inputs/Germany/Q_processed'
TW_RAW = 'Inputs/Germany/Tw_raw'
TW_PROCESSED = 'Inputs/Germany/Tw_processed'

PLANTS = 'Inputs/Germany/Plants/RhinePlants.csv'

##################### Scenario/model/combinations
SCENARIOS = {'rcp85'}
MODELS = {
        'CANESM2_r1_REMO',
        'ECEARTH_r12_CCLM','ECEARTH_r1_RACMO','ECEARTH_r12_RACMO','ECEARTH_r12_REMO','ECEARTH_r12_RCA',
        'HADGEM2_r1_RACMO',	'HADGEM2_r1_REMO', 'HADGEM2_r1_RCA',
        'IPSL_r1_RCA',
        'MIROC5_r1_CCLM', 'MIROC5_r1_REMO', 
        'MPIESM_r1_CCLM', 'MPIESM_r1_REMO', 'MPIESM_r2_REMO','MPIESM_r1_RCA'
    }
VALID_COMBINATIONS = {
    (scenario, model)
    for scenario in SCENARIOS
    for model in MODELS
}

############################################################################################################

########### Hydro data preprocessing

def cleanGermanyQTimeseries(input_root=Q_RAW, output_folder=Q_PROCESSED):
    """
    Clean and split Q (discharge) timeseries from tab-separated scenario files
    into station-specific CSV files named Q_scenario_stationname.csv, with
    canonical scenario and model names, and generate a single metadata CSV.

    - Input format: stationname_scenario.tab
    - Scenarios in filename are RCP26, RCP45, RCP85 → mapped to rcp26, rcp45, rcp85
    - Model columns are renamed using TARGET_MODELS mapping keys
    - Warns if a station is missing one or more scenarios
    """

    # === Local mapping of scenarios ===
    SCENARIO_MAP = {
        'RCP26': 'rcp26',
        'RCP45': 'rcp45',
        'RCP85': 'rcp85'
    }

    # === Local mapping of canonical model names ===
    TARGET_MODELS = {
        'CANESM2_r1_REMO': 'CANESM2_r1_REMO_BC-EXP_LSM-ME',
        'ECEARTH_r12_CCLM': 'ECEARTH_r12_CCLM_BC-EXP_LSM-ME',
        'ECEARTH_r1_RACMO': 'ECEARTH_r1_RACMO_BC-EXP_LSM-ME',
        'ECEARTH_r12_RACMO': 'ECEARTH_r12_RACMO_BC-EXP_LSM-ME',
        'ECEARTH_r12_REMO': 'ECEARTH_r12_REMO_BC-EXP_LSM-ME',
        'ECEARTH_r12_RCA': 'ECEARTH_r12_RCA_BC-EXP_LSM-ME',
        'HADGEM2_r1_RACMO': 'HADGEM2_r1_RACMO_BC-EXP_LSM-ME',    
        'HADGEM2_r1_REMO': 'HADGEM2_r1_REMO_BC-EXP_LSM-ME',
        'HADGEM2_r1_RCA': 'HADGEM2_r1_RCA_BC-EXP_LSM-ME',
        'IPSL_r1_RCA': 'IPSL_r1_RCA_BC-EXP_LSM-ME',
        'MIROC5_r1_CCLM': 'MIROC5_r1_CCLM_BC-EXP_LSM-ME',    
        'MIROC5_r1_REMO': 'MIROC5_r1_REMO_BC-EXP_LSM-ME',    
        'MPIESM_r1_CCLM': 'MPIESM_r1_CCLM_BC-EXP_LSM-ME',    
        'MPIESM_r1_REMO': 'MPIESM_r1_REMO_BC-EXP_LSM-ME',
        'MPIESM_r2_REMO': 'MPIESM_r2_REMO_BC-EXP_LSM-ME',
        'MPIESM_r1_RCA': 'MPIESM_r1_RCA_BC-EXP_LSM-ME'
    }

    os.makedirs(output_folder, exist_ok=True)
    meta_output_file = os.path.join(output_folder, 'station_metadata.csv')

    station_metadata = []
    station_scenarios_found = {}

    print('Starting to read Q files...')

    for fname in os.listdir(input_root):
        if not fname.endswith('.tab'):
            continue

        print(f'Reading file {fname}...')

        match = re.match(r"(.+?)_(RCP26|RCP45|RCP85)\.tab", fname, re.IGNORECASE)
        if not match:
            print(f"⚠️ Skipping file with unexpected name format: {fname}")
            continue

        station_name, raw_scenario = match.groups()
        raw_scenario = raw_scenario.upper()
        scenario = SCENARIO_MAP.get(raw_scenario)
        if not scenario:
            print(f"⚠️ Scenario '{raw_scenario}' not recognized in file {fname}")
            continue

        full_path = os.path.join(input_root, fname)

        # Read header and extract metadata
        with open(full_path, 'r') as f:
            lines = f.readlines()

        header_start = next(i for i, line in enumerate(lines) if line.strip().startswith('/*'))
        header_end = next(i for i, line in enumerate(lines) if line.strip().endswith('*/'))
        header_text = ''.join(lines[header_start:header_end + 1])

        lat = float(re.search(r'Latitude:\s*([0-9.+-]+)', header_text).group(1))
        lon = float(re.search(r'Longitude:\s*([0-9.+-]+)', header_text).group(1))
        river = re.search(r'River:\s*(.*)', header_text).group(1).strip()

        # Locate data start
        data_start = next(i for i, line in enumerate(lines) if line.strip().startswith('Date'))

        # Read data
        df = pd.read_csv(full_path, sep='\t', skiprows=data_start, parse_dates=['Date'])
        df.rename(columns=lambda c: 'Gregorian_day' if c == 'Date' else c.replace(raw_scenario + '_', ''), inplace=True)
        df['Gregorian_day'] = df['Gregorian_day'].dt.strftime('%Y-%m-%d')
        df.sort_values('Gregorian_day', inplace=True)
        
        # Keep only columns matching TARGET_MODELS values and rename them to keys
        model_col_map = {v: k for k, v in TARGET_MODELS.items()}
        cols_to_keep = ['Gregorian_day'] + [col for col in df.columns if col in model_col_map]
        print(f'Columns to keep : {cols_to_keep}')
        df = df[cols_to_keep]
        df.rename(columns=model_col_map, inplace=True)

        # Save file
        output_file = os.path.join(output_folder, f"Q_{scenario}_{station_name}.csv")
        df.to_csv(output_file, index=False)

        # Track scenarios for warnings
        station_scenarios_found.setdefault(station_name, set()).add(scenario)

        # Add metadata
        match river.lower():
            case 'rhein':
                river_km = get_rhine_km_id(lat, lon)
            case 'elbe':
                river_km = get_elbe_km(lat,lon)
            case _:
                river_km = ''
    
        station_metadata.append({
            'Station': station_name,
            'Latitude': lat,
            'Longitude': lon,
            'River': river,
            'river_km': river_km
        })

    # Warn for missing scenarios
    all_scenarios = set(SCENARIO_MAP.values())
    for station, scenarios in station_scenarios_found.items():
        missing = all_scenarios - scenarios
        if missing:
            print(f"⚠️ Station '{station}' is missing scenarios: {', '.join(sorted(missing))}")

    # Write metadata
    pd.DataFrame(station_metadata).drop_duplicates().to_csv(meta_output_file, index=False)
    print(f"✅ Metadata written to {meta_output_file}")

def cleanGermanyTwTimeseries(input_folder =TW_RAW , output_folder = TW_PROCESSED):
    """
    Clean and split WT (water temperature) timeseries into station-specific CSV files
    and create a metadata.csv file. Assumes scenario is RCP85 by default!!
    
    Args:
        input_files (list): List of input CSV file paths (Rhine, Elbe, etc.).
        output_folder (str): Folder where cleaned outputs will be saved.
    """
    # Mapping from file names to canonical names
    model_name_map = {
        'CAN_01_REM_RCP85': 'CANESM2_r1_REMO',
        'ECE_01_RAC_RCP85': 'ECEARTH_r1_RACMO',
        'ECE_12_CLM_RCP85': 'ECEARTH_r12_CCLM',
        'ECE_12_RAC_RCP85': 'ECEARTH_r12_RACMO',
        'ECE_12_RCA_RCP85': 'ECEARTH_r12_RCA',
        'ECE_12_REM_RCP85': 'ECEARTH_r12_REMO',
        'HAD_01_RAC_RCP85': 'HADGEM2_r1_RACMO',
        'HAD_01_RCA_RCP85': 'HADGEM2_r1_RCA',
        'HAD_01_REM_RCP85': 'HADGEM2_r1_REMO',
        'IPS_01_RCA_RCP85': 'IPSL_r1_RCA',
        'MIC_01_CLM_RCP85': 'MIROC5_r1_CCLM',
        'MIR_01_CLM_RCP85': 'MIROC5_r1_CCLM',      # Naming convention mismatch between Elbe and Rhine files : MIC/MIR
        'MIC_01_REM_RCP85': 'MIROC5_r1_REMO',
        'MPI_01_CLM_RCP85': 'MPIESM_r1_CCLM',
        'MPI_01_RCA_RCP85': 'MPIESM_r1_RCA',
        'MPI_01_REM_RCP85': 'MPIESM_r1_REMO',
        'MPI_02_REM_RCP85': 'MPIESM_r2_REMO'
    }

    # Exact location of stations not included in original files so manually filled in/approximated
    station_coordinates = {
        'Bad_Honnef': (50.641004, 7.211950),
        'Bimmen': (51.860959, 6.072031),
        'Dommitzsch': (51.668646, 12.849120),
        'Geesthacht': (53.426207, 10.367862),
        'Iffezheim': (48.834013, 8.111149),
        'Karlsruhe': (49.036142, 8.303357),
        'Koblenz': (50.357215, 7.606871),
        'Magdeburg': (52.13471,11.644323),
        'Mainz': (50.004868,8.275263),
        'Schnackenburg': (53.052233, 11.547663),
        'Worms': (49.631801,8.377438),
    }

    os.makedirs(output_folder, exist_ok=True)
    metadata = []
    all_models_in_files = set()

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Extract header block between /* and */
        header_start = next(i for i, line in enumerate(lines) if line.strip().startswith('/*'))
        header_end = next(i for i, line in enumerate(lines) if line.strip().endswith('*/'))
        header_text = ''.join(lines[header_start:header_end + 1])

        # Extract river name
        river_match = re.search(r'River:\s*(.*)', header_text)
        river_name = river_match.group(1).strip() if river_match else 'Unknown'

        # Load data after header_end
        df = pd.read_csv(file_path, skiprows=header_end + 1)

        # Map model names
        df['scenario'] = df['scenario'].map(model_name_map).fillna(df['scenario'])
        models_in_file = set(df['scenario'].unique())
        all_models_in_files.update(models_in_file)

        # Melt into long format for easier splitting
        df_long = df.melt(id_vars=['scenario', 'date'], var_name='Station', value_name='WT')

        # Split into separate CSVs for each station
        for station_name, group in df_long.groupby('Station'):
            dupes = group.duplicated(subset=['date', 'scenario'], keep=False)
            if dupes.any():
                dup_count = group[dupes].groupby(['date', 'scenario']).size()
                print(f"⚠️ Warning: Station '{station_name}' has duplicate date-scenario pairs:")
                print(dup_count.head(10))
                print(f"Total duplicate rows: {dupes.sum()} — averaging duplicates when pivoting.")

            pivot_df = (
                group
                .pivot_table(
                    index='date',
                    columns='scenario',
                    values='WT',
                    aggfunc='mean'
                )
                .reset_index()
            )
            pivot_df.rename(columns={'date':'Gregorian_day'}, inplace=True)
            
            # Save with fixed naming (only rcp85 scenario)
            pivot_df.to_csv(
                os.path.join(output_folder, f"Tw_rcp85_{station_name}.csv"),
                index=False
            )

            lat, lon = station_coordinates.get(station_name, (None,None))
            
            match river_name.lower():
                case 'rhein':
                    river_km = get_rhine_km_id(lat, lon)
                case 'elbe':
                    river_km = get_elbe_km(lat,lon)
                case _:
                    river_km = None

            metadata.append({
                'Station': station_name,
                'River': river_name,
                'Latitude': lat,
                'Longitude': lon,
                'river_km': river_name
            })

    print(f'Timeseries saved to folder {TW_PROCESSED}')

    # Save metadata
    pd.DataFrame(metadata).drop_duplicates().to_csv(
        os.path.join(output_folder, 'metadata.csv'),
        index=False
    )

    print(f'Metadata saved to : {os.path.join(output_folder, 'metadata.csv')}')
    # Report unmatched models
    extra_in_file = all_models_in_files - MODELS
    missing_in_file = MODELS - all_models_in_files
    
    if extra_in_file:
        print("⚠️ Models found in file but NOT in expected MODELS list:")
        for m in sorted(extra_in_file):
            print("  ", m)
    if missing_in_file:
        print("⚠️ Models expected in MODELS list but NOT found in file:")
        for m in sorted(missing_in_file):
            print("  ", m)# Mapping from file names to canonical names    

########### Plants initialisation
def initialiseRhinePlants():
    return PowerPlant.loadFromCSV(PLANTS, sep=';')

def buildRhineTimeSeries(plant_list, output_dir : str, 
                         scenario : str, model : str,
                         Q_offset :float = 1, Tw_offset : float = 0, Tair_offset : float = 0, RH_offset : float = 0):
    '''
    Generates plant-specific climatic input timeseries CSV files for a given plant list.  
    Each file contains input hydroclimatic conditions `Q`, `Tw`, `Tair`, `RH` (where applicable) for the plant,  
    and a time dimension `Gregorian_day`.  

    - If no data is available for a specific plant (e.g., an air-cooled plant with no hydro input),  
    timeseries will still be generated with the available variables.  
    - If data is missing, output is truncated accordingly (e.g., if meteorological data ends on 2100-12-31 but hydro simulation ends on 2100-07-31, output will end on 2100-07-31).  
    - Output files are saved under:  
    `output_dir/{plant.name}_{scenario}_{gcm_rcm}_timeseries.csv`

    **Inputs:**  
    - `plant_list`: list of instances from the `PowerPlant` class, for which timeseries are to be extracted   
    - `output_dir`: directory where the timeseries CSV files will be stored  
    - `scenario`: RCP scenario
    - `gcm_rcm`: GCM-RCM combination
    - `Q_offset` : multiplicative offset factor for streamflow
    - `Tw_offset`: additive offset for water temperature (°C)
    - `Tair_offset`: additive offset for air temperature (°C)
    - `RH_offset`: additive offset for relative humidity (%)
    '''
    Q_dir = Q_PROCESSED
    Tw_dir = TW_PROCESSED

    for plant in plant_list:
        Q_station = plant.Q_station
        Tw_station = plant.Tw_station

        Q_filepath = os.path.join(Q_dir, f'Q_{scenario}_{Q_station}.csv')
        Tw_filepath = os.path.join(Tw_dir, f'Tw_{scenario}_{Tw_station}.csv')

        Q_df = pd.read_csv(Q_filepath, parse_dates=["Gregorian_day"])
        Tw_df = pd.read_csv(Tw_filepath)

        q_exists = model in Q_df.columns
        tw_exists = model in Tw_df.columns

        # atm_exists = os.path.isfile(atm_filepath)
        atm_exists = False
        # print(f'For plant : {plant.name}, grid point {SAFRAN_point} \n Data found : {atm_exists}')

        # # Load atmospheric data if existing
        # if atm_exists:
        #     atm_df = pd.read_csv(atm_filepath, parse_dates=["date"])
        #     atm_df.rename(columns={'date': 'Gregorian_day'}, inplace=True)
        # else:
        #     atm_df = None
        atm_df = None

        # Start assembling the timeseries DataFrame
        base_df = Q_df[['Gregorian_day']].copy()

        if q_exists:
            base_df['Q'] = Q_df[model]
        if tw_exists:
            base_df['Tw'] = Tw_df[model]
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
        output_filename = f'{plant.name}_{scenario}_{model}_timeseries.csv'
        output_path = os.path.join(output_dir, output_filename)
        base_df.to_csv(output_path, index=False)
        print(f"Saved input environmental time series for plant {plant.name} to {output_path}")

if __name__ == '__main__':
    cleanGermanyQTimeseries()
    cleanGermanyTwTimeseries()

