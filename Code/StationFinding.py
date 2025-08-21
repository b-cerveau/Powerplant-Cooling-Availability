import numpy as np
import pandas as pd

################################################
LOIRE_ATM_STATIONS = 'Inputs/Loire/DRIAS_processed/SAFRAN_metadata.csv'
GERMANY_Q_STATIONS = 'Inputs/Germany/Q_processed/station_metadata.csv'
GERMANY_TW_STATIONS = 'Inputs/Germany/Tw_processed/metadata.csv'

################################################

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    Input coordinates are in decimal degrees.
    Output distance is in kilometers.
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371  # Radius of Earth in kilometers
    return c * r

def findClosestStation(point_lon, point_lat, region : str):
    ''' 
    For a given input point (lon, lat) in decimal degrees, finds its closest neighbor in the list of points of a region-dependent `metadata.csv` file (paths indicated in file header).
    Used to match plant coordinates with their closest point in the atmospheric simulation grid.
    '''
    if region == 'Loire':
        metadata_csv = LOIRE_ATM_STATIONS
        df = pd.read_csv(metadata_csv, dtype=str)
        distances = haversine(point_lon, point_lat, df['lon'].astype(float).values, df['lat'].astype(float).values)
        closest_index = distances.argmin()
        return str(df.iloc[closest_index]['point_id'])
    else:
        raise ValueError(f'Unsupported region for closest point matching : {region}')

def findClosestUpstreamStation(river_km: float, river : str, variable : str, plant_name = 'Unknown') -> str:
    """
    Find the closest upstream station on a river given a river_km location.
    
    Args:
        river_km (float): The river kilometer to compare to.
        variable (str) : 'Q' or 'Tw'
        plant_name (str) :  for easier error logging
    
    Returns:
        str: The name of the closest upstream station (if found), or None.
    """
    match river : 
        case 'Rhine'| 'Elbe':
            match variable:
                case 'Q':
                    metadata_path = GERMANY_Q_STATIONS
                case 'Tw':
                    metadata_path = GERMANY_TW_STATIONS
                case _ :
                    raise ValueError(f"Wrong 'variable' argument given for Rhine station matching : {variable}. Expected 'Q' or 'Tw'.")

            # Load metadata
            df = pd.read_csv(metadata_path)

            # Ensure river_km is numeric and drop missing
            match variable :                    
                case 'Q':
                    df = df[df['River'] == 'Rhein'].copy() if river =='Rhine' else df[df['River']=='Elbe'].copy()
                case 'Tw':
                    df = df[df['River'] == 'Rhine'].copy() if river =='Rhine' else df[df['River']=='Elbe'].copy()      
                    # maybe fix Rhine naming mismatch sometime..
            
            df = df[pd.to_numeric(df['river_km'], errors='coerce').notnull()]
    
            df['river_km'] = df['river_km'].astype(float)
            upstream = df[df['river_km'] < river_km]
            
            if not upstream.empty:
                closest_upstream = upstream.loc[upstream['river_km'].idxmax()]
                print(f'Plant {plant_name} at Rhine km {river_km}, matched with closest upstream {variable} station {closest_upstream['Station']} at km {closest_upstream['river_km']}')
                return closest_upstream['Station']

            # Fallback to closest downstream station if no matching upstream found
            downstream = df[df['river_km'] >= river_km]
            if not downstream.empty:
                closest_downstream = downstream.loc[downstream['river_km'].idxmin()]
                print(f"Plant {plant_name} : No upstream {variable} station found for river km = {river_km}. "
                    f"Matched to closest downstream station '{closest_downstream['Station']}' at km {closest_downstream['river_km']}.")
                return closest_downstream['Station']

            # No stations at all
            print(f"No matching stations found for river km = {river_km}.")
            return None
        
        case _ :
            raise ValueError(f'Unsupported river name input to hydrological station matching : {river}')

if __name__ == '__main__':
    stat = findClosestUpstreamStation(157, 'Rhine', 'Tw', plant_name='Test')
    print(f'Station found: {stat}')