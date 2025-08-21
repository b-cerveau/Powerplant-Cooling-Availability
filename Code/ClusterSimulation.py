import numpy as np
import copy
import os
import pandas as pd

from sklearn.cluster import DBSCAN
from tqdm import tqdm
from collections import defaultdict

from Code.PowerPlantSimulation import PowerPlant

class PlantCluster:
    """
    Represents a cluster of geographically close power plants. Handles:
    
    - Lumping of identical plants
    - Dispatch order logic based on water and cooling preferences
    - Per-timestep simulation
    - Redistribution of output to original plants

    Supports repeated simulation steps with variable environmental conditions.
    """

    WATER_TYPE_PREFERENCE_ORDER = {
        'Seawater': 0,
        'Freshwater': 1
    }

    COOLING_TYPE_PREFERENCE_ORDER = {
        'Air Cooling': 0,
        'Cooling Tower': 1,
        'Once-Through': 2
    }

    ## Initialisation
    @staticmethod
    def _lumpIdenticalPlants(plant_list):
        """
        Groups compatible PowerPlant instances into lumped plants. 
        Compatibility is determined by PowerPlant.isCompatible() : see caveats for custom environmental regulations.

        Args:
            plant_list (list): List of PowerPlant instances.

        Returns:
            tuple:
                lumped_plants (list): List of new, lumped PowerPlant instances.
                lumped_mapping (dict): {id(lumped_plant): [original_plant1, ...]}
        """
        lumped_plants = []
        lumped_mapping = {}
        assigned = [False] * len(plant_list)

        for i, plant in enumerate(plant_list):
            if assigned[i]:
                continue

            compatible_group = [plant]
            assigned[i] = True

            for j in range(i + 1, len(plant_list)):
                if not assigned[j] and PowerPlant.isCompatible(plant, plant_list[j]):
                    compatible_group.append(plant_list[j])
                    assigned[j] = True

            # Combine group
            if len(compatible_group) == 1:
                lumped_plant = compatible_group[0]
            else:
                ref = compatible_group[0]
                lumped_plant = copy.copy(ref)
                lumped_plant.name = '+'.join(p.name for p in compatible_group)
                lumped_plant.capacity_net = sum(p.capacity_net for p in compatible_group)
                lumped_plant.capacity_gross = sum(p.capacity_gross for p in compatible_group)
                lumped_plant.P_min = sum(p.P_min for p in compatible_group)

            lumped_plants.append(lumped_plant)
            lumped_mapping[id(lumped_plant)] = compatible_group

        return lumped_plants, lumped_mapping

    @staticmethod
    def _prepareCluster(plants):
        """
        Prepares a cluster by:
        - Lumping identical plants
        - Sorting by dispatch order : first by water type, then by cooling type, then by efficiency

        Args:
            plants (list): Original list of PowerPlant instances.

        Returns:
            tuple:
                lumped_sorted (list): Lumped plants in dispatch order.
                lumped_mapping (dict): {id(lumped_plant): [original_plant1, ...]}
        """
        lumped_plants, lumped_mapping = PlantCluster._lumpIdenticalPlants(plants)

        lumped_sorted = sorted(
            lumped_plants,
            key=lambda p: (
                PlantCluster.WATER_TYPE_PREFERENCE_ORDER.get(p.water_type, 99),
                PlantCluster.COOLING_TYPE_PREFERENCE_ORDER.get(p.cooling_type, 99),
                -p.efficiency_net
            )
        )

        return lumped_sorted, lumped_mapping

    def __init__(self, plants):
        """
        Initializes a PlantCluster instance from a list of PowerPlant objects.

        Args:
            plants (list): List of original PowerPlant instances in the cluster.
        """
        self.original_plants = plants
        self.lumped_sorted, self.lumped_mapping = PlantCluster._prepareCluster(plants)

    def __repr__(self):
        """
        Returns a readable string representation of the cluster and its structure.

        Example:
            Cluster@0x1234:
              Lumped plant #1 NAME (500.0 MW):
                → Subplant A (250.0 MW)
                → Subplant B (250.0 MW)
        """
        lines = [f"Cluster@{hex(id(self))}:"]
        for i, lumped in enumerate(self.lumped_sorted, 1):
            subplants = self.lumped_mapping[id(lumped)]
            lines.append(f"  Lumped plant #{i} {lumped.name} ({lumped.capacity_net:.1f} MW):")
            for p in subplants:
                lines.append(f"    → {p.name} ({p.capacity_net:.1f} MW)")
        return "\n".join(lines)

    @staticmethod
    def clustersFromList(plants, radius_km=10):
        """
        Static constructor that groups plants into clusters based on geographic proximity.

        Args:
            plants (list): List of PowerPlant instances.
            radius_km (float): Clustering radius in kilometers.

        Returns:
            list: List of PlantCluster instances.
        """
        coords = np.array([[p.lat, p.lon] for p in plants])
        coords_rad = np.radians(coords)
        eps_rad = radius_km / 6371.0
        db = DBSCAN(eps=eps_rad, min_samples=1, metric='haversine')
        labels = db.fit_predict(coords_rad)

        clusters = {}
        for label, plant in zip(labels, plants):
            clusters.setdefault(label, []).append(plant)

        return [PlantCluster(cluster_plants) for cluster_plants in clusters.values()]

    ## Simulation logic

    # Helper functions 
    def simulateStep(self, Q : float = None, Tw : float = None, Tair_dict: float = None, RH_dict : float = None, day = None,
                      dynamicTower : bool = True):
        """
        Simulates one timestep for the cluster with given environmental inputs.
        Downstream impact of one plant to the other is taken into account.
        For lumped plants, weighted average of subplants atmospheric conditions (Tair and RH) is used.

        Args:
            Q (float): Current upstream flow rate (m3/s).
            Tw (float): Current upstream water temperature (°C).
            Tair_dict (dict): {plant_name: air temperature (°C)}.
            RH_dict (dict): {plant_name: relative humidity (%)}.

        Returns:
            tuple:
                cluster_output (dict): {id(lumped_plant): net_output (MW)}.
                Q (float): Updated downstream flow (m3/s).
                Tw (float): Updated downstream temperature (°C).
        """
        cluster_output = {}

        for plant in self.lumped_sorted:
            group = self.lumped_mapping[id(plant)]
            total_capacity = sum(p.capacity_net for p in group)
            tair = sum(p.capacity_net * Tair_dict[p.name] for p in group) / total_capacity
            rh = sum(p.capacity_net * RH_dict[p.name] for p in group) / total_capacity

            net_out, Q, Tw = plant.powerAvailability(Q=Q, Tw=Tw, Tair=tair, RH=rh, day = day,
                                                     enforce_Pmin= bool(len(self.lumped_sorted)-1), dynamicTower = dynamicTower)
            cluster_output[id(plant)] = net_out

        return cluster_output, Q, Tw

    def redistributeOutput(self, cluster_output):
        """
        Redistributes output from lumped plants back to individual plants
        based on their relative capacities.

        Args:
            cluster_output (dict): {id(lumped_plant): total_output}

        Returns:
            dict: {original_plant_name: redistributed_output}
        """
        detailed_output = {}

        for lumped_id, total_output in cluster_output.items():
            plants = self.lumped_mapping[lumped_id]
            total_capacity = sum(p.capacity_net for p in plants)

            for plant in plants:
                share = plant.capacity_net / total_capacity if total_capacity > 0 else 0
                detailed_output[plant.name] = total_output * share

        return detailed_output

    @staticmethod
    def _get_reference_plant(plants):
        '''Determines plant to use as hydrology reference when simulating a cluster.
        For Loire: plants make up independent clusters. For the others, gets most upstream plant using rhein_km'''
        sample = plants[0]
        match sample.region:
            case'Loire':
                return sample
            case 'Rhine','Elbe':
                return min(plants, key=lambda p: getattr(p, 'river_km', float('inf')))
            case _ :
                raise ValueError(f"Region {sample.region} not supported for clustering simulation selection.")

    # Full simulator
    def simulateClusterProduction(cluster, configdir: str, scenario: str, model: str,
                              show_progress: bool = True, dynamicTower: bool = True, inputpath : str = None):
        """
        Simulates a PlantCluster over time and writes back updated CSVs with 'Power' column
        for each subplant using their own timeseries files and shared hydrological data.
        Expects timeseries under configdir/region/scenario/model/Timeseries/plantname_scenario_model_timeseries.csv
        Can be overriden by passing a dedicated inputpath.

        Args:
            cluster (PlantCluster): the plant cluster to simulate
            configdir (str): directory where timeseries are stored
            scenario (str): climate scenario
            model (str): GCM-RCM identifier
            show_progress (bool): if True, show a progress bar
            dynamicTower (bool): passed to plant.powerAvailability
            inputpath (str): path to timeseries directory, to override standard path finding
        """

        # All original subplants in the cluster except air-cooled ones, which don't have hydro data
        all_plants = [p for group in cluster.lumped_mapping.values() for p in group]
        water_plants = [p for p in all_plants if p.cooling_type!= 'Air cooling']
        region = all_plants[0].region

        # Determine most upstream water-dependent plant, if there are water dependent plants at all
        if len(water_plants)==0:    
            noHydro = True
            reference_plant = all_plants[0]
        else :
            noHydro = False
            reference_plant = PlantCluster._get_reference_plant(water_plants)
        
        print(f'Starting to simulate production under scenario {scenario}, model {model}, for cluster : {cluster}')
        # Load all timeseries 
        timeseries_dir = (inputpath if inputpath
                          else os.path.join(configdir,region, scenario, model, 'Timeseries'))
        
        plant_data = {}
        shared_file = f"{reference_plant.name}_{scenario}_{model}_timeseries.csv"
        shared_df = pd.read_csv(os.path.join(timeseries_dir, shared_file))
        try :
            if not noHydro:
                Q_series = shared_df["Q"]
                Tw_series = shared_df["Tw"]
            day_series = shared_df["Gregorian_day"]
            n_rows = len(shared_df)
    
        except KeyError as e:
                print(f"KeyError encountered: {e} in region {region}, scenario {scenario}, model {model}, when simulating cluster {cluster}.")
                print(f"All plants : {all_plants}")
                print(f'Water dependent plants : {water_plants}')
                print(f'Water dependent plants present : {not noHydro}')
                print(f'Reference plant : {reference_plant}')
                raise
        
        except Exception as e:
            print(f"Unexpected error: {e} in region {region}, scenario {scenario}, model {model}, when simulating cluster {cluster}. ")
            raise

        for plant in all_plants:
            file = f"{plant.name}_{scenario}_{model}_timeseries.csv"
            df = pd.read_csv(os.path.join(timeseries_dir, file))
            plant_data[plant.name] = df

        # Run simulation timestep by timestep
        
        if show_progress:
            iterator = tqdm(range(n_rows), desc=f"Simulating {scenario} / {model}")
        else:
            iterator = range(n_rows)

        # Store outputs per plant name
        output_by_plant = defaultdict(list)

        for t in iterator:
            # Build input dictionaries
            Tair_dict = {}
            RH_dict = {}
            for plant in all_plants:
                df = plant_data[plant.name]
                if "Tair" in df.columns:
                    Tair_dict[plant.name] = df.loc[t, "Tair"]
                if "RH" in df.columns:
                    RH_dict[plant.name] = df.loc[t, "RH"]

            # Run cluster simulation for timestep t
            try:
                cluster_output, Q_next, Tw_next = cluster.simulateStep(
                    Q=Q_series[t] if not noHydro else None,
                    Tw=Tw_series[t] if not noHydro else None,
                    Tair_dict=Tair_dict,
                    RH_dict=RH_dict,
                    day = day_series[t],
                    dynamicTower = dynamicTower
                )
            except KeyError as e:
                print(f"KeyError encountered: {e} at timestep {t} in region {region}, scenario {scenario}, model {model}, when simulating cluster {cluster}. \n noHydro {noHydro}")
                raise
            except Exception as e:
                print(f"Unexpected error: {e} at timestep {t} in region {region}, scenario {scenario}, model {model}, when simulating cluster {cluster}. \n noHydro {noHydro}")
                raise

            # Redistribute to subplants
            redistributed = cluster.redistributeOutput(cluster_output)
            for plant_name, power in redistributed.items():
                output_by_plant[plant_name].append(power)

       
        # Write outputs to CSVs

        for plant in all_plants:
            df = plant_data[plant.name]
            df["Power"] = output_by_plant[plant.name]
            filename = f"{plant.name}_{scenario}_{model}_timeseries.csv"
            df.to_csv(os.path.join(timeseries_dir, filename), index=False)
            print(f"→ Updated power output written for {plant.name}")
    
    @staticmethod
    def clusterAndSimulate(plant_list, configdir : str, scenario : str, model : str,
                            showProgress = True, dynamicTower = True, inputpath = None):
        ''' Builds clusters and simulates output for a group of plants. Arguments match `simulateClusterProduction` above.'''
        clusters = PlantCluster.clustersFromList(plant_list)
        for cluster in clusters:
            cluster.simulateClusterProduction(configdir, scenario, model, 
                                              show_progress=showProgress, dynamicTower=dynamicTower, inputpath=inputpath)

    @staticmethod 
    def clusterAndSimulate_wrapper(args):
        plant_list, configdir, scenario, model, inputpath = args
        PlantCluster.clusterAndSimulate(plant_list, configdir, scenario, model, inputpath=inputpath, showProgress=False)

