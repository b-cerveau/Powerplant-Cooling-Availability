import numpy as np
import pandas as pd
import inspect

from Code.constants import rho_w,L_vap,c_p
from Code.StationFinding import findClosestStation, findClosestUpstreamStation
from Code.Rheinkilometer import get_rhine_km_id
from Code.Elbekilometer import get_elbe_km
from Code.CoolingTower import evap_flow, discharge_temp, relative_gross_efficiency, fast_Twetbulb

class LoireEnvironmentalChecker:
    '''
    This class doesn’t contain any meaningful new information: it is a workaround to allow for parallelisation of the \
    simulation code using multiprocessing. 

    Initialising the PowerPlant objects for simulations requires to create EnvironmentalReg instances, which in turn requires to build the corresponding regulation functions. \
    As the Loire plants follow the same regulation patterns with varying withdrawal flow values, it seems logical to write a \
    build_environmental_checker(Q_wdmax) function, which returns a callable regulation function. However nested function \
    definitions like these do not fare well with multiprocessing: hence the workaround with this class, as callable classes \
    can be passed in multiprocessing (and named functions cannot). The class is only used to be called in the initialisation \
    of the plants, by the EnvironmentalReg.loire_regulation() function.

    '''

    def __init__(self, Q_wd_max, T_regulation = True, Q_regulation = True):
        self.Q_wd_max = Q_wd_max
        self.T_regulation = T_regulation
        self.Q_regulation = Q_regulation

    def __call__(self, Q=np.inf, Tw=-np.inf, T_outlet=-np.inf, Q_wd=0, Q_evap=0, **kwargs):
        if not self.Q_regulation and not self.T_regulation:
            return True
        
        elif self.Q_regulation and Q_wd > self.Q_wd_max:
            return False
        
        if not self.T_regulation:
            return True
        else: 
            Q_outlet = Q_wd - Q_evap
            T_downstream = (Q_outlet * T_outlet + (Q - Q_wd) * Tw) / (Q - Q_evap)
            DeltaT = T_downstream - Tw

            if Q < 100 and T_downstream < 15:
                return DeltaT <= 1.5
            else:
                return DeltaT <= 1
           
class EnvironmentalReg:
    '''
    The EnvironmentalReg class is a way to encode the set of environmental constraints for each plant in a single object, \
        accommodating for possibly ‘exotic’ regulation constraints (e.g. thresholds depending on upstream river flow, on DOY...). 

    A regulation set for a plant can be built from various float values representing usual constraint types found below, or from passing\
    a named regulation_function , which is 

    '''
    def __init__(
        self,
        regulation_function=None,
        DeltaT_river_max=None,
        T_outlet_max=None,
        T_downstream_max=None,
        Q_wd_max=None,
        Q_evap_max=None,
        Q_downstream_min=None,
        gamma=None
    ):
        self.isCustom = regulation_function is not None
        self.regulation_function = regulation_function

        # Neutral/default values
        defaults = {
            'DeltaT_river_max': np.inf,
            'T_outlet_max': np.inf,
            'T_downstream_max': np.inf,
            'Q_wd_max': np.inf,
            'Q_evap_max': np.inf,
            'Q_downstream_min': 0.0
        }

        # Set parameters based on custom or standard logic
        for attr, default in defaults.items():
            user_value = locals()[attr]
            if self.isCustom:
                setattr(self, attr, user_value)  # keep None if unspecified
            else:
                setattr(self, attr, user_value if user_value is not None else default)

        # Gamma logic
        if gamma is not None:
            self.gamma = float(gamma)
        elif not self.isCustom:
            # In standard case, decide based on effective constraints
            if (self.Q_wd_max == np.inf and 
                self.Q_evap_max == np.inf and 
                self.Q_downstream_min < 1e-3):
                self.gamma = 0.75  # default constraint if nothing else
            else:
                self.gamma = 1.0
        else:
            # In custom case, preserve gamma as None if unspecified
            self.gamma = gamma

    def __repr__(self):
        attrs = [
            f"DeltaT_river_max={self.DeltaT_river_max}",
            f"T_outlet_max={self.T_outlet_max}",
            f"T_downstream_max={self.T_downstream_max}",
            f"Q_wd_max={self.Q_wd_max}",
            f"Q_evap_max={self.Q_evap_max}",
            f"Q_downstream_min={self.Q_downstream_min}",
            f"gamma={self.gamma}",
            f"isCustom={self.isCustom}"
        ]
        return f"EnvironmentalReg({', '.join(attrs)})"
    
    @staticmethod
    def downstream_temp(Q : float ,Tw : float , q_wd : float ,q_evap : float , T_discharge : float ):
        Q_outlet = q_wd - q_evap
        T_downstream = (Q_outlet * T_discharge + (Q-q_wd)* Tw) / (Q-q_evap)
        return T_downstream

    @staticmethod
    def isEqual(reg1 : 'EnvironmentalReg', reg2 : 'EnvironmentalReg', tol : float = 1e-4):
        ''' 
        Checks equality between different instances of the class. 
        Handles floats with tolerances, including comparisons to np.inf.
        
        IMPORTANT:
        Doesn't implement an advanced equality check for regulation_function, simply an == comparing memory addresses.
        For instance, two deep copies of the same regulation will be returned unequal; shallow copies will work.
        '''
        if reg1.isCustom != reg2.isCustom:
            return False

        if reg1.isCustom:
            # Note: will return False for semantically identical functions at different memory addresses
            return reg1.regulation_function == reg2.regulation_function

        # Safe float comparison
        def float_eq(a, b, tol):
            if a == b:
                return True
            if np.isinf(a) and np.isinf(b) and np.sign(a) == np.sign(b):
                return True
            if np.isnan(a) and np.isnan(b):
                return True
            return abs(a - b) < tol

        return (
            float_eq(reg1.Q_wd_max, reg2.Q_wd_max, tol) and
            float_eq(reg1.Q_evap_max, reg2.Q_evap_max, tol) and
            float_eq(reg1.Q_downstream_min, reg2.Q_downstream_min, tol) and
            float_eq(reg1.gamma, reg2.gamma, tol) and
            float_eq(reg1.DeltaT_river_max, reg2.DeltaT_river_max, tol) and
            float_eq(reg1.T_outlet_max, reg2.T_outlet_max, tol) and
            float_eq(reg1.T_downstream_max, reg2.T_downstream_max, tol)
        )
    
    @staticmethod
    def loire_regulation(Q_wd_max : float, Q_regulation = True, T_regulation = True):
        return EnvironmentalReg(regulation_function=LoireEnvironmentalChecker(Q_wd_max, 
                                                                              Q_regulation=Q_regulation, 
                                                                              T_regulation=T_regulation))
    
    def is_compliant(self, **kwargs):

        """
        Check if a given set of operational parameters comply with regulations.
        If a custom function is set, it takes precedence.
        Otherwise, check standard constraints.
        Expects kwargs `Q`,`Tw`,`Q_wd`,T_outlet`
        """
        if self.isCustom and callable(self.regulation_function):
            return self.regulation_function(**kwargs)

        # Default logic (simplified)
        Q = kwargs.get('Q', np.inf)
        Tw = kwargs.get('Tw', -np.inf)
        Q_wd = kwargs.get('Q_wd', 0)
        Q_evap = kwargs.get('Q_evap', 0)
        T_outlet = kwargs.get('T_outlet', -np.inf)

        Q_outlet = Q_wd - Q_evap
        T_downstream = (Q_outlet * T_outlet + (Q-Q_wd)* Tw) / (Q-Q_evap)
        DeltaT = T_downstream - Tw

        if Q_wd > self.Q_wd_max:
            return False
        if Q_evap > self.Q_evap_max:
            return False
        if (Q - Q_wd) < self.Q_downstream_min:
            return False
        if T_outlet > self.T_outlet_max:
            return False
        if T_downstream > self.T_downstream_max:
            return False
        if DeltaT > self.DeltaT_river_max:
            return False
        if Q_wd > self.gamma * Q:
            return False

        return True
    
class PowerPlant:
    ########################## INITIALISATION #####################################################

    # Flexible plant type mapping
    TYPE_ALIASES = {
        'Coal': {'Lignite', 'Brown coal', 'Coal', 'Fossil hard coal'},
        'Nuclear': {'Nuclear'},
        'Gas simple cycle': {'Fossil gas combined cycle', 'CCGT', 'Fossil gas CC', 'Combined cycle gas turbine'},
        'Gas combined cycle': set(),
        'Gas': {'Natural gas', 'Gas', 'NG', 'NatGas', 'Fossil gas', 'Fossil Coal-derived gas'},
    }

    # Flexible cooling type mapping : no distinction made between natural and mechanical draught cooling towers
    COOLING_TYPE_ALIASES = {
        'Once-through': {'once through','open-loop', 'open-loop'},
        'Cooling tower': {'wet recirculation cooling','recirculating','closed loop','closed-loop', 'Mechanical Draught tower', 'Natural Draught tower'},
        'Air cooling': {'dry cooling', 'air-cooling','dry-cooling'}
    }

    @staticmethod
    def autofill_efficiency_capacity(capacity_net=None, capacity_gross=None, efficiency_net=None, efficiency_gross=None, plant_name=None):
        '''
        Helper function to fill in missing (None) gross/net efficiency or capacity values.
        '''
        # Ensure floats or None
        capacity_net = float(capacity_net) if capacity_net is not None else None
        capacity_gross = float(capacity_gross) if capacity_gross is not None else None
        efficiency_net = float(efficiency_net) if efficiency_net is not None else None
        efficiency_gross = float(efficiency_gross) if efficiency_gross is not None else None

        if efficiency_net is None and efficiency_gross is None:
            raise AttributeError(f'No efficiency information provided for plant {plant_name}')
        
        if capacity_net is None and capacity_gross is None:
            raise AttributeError(f'No capacity information provided for plant {plant_name}')
        
        if capacity_net is not None and efficiency_net is not None:
            if capacity_gross is None and efficiency_gross is None:
                capacity_gross = capacity_net
                efficiency_gross = efficiency_net
            elif capacity_gross is None:
                capacity_gross = capacity_net * efficiency_gross / efficiency_net
            elif efficiency_gross is None:
                efficiency_gross = efficiency_net * capacity_gross / capacity_net
            elif abs(capacity_gross / efficiency_gross - capacity_net / efficiency_net) > 1e-2:
                raise AttributeError(f'Provided net/gross efficiency and capacity mismatch for plant {plant_name}')
        
        elif capacity_net is None:
            capacity_net = capacity_gross
            efficiency_net = efficiency_net if efficiency_net is not None else efficiency_gross

        elif efficiency_net is None:
            efficiency_net = efficiency_gross * capacity_gross / capacity_net

        else:
            efficiency_net = efficiency_gross
            capacity_net = capacity_gross

        return capacity_net, capacity_gross, efficiency_net, efficiency_gross

    def _normalise_type(self, raw_type: str) -> str:
        f'''
        Matches the `'type'` string provided by user to the actual "canonical" `'type'` used inside the model, by looking up a dictionary.
        Example : the input string `'Fossil gas'` will directly map to the supported type `'Gas'`, `'Fossil coal` maps to `Coal` etc.

        For gas power plants, distinction between simple-cycle and combined cycle made on the following basis: efficiency >50
        List of type maps implemented :

        {PowerPlant.TYPE_ALIASES}
        '''
        raw_type_clean = raw_type.strip().lower()

        # Allow direct match to "canonical" type (ie the one used inside the model)
        for canonical in self.TYPE_ALIASES:
            if raw_type_clean == canonical.lower():
                if canonical == 'Gas':
                    return (
                        'Gas combined cycle'
                        if self.efficiency_net and self.efficiency_net >= 0.50
                        else 'Gas simple cycle'
                    )
                return canonical

        # If not found check the inner dictionary
        for standard, aliases in self.TYPE_ALIASES.items():
            normalised_aliases = {a.lower() for a in aliases}
            if raw_type_clean in normalised_aliases:
                if standard == 'Gas':
                    return (
                        'Gas combined cycle'
                        if self.efficiency_net and self.efficiency_net >= 0.50
                        else 'Gas simple cycle'
                    )
                return standard

        raise ValueError(f"Unknown or unsupported plant type: '{raw_type}'")

    @staticmethod
    def _normalise_cooling_type(raw_type : str) -> str :
        f'''
        Matches the `'cooling_type'` string provided by user to the actual "canonical" `cooling_type` used inside the model, by looking up a dictionary.
        Example : the input string `'Open-loop'` will directly map to the supported type `'Once-through'`

        List of type maps implemented :

        {PowerPlant.COOLING_TYPE_ALIASES}
        '''
        raw_type_clean = raw_type.strip().lower()

        # First: direct match with canonical keys
        for canonical in PowerPlant.COOLING_TYPE_ALIASES:
            if raw_type_clean == canonical.lower():
                return canonical

        # Then: check aliases
        for standard, aliases in PowerPlant.COOLING_TYPE_ALIASES.items():
            normalised_aliases = {a.lower() for a in aliases}
            if raw_type_clean in normalised_aliases:
                return standard

        raise ValueError(f"Unknown or unsupported cooling type: '{raw_type}'")

    def __init__(self,
                 name: str,
                 region: str,
                 lon: float,
                 lat: float,
                 type: str,
                 water_type: str,
                 cooling_type: str,
                 capacity_net: float = None,
                 capacity_gross: float = None,
                 efficiency_net: float = None,
                 efficiency_gross: float = None,
                 eic_p: str = None,
                 eic_g: str = None,
                 country: str = None,
                 environmental_reg: EnvironmentalReg = None,
                 TNET_reach: int = None,
                 ncc: float = None,
                 k_tower : float = None,
                 approach_temp : float = None,
                 P_min: float = None,
                 **kwargs):

        self.name = name
        self.region = region
        self.lat, self.lon = lat, lon
        self.country = country

        self.cooling_type = PowerPlant._normalise_cooling_type(cooling_type)
        self.water_type = water_type
        self.eic_p = eic_p
        self.eic_g = eic_g

        # Flexible capacity and efficiency attribution based on given info
        (self.capacity_net, 
        self.capacity_gross, 
        self.efficiency_net, 
        self.efficiency_gross) = self.autofill_efficiency_capacity(
            capacity_net, capacity_gross, efficiency_net, efficiency_gross, plant_name=name)
        
        ##### Type normalisation
        self.type = self._normalise_type(type)

        ##### Region specific behaviour
        match region:
            case 'Loire':
                self.TNET_reach = int(TNET_reach) if TNET_reach is not None else None           # Hydro data
                self.SAFRAN_point = findClosestStation(self.lon, self.lat, region=region)       # Atm data
            
            case 'Rhine':    
                self.river_km = get_rhine_km_id(self.lat,self.lon)
                if self.river_km is not None:
                    self.Q_station = findClosestUpstreamStation(self.river_km, 'Rhine', 'Q', plant_name= self.name)
                    self.Tw_station = findClosestUpstreamStation(self.river_km, 'Rhine', 'Tw', plant_name= self.name)

            case 'Elbe':
                self.river_km = get_elbe_km(self.lat, self.lon)
                if self.river_km is not None:
                    self.Q_station = findClosestUpstreamStation(self.river_km, 'Elbe', 'Q', plant_name= self.name)
                    self.Tw_station = findClosestUpstreamStation(self.river_km, 'Elbe', 'Tw', plant_name= self.name)
                
        ##### Environmental regulation
        self.environmental_reg = environmental_reg or EnvironmentalReg(
            DeltaT_river_max=3, T_downstream_max=28, T_outlet_max=30, gamma=0.75
        )

        ##### Operating power threshold
        self.P_min = float(P_min) if P_min is not None else (0.2 * self.capacity_net if self.capacity_net else None)

        ##### Cooling tower parameters
        self.k_tower = k_tower if k_tower else 0.98
        self.approach_temp = approach_temp if approach_temp else 11

        if ncc is not None:
            self.ncc = ncc
        elif self.cooling_type == 'Cooling tower':
            self.ncc = 1.5 if self.type == 'Nuclear' else 3
        else:
            self.ncc = None

    def __repr__(self):
        return (
            f"<PowerPlant '{self.name}' | Type: {self.type} | Net Capacity: {self.capacity_net} MW | "
            f"Efficiency: {self.efficiency_net:.2%} | Location: ({self.lat:.4f}, {self.lon:.4f}) | "
            f"Cooling: {self.cooling_type} ({self.water_type})>"
        )
    
    @staticmethod
    def isCompatible(plant1 : 'PowerPlant', plant2 : 'PowerPlant'):
        '''
        Checks if a plant is compatible with another plant to be lumped together in a clustered setting. 

        Checks equality between plant type, water type, cooling technology, net and gross efficiencies, water consumption factor ncc.

        For minimum power threshold : checks equality between relative power thresholds, ie P_min/capacity_net 

        For environmental regulation : see caveats about custom regulations in EnvironmentalReg.isEqual.
        
        Checks plants are from same region.'''

        try :
            return (plant1.region == plant2.region and
                    plant1.type == plant2.type and
                    plant1.water_type == plant2.water_type and
                    plant1.cooling_type == plant2.cooling_type and

                    abs(plant1.efficiency_net - plant2.efficiency_net) < 1e-3 and
                    abs(plant1.efficiency_gross - plant2.efficiency_gross) < 1e-3 and
                    ((plant1.ncc is plant2.ncc) or abs(plant1.ncc - plant2.ncc) < 1e-3) and

                    abs(plant1.P_min/plant1.capacity_net - plant2.P_min/plant2.capacity_net) < 1e-3 and

                    EnvironmentalReg.isEqual(plant1.environmental_reg, plant2.environmental_reg))
        except TypeError as e:
            print(f'Warning : attribute type compatibility mismatch detected when comparing plants {plant1.name} and {plant2.name}. Returned False')
            print(f'Plant 1 : {plant1}')
            print(f'Plant 2 : {plant2}')
            return False #if matching none with a number, returns False

    @staticmethod
    def loadFromCSV(filepath: str, sep=';'):
        df = pd.read_csv(filepath, sep=sep)
        # print('Debug :')
        # print(df)

        # Get list of __init__ parameters (excluding self and kwargs)
        init_params = inspect.signature(PowerPlant.__init__).parameters
        valid_args = {k for k in init_params if k != 'self' and init_params[k].kind != inspect.Parameter.VAR_KEYWORD}

        # Mapping for fuzzy column name matching
        COLUMN_ALIASES = {
            'eic_p': ['eicp', 'eic_p', 'eic_production'],
            'eic_g': ['eicg', 'eic_g', 'eic_generation'],
            'name': ['plant_name', 'name_g', 'plant'],
            'capacity_net': ['cap_net', 'net_capacity', 'capacity_n', 'cap_n','capacity_g'],
            'capacity_gross': ['cap_gross', 'gross_capacity'],
            'efficiency_net': ['eff_net', 'net_eff', 'eta_net','eff'],
            'efficiency_gross': ['eff_gross', 'gross_eff', 'eta_gross', 'gross_efficiency'],
            'lat': ['latitude'],
            'lon': ['longitude'],
            'type': ['type_g', 'plant_type', 'fuel'],
            'region': [],
            'water_type': ['water_source', 'source_water'],
            'cooling_type': ['cooling', 'cooling_system', 'cooling_tech'],
            'country': [],
            'TNET_reach': ['tnet', 'reach_id','hydro_id'],
            'ncc': ['n_cc'],
            'P_min': ['minimum_power', 'pmin', 'p_min']
        }

        # Invert alias dictionary for fast lookup
        flat_aliases = {}
        for std_name, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                flat_aliases[alias.strip().lower()] = std_name

        # Add canonical keys as their own aliases to catch direct matches (e.g., "region")
        for key in COLUMN_ALIASES.keys():
            flat_aliases[key.lower()] = key

        # Determine which CSV columns will not be used
        used_keys = set(valid_args)
        used_keys.update(flat_aliases.values())
        csv_columns_lower = [col.strip().lower() for col in df.columns]
        
        unused_columns = []
        for col in csv_columns_lower:
            if col not in valid_args and col not in flat_aliases:
                unused_columns.append(col)

        if unused_columns:
            print("Warning: The following CSV columns could not be matched and will be ignored for initialisation:")
            for col in unused_columns:
                print(f"   - {col}")

        # Parse rows
        plants = []
        for _, row in df.iterrows():
            init_kwargs = {}
            for col in row.index:
                col_clean = col.strip().lower()

                # Determine key to use
                if col_clean in valid_args:
                    key = col_clean
                elif col_clean in flat_aliases:
                    key = flat_aliases[col_clean]
                else:
                    continue

                if key in valid_args and pd.notnull(row[col]):
                    init_kwargs[key] = row[col]

            # print(f"Debug: Initializing with keys: {list(init_kwargs.keys())}")
            # print(f"Debug: Initializing with values: {list(init_kwargs.values())}")
            try:
                plant = PowerPlant(**init_kwargs)
                plants.append(plant)
            except Exception as e:
                print(f"Error initializing plant '{init_kwargs.get('name', '[Unknown]')}' — {e}")

        for plant in plants:
            plant.name = plant.name.replace(" ","_")
        print('End of plant initialisation from CSV file.')
  
        return plants

    ############################# SIMULATION : POWER AVAILABILITY ################################

    # Helper functions
    def checkEnvironmentalCompliance(self, **kwargs):
        '''
        Expects keyword arguments under format Q, Tw, Q_evap, Q_wd, T_outlet, day.
        '''

        return self.environmental_reg.is_compliant(**kwargs)
    
    def getIndividualEfficienciesCCGT(self):
        ''' 
        Decomposes the overall efficiency information provided for a CC plant (`type` : `'Gas combined cycle'`) in gas cycle, steam cycle and recovery efficiencies.

        **Outputs**
        `gas_eff` (`float`), `steam_eff` (`float`), `recovery_eff` (`float`)

        Relies on assumptions detailed below for now. Could be modified by implementing more detailed attributes in plant initialisation.

        Gross efficiency is given by the following formula:
        eta(CCG) = eta(gas) + (1- eta(gas)) * epsilon * eta(steam). 
        
        Assumptions : 

        - Simple-cycle gas turbine efficiency is set at 0.4 
        - Recovery efficiency is assumed to be fixed at a median value of 87%, based on the indicated 80-95% range in source below
        - Steam turbine efficiency is deduced to match the provided `plant.efficiency` value.

        Source : 2021-03-10-GT_climat_4_nucléaire_vclean-min (https://www.concerte.fr/system/files/u12200/2021-03-10-GT_climat_4_nucl%C3%A9aire_vclean-min.pdf)
        '''
        if self.type != 'Gas combined cycle':
            raise ValueError('Efficiency decomposition only available for CCGT turbines')
        
        global_efficiency = self.efficiency_gross

        ## Default method here but in future implementation, check beforehand if some actual parameters have been provided ?
        gas_eff = 0.4
        epsilon = 0.87
        steam_eff = (global_efficiency - gas_eff)/(epsilon * (1- gas_eff)) 
        return gas_eff, steam_eff, epsilon

    def adjustedGrossEfficiency(plant, cooling_water_temp: float =None, air_temp : float = None):
        ''' Quantifies performance decrease with increasing coolant temperature (cooling water or air), based on plant type. 
        
        - For Rankine-cycle plants (`type` : `'Coal'` or `'Nuclear'`), steam turbine relative efficiency decrease based on water temperature 
        was modelled using the reported values for the Golfech nuclear power plant (source A below), scaled in terms of relative efficiency.

        - For simple Brayton-cycle gas plants (`type` : `'Gas simple cycle'`), turbine efficiency is calculated by implementing a 0.6% /K 
        relative efficiency decrease with intake air temperature (see source B below), based on a reference temperature of 16°C.

        - For combined-cycle power plants (`type` : `'Gas combined cycle'`), efficiencies are decomposed according to the above 
        `getIndividualEfficienciesCCGT`function. 
        Steam turbine efficiency is then calculated using the above method for Rankine cycles : if no cooling water temperature is indicated, air temperature is taken as a proxy, which yields satisfying results as mentioned in C.
        Gas turbine efficiency is then calculated by using the above method for Brayton cycles.

        **Input parameters :**

        - `plant` (`PowerPlant`) : plant to simulate
        - `cooling_water_temp` (`float`) : cooling water temperature in °C
        - `air_temp` (`float`) : air temperature in °C

        **Output:**

        - `float` : Plant's adjusted gross efficiency under current operating conditions 

        **Sources** : 

        A) Supplementary information to : Guenand et al. (10.1016/j.energy.2024.132648)

        B) GE Gas Turbine – Performance Characteristics (https://www.gevernova.com/content/dam/gepower-new/global/en_US/downloads/gas-new-site/resources/reference/ger-3567h-ge-gas-turbine-performance-characteristics.pdf)
        
        C) 2021-03-10-GT_climat_4_nucléaire_vclean-min (https://www.concerte.fr/system/files/u12200/2021-03-10-GT_climat_4_nucl%C3%A9aire_vclean-min.pdf)
        '''

        efficiency = plant.efficiency_gross

        match plant.type:
            ## Rankine cycle : values for condenser performance taken from the reported Golfech performance values
            case 'Nuclear' | 'Coal':
                if plant.cooling_type == 'Air Cooling':
                    print('WARNING: Air cooled Rankine-cycle efficiency decrease not implemented: returning initial value.')
                    return efficiency
                
                if cooling_water_temp is None:
                    raise ValueError('No cooling water temperature input for steam turbine efficiency calculation: Rankine cycle assumed to be water-cooled')
                return efficiency * relative_gross_efficiency(cooling_water_temp)
            
            case 'Gas simple cycle':
                if air_temp is None :
                    raise ValueError('No air temperature provided for gas combustion turbine efficiency calculation')
                
                return efficiency * (1-0.006 * (air_temp - 16))
            
            ## Natural gas : combined cycle. If no assumption made, water temperature assumed to be equal to air temperature. 
            # Reference gas cycle efficiency taken as 0.4 for simplicity, steam turbine efficiency calculated from there.

            case 'Gas combined cycle':
                gas_eff_0, steam_eff_0, epsilon = plant.getIndividualEfficienciesCCGT()

                if cooling_water_temp is None:
                    cooling_water_temp = air_temp
                
                gas_eff = gas_eff_0 * (1-0.006 * (air_temp - 16))
                steam_eff = steam_eff_0 * relative_gross_efficiency(cooling_water_temp)

                return gas_eff + (1-gas_eff) * epsilon * steam_eff
            
            case _ :
                print(f'No efficiency decrease calculation supported for type : {plant.type}')
                return efficiency

    def availableCapacityFromRiverLoad(plant, P_river, cooling_water_temp = None, Tair = None):
        '''
        For a given maximal acceptable river cooling load, deduces corresponding maximal plant net and gross outputs.

        Assumes plant thermal power is capped at its value in nameplate case : capacity_net/efficiency_net.

        **Inputs:**
        - `P_river`: maximum acceptable river thermal load in MW
        - `Tw` : cooling water temperature in °C for efficiency calculation (normally : river water temperature)
        - `Tair` : cooling air temperature in °C for efficiency calculation

        **Returns:**
        `(P_gross , P_net, P_toriver)`: gross/net electrical output (MW) and actual cooling load (MW) in maximal case
        '''
        net_capacity = plant.capacity_net
        gross_capacity = plant.capacity_gross
        self_consumption = gross_capacity - net_capacity
        gross_efficiency = plant.efficiency_gross
        adj_gross_efficiency = PowerPlant.adjustedGrossEfficiency(plant, cooling_water_temp=cooling_water_temp, air_temp = Tair)

        P_thermal_max = gross_capacity/gross_efficiency

        if plant.type == 'Gas combined cycle':
            # For CC plants, part of the cooling load is discharged to the air and part to the river, based on recovery efficiency

            _ , eta_steam, epsilon = plant.getIndividualEfficienciesCCGT()
            eta_steam *= relative_gross_efficiency(cooling_water_temp)
            P_cooling = P_river * (1- epsilon * eta_steam) / (epsilon * (1- eta_steam))        # air + water
        
        else:
            P_cooling = P_river

        available_gross_capacity = min(P_thermal_max, P_cooling/(1-adj_gross_efficiency)) * adj_gross_efficiency
        available_net_capacity = available_gross_capacity - self_consumption

        # Actual cooling load may differ due to thermal power capping
        actual_P_cooling = available_gross_capacity * (1 - adj_gross_efficiency)/adj_gross_efficiency 

        P_toriver = (actual_P_cooling * epsilon * (1- eta_steam)/(1- epsilon * eta_steam) 
                     if plant.type == 'Gas combined cycle'
                     else actual_P_cooling)

        return available_gross_capacity, available_net_capacity, P_toriver

    def outputPowersFromPrimaryPower(plant, P_thermal : float, cooling_water_temp :float = None, Tair : float = None):
        '''
        For a given plant primary power level, calculates plant gross/net electrical outputs P_gross, P_net.
        Calculates cooling load to river P_toriver.
        **Inputs :**

        `P_thermal` : primary thermal power (MW)
        `Tw`: cooling water temperature

        **Returns :**

        `P_gross`, `P_net`, `P_toriver`
        '''

        net_capacity = plant.capacity_net
        gross_capacity = plant.capacity_gross
        self_consumption = gross_capacity - net_capacity
    
        adj_gross_efficiency = PowerPlant.adjustedGrossEfficiency(plant, cooling_water_temp=cooling_water_temp, air_temp = Tair)

        P_el_gross = P_thermal * adj_gross_efficiency
        P_el_net = P_el_gross - self_consumption

        if plant.type == 'Gas combined cycle':
            # For CC plants, part of the cooling load is discharged to the air and part to the river, based on recovery efficiency

            _ , eta_steam, epsilon = plant.getIndividualEfficienciesCCGT()
            eta_steam *= relative_gross_efficiency(cooling_water_temp)
            P_toriver = (P_thermal - P_el_gross) * epsilon * (1- eta_steam)/(1- epsilon * eta_steam)
        
        else :
            P_toriver = P_thermal - P_el_gross
        
        return P_el_gross, P_el_net, P_toriver

    def checkOnceThroughPowerCompatibility(plant,Q : float, Tw : float, P_cooling: float, day :str = None,
                                           T_outlet_upper_bound : float = 60, coarse_steps : int = 30, fine_steps : int = 100):
        r'''
        For once-through cooled power plants with a custom environmental ruleset (`plant.environmental_reg.isCustom == True`), 
        this function checks if a power level is compatible with regulations. This implies finding a (`Q_wd`, `T_outlet`) state compatible 
        with the plant's regulation.

        If some thresholds stay fixed and are provided by the user in the `EnvironmentalReg` object (aside from the `regulation_function`), 
        they are used to accelerate the check:

        - If all streamflow regulation parameters `Q_wd_max`, `gamma`, `Q_downstream_min` are specified, it is assumed that the streamflow 
        side of regulations is standard. The maximal withdrawable flow `Q_available` can then be calculated. Only the state with 
        this flow value will be checked, as it yields the most favorable outlet temperature.
        - Similarly, if all temperature parameters `T_outlet_max`, `T_downstream_max`, `DeltaT_river_max` are specified, it is assumed that 
        the temperature side of regulation is standard. Thus, only the maximal `T_outlet` value is checked, as it yields the lowest `Q_wd` 
        value.
        - Otherwise, a compatible pair is searched by running through possible outlet temperature values between `Tw` and `T_outlet_upper_bound`. 
        To improve performance, this search is performed in two steps : a coarse search, and a fine search if no matching value has been found yet.

        The relationship between `Q_wd` and `T_outlet`, at a given thermal load, is as follows:

        $$
        T_{outlet} = Tw + \frac{P_{cooling}}{\rho_w C_p Q_{wd}}
        $$

        This function is designed to be used within the `powerAvailabilityOnceThrough` function, and assumes that:
        - `Q` exceeds `Q_downstream_min` if defined
        - `Tw` doesn't exceed `T_downstream_max` or `T_outlet_max` whenever these are defined.  If both the latter are defined, the power level is also assumed to be compatible with river-scale temperature regulations.

        ### Parameters

        - `plant` (`PowerPlant`): Instance of the plant  
        - `Q` (`float`): Streamflow in m³/s  
        - `Tw` (`float`): Water temperature in °C  
        - `P_cooling` (`float`): Cooling thermal load in MW
        - `T_outlet_upper_bound` (`float`): Upper bound for the search of a compatible outlet temperature (°C)
        - `coarse_steps` (`int`) : number of steps for the first coarse outlet temperature search
        - `fine_steps`(`ìnt`) : number of steps for the fine outlet temperature search

        ### Returns

        - `bool`: True if a compliant (Q_wd, T_outlet) pair is found
        '''
        
        # Plant parameters
        reg  = plant.environmental_reg

        max_outlet_temp = reg.T_outlet_max
        max_DeltaT = reg.DeltaT_river_max
        max_downstream_temp = reg.T_downstream_max
        Q_wd_max = reg.Q_wd_max
        Q_downstream_min = reg.Q_downstream_min
        gamma = reg.gamma

        if max_outlet_temp is not None and max_downstream_temp is not None and max_DeltaT is not None:
            # Compatibility with the latter two parameters has already been checked beforehand in the parent code
            # If not even max outlet temp yields a compliant Q_wd then no other value will
            Q_wd = P_cooling *1e6 /((max_outlet_temp - Tw)* rho_w * c_p)
            return reg.is_compliant(Q=Q, Tw= Tw, Q_wd = Q_wd, T_outlet = max_outlet_temp, day=day)
        
        if Q_wd_max is not None and Q_downstream_min is not None and gamma is not None:
            # If streamflow regs are standard, highest Q_wd can be calculated. This is the most favorable T_outlet value
            Q_available = min(Q_wd_max, Q - Q_downstream_min, gamma*Q)
            T_outlet = Tw + P_cooling *1e6 /(Q_available * rho_w * c_p)

            return reg.is_compliant(Q=Q, Tw=Tw, T_outlet=T_outlet, Q_wd = Q_available, day = day)
        
        else:
            # Otherwise a compliant T_outlet value is searched. 
            # To improve runtime, first a coarse check is performed, and if no matching value has been found, a fine check is ran to confirm the power value is unacceptable.

            for T_outlet in np.linspace(Tw+0.01, T_outlet_upper_bound, coarse_steps):
                Q_wd = P_cooling * 1e6 /((T_outlet - Tw)* rho_w * c_p)
                if reg.is_compliant(Q=Q, Tw= Tw, Q_wd = Q_wd, T_outlet = T_outlet, day = day):
                    return True

            for T_outlet in np.linspace(Tw+0.01, T_outlet_upper_bound, fine_steps):
                Q_wd = P_cooling * 1e6 /((T_outlet - Tw)* rho_w * c_p)
                if reg.is_compliant(Q=Q, Tw= Tw, Q_wd = Q_wd, T_outlet = T_outlet, day = day):
                    return True

        return False

    def coolingTowerBasinTemperature(plant, Tair : float, RH : float, dynamicTower : bool):

        return discharge_temp(Tair,RH) if dynamicTower else fast_Twetbulb(Tair, RH) + plant.approach_temp
    
    def coolingTowerRiverImpact(plant, Q : float, Tw : float, Tair : float, RH : float, T_basin : float, P_cooling: float, dynamicTower : bool):
        ncc = plant.ncc
        if dynamicTower:
            q_evap = evap_flow(Tair, RH, P_cooling)
        else:
            k_tower = plant.k_tower
            k_latent = (1-k_tower)/k_tower * L_vap * (ncc-1)/ (c_p * (T_basin-Tw))
            q_evap = (P_cooling)*k_tower*k_latent/(rho_w*c_p*L_vap)

        q_wd = ncc / (ncc - 1) * q_evap
        T_downstream = EnvironmentalReg.downstream_temp(Q,Tw,q_wd,q_evap, T_basin)

        return q_wd, q_evap, T_downstream

    # Simulation per cooling type
    def powerAvailabilityOnceThrough(plant, Q : float, Tw : float, 
                                     Tair = None, day : str = None,
                                     enforce_Pmin : bool = True, result_precision_MW = 5, **kwargs):
        '''
        Calculates power availability for a once-through (open loop) cooling plant based on environmental conditions.

        If the plant has a standard ruleset (i.e. only fixed thresholds), available power is calculated directly by calculating maximal river thermal load. 

        Otherwise, the maximal primary energy input value is determined by binary search. 
        For each value, a valid (Q_wd, T_outlet) state is searched using `checkOnceThroughPowerCompatibility`: 
        if no valid couple is found, the corresponding primary power level is rejected.

        **Input parameters:**  
        - `plant` (PowerPlant): An instance of the `PowerPlant` class representing the plant to simulate  
        - `Q` (float): River streamflow in m³/s  
        - `Tw` (float): Water temperature in °C  

        **Returns:**  
        - `float`: Available net capacity in MW
        '''
        # Plant parameters
        net_capacity = plant.capacity_net
        gross_capacity = plant.capacity_gross
        self_consumption = gross_capacity - net_capacity
        gross_efficiency = plant.efficiency_gross
        adj_gross_efficiency = PowerPlant.adjustedGrossEfficiency(plant, cooling_water_temp=Tw)
        
        P_min = plant.P_min if enforce_Pmin else 0

        reg  = plant.environmental_reg
        max_outlet_temp = reg.T_outlet_max
        max_DeltaT = reg.DeltaT_river_max
        max_downstream_temp = reg.T_downstream_max
        Q_wd_max = reg.Q_wd_max
        Q_downstream_min = reg.Q_downstream_min
        gamma = reg.gamma

        # If no custom ruleset, available power is calculated directly from the maximum thresholds
        if not reg.isCustom:
            
            # Cooling load calculation : calculate max cooling load around plant (at the outlet and from river perspective)
            Q_available = min(Q_wd_max, Q - Q_downstream_min, gamma*Q)
            
            allowed_downstream_DeltaT = min(max_DeltaT, max(max_downstream_temp - Tw,0))
            allowed_plant_DeltaT = max(0, max_outlet_temp - Tw)

            allowed_river_load = rho_w * c_p *\
                min(Q * allowed_downstream_DeltaT, Q_available * allowed_plant_DeltaT) /1e6             # in MW

            _ , P_net, P_toriver = plant.availableCapacityFromRiverLoad(allowed_river_load, cooling_water_temp=Tw, Tair=Tair)

            ## Calculation of river state afterwards : 
            # Q is unchanged, Tw deduced from input cooling load
        
            if P_net < P_min :
                return 0, Q, Tw

            else :
                T_downstream = Tw + P_toriver /(rho_w*c_p)

            return P_net,Q,T_downstream
        
        # For a custom ruleset, maximum cooling load can't be computed as thresholds may vary and aren't known.
        # Binary search for highest primary energy input. For each value, there is an additional degree of freedom on Q_wd so a correct operation state is searched
        # If some thresholds are given, they are used to accelerate the search process: e.g. if all streamflow conditions are given, only maximal possible withdrawal is tested as it will yield the lowest temperature rise.
        else:   
            if Q_downstream_min is not None and Q < Q_downstream_min:
                return 0, Q, Tw

            if max_downstream_temp is not None and Tw >= max_downstream_temp:
                return 0, Q, Tw
            
            if max_outlet_temp is not None and Tw >= max_outlet_temp:
                return 0, Q, Tw
            
            P_thermal_max = gross_capacity / gross_efficiency

            # If max cooling load already exceeds known river temp regulations, adjust starting max power to match
            if max_downstream_temp is not None and max_DeltaT is not None:
                allowed_downstream_DeltaT = min(max_DeltaT, max(max_downstream_temp - Tw,0))
                allowed_river_load = allowed_downstream_DeltaT * rho_w * c_p * Q

                _,_, P_toriver = plant.outputPowersFromPrimaryPower(P_thermal_max, cooling_water_temp=Tw, Tair = Tair)
                
                if P_toriver > allowed_river_load:
                    P_el_gross, *_ = plant.availableCapacityFromRiverLoad(allowed_river_load, cooling_water_temp=Tw, Tair = Tair)
                    P_thermal_max = P_el_gross/adj_gross_efficiency

            P_thermal_min = (P_min + self_consumption)/adj_gross_efficiency

            # In case the above check on river temperature already means operation is impossible :
            if P_thermal_max < P_thermal_min:
                return 0, Q, Tw
            
            # After these coherence checks the rest of the algorithm can start

            precision = result_precision_MW/adj_gross_efficiency  # precision on thermal power based on desired precision on output power

            # First check if full power OK
            P_thermal = P_thermal_max
            _, P_el_net, P_toriver = plant.outputPowersFromPrimaryPower(P_thermal, cooling_water_temp= Tw, Tair= Tair)

            if PowerPlant.checkOnceThroughPowerCompatibility(plant=plant, Q=Q, Tw=Tw, P_cooling = P_toriver, day = day):
                if P_el_net < P_min :
                    return 0, Q, Tw
                else :
                    T_downstream = Tw + P_toriver /(rho_w*c_p)
                    return P_el_net, Q, T_downstream

            # Then check if plant operates at all
            P_thermal = P_thermal_min
            _, P_el_net, P_toriver = plant.outputPowersFromPrimaryPower(P_thermal, cooling_water_temp= Tw, Tair= Tair)

            if not PowerPlant.checkOnceThroughPowerCompatibility(plant=plant, Q=Q, Tw=Tw, P_cooling=P_toriver, day = day):
                return 0, Q, Tw

            # Then binary search for the threshold value if intermediate
            best_P_el_net, best_P_toriver = P_min, P_toriver

            while P_thermal_max - P_thermal_min > precision:

                P_thermal = (P_thermal_max + P_thermal_min) / 2
                _, P_el_net, P_toriver = plant.outputPowersFromPrimaryPower(P_thermal, cooling_water_temp= Tw, Tair= Tair)

                if PowerPlant.checkOnceThroughPowerCompatibility(plant=plant, Q=Q, Tw=Tw, P_cooling=P_toriver, day = day):
                    best_P_el_net, best_P_toriver = P_el_net, P_toriver
                    P_thermal_min = P_thermal  # try higher power
                else:
                    P_thermal_max = P_thermal  # try lower power

            T_downstream = Tw + best_P_toriver/(rho_w*c_p)

            return best_P_el_net, Q, T_downstream

    def powerAvailabilityCoolingTower(plant, Q : float , Tw : float , Tair : float , RH : float , day : str = None,
                                      dynamicTower: bool = True, enforce_Pmin : bool = True, result_precision_MW = 5):
        
        T_basin = plant.coolingTowerBasinTemperature(Tair, RH, dynamicTower)

        self_consumption = plant.capacity_gross - plant.capacity_net
        gross_eff_adj = plant.efficiency_gross * relative_gross_efficiency(T_basin)
        P_thermal_max = plant.capacity_net / plant.efficiency_net  

        P_min = plant.P_min if enforce_Pmin else 0
        
        # Minimum input thermal power, based on minimal net output
        P_thermal_min = (P_min+self_consumption)/gross_eff_adj

        precision = result_precision_MW/gross_eff_adj  # precision on thermal power based on precision on output power

        # First check if full power OK
        P_thermal = P_thermal_max
        _ , P_el_net , P_toriver = plant.outputPowersFromPrimaryPower(P_thermal,cooling_water_temp= T_basin, Tair=Tair)

        q_wd, q_evap, T_downstream = plant.coolingTowerRiverImpact(Q, Tw, Tair, RH, T_basin, P_toriver, dynamicTower)

        if plant.checkEnvironmentalCompliance(Q=Q, Tw=Tw, T_outlet=T_basin, Q_evap=q_evap, Q_wd=q_wd, day = day):
            return P_el_net, Q-q_evap, T_downstream

        # Then check if plant operates at all
        P_thermal = P_thermal_min
        _ , P_el_net , P_toriver = plant.outputPowersFromPrimaryPower(P_thermal,cooling_water_temp= T_basin, Tair=Tair)

        q_wd, q_evap, T_downstream = plant.coolingTowerRiverImpact(Q, Tw, Tair, RH, T_basin, P_toriver, dynamicTower)

        if not plant.checkEnvironmentalCompliance(Q=Q, Tw=Tw, T_outlet=T_basin, Q_evap=q_evap, Q_wd=q_wd, day = day):
            return 0, Q, Tw

        best_P_el_net, best_T_downstream = P_min, Tw      #Minimal case

        # Then binary search for the threshold value if intermediate
        while P_thermal_max - P_thermal_min > precision:
            P_thermal = (P_thermal_max + P_thermal_min) / 2

            _ , P_el_net , P_toriver = plant.outputPowersFromPrimaryPower(P_thermal,cooling_water_temp= T_basin, Tair=Tair)

            q_wd, q_evap, T_downstream = plant.coolingTowerRiverImpact(Q, Tw, Tair, RH, T_basin, P_toriver, dynamicTower)

            if plant.checkEnvironmentalCompliance(Q=Q, Tw=Tw, T_outlet=T_basin, Q_evap=q_evap, Q_wd=q_wd, day = day):
                best_P_el_net, best_T_downstream = P_el_net, T_downstream
                P_thermal_min = P_thermal  # try higher power
            else:
                P_thermal_max = P_thermal  # try lower power

        return best_P_el_net, Q-q_evap, best_T_downstream    

    def powerAvailabilityAirCooling(plant, Tair : float,
                                    enforce_Pmin : bool = True, **kwargs):
        '''
        Returns air-cooled power plant availability : only efficiency decrease is taken into account. 
        Assumes that primary power cannot exceed nameplate primary power (ie under max capacity and nameplate efficiency).

        **Input parameters:**
        
        - `plant`(`PowerPlant`) : plant to be modelled
        - `Tair` (`float`) : air temperature (°C)

        **Outputs:**

        - `float` : plant availability in MW
        '''

        net_capacity = plant.capacity_net
        gross_capacity = plant.capacity_gross
        self_consumption = gross_capacity - net_capacity
        adj_gross_efficiency = PowerPlant.adjustedGrossEfficiency(plant, air_temp=Tair)

        P_min = plant.P_min if enforce_Pmin else 0

        P_th_0 = gross_capacity/plant.efficiency_gross
        max_P_el_gross = P_th_0 * adj_gross_efficiency

        if max_P_el_gross - self_consumption < P_min :
            return 0
        else :
            return max_P_el_gross - self_consumption

    # Overarching function
    def powerAvailability(plant, Q: float =None, Tw : float  = None, Tair : float = None, RH : float  = None, day : str = None,
                           dynamicTower : bool = True, enforce_Pmin : bool = True):

        """
        This function returns the maximal available capacity for a power plant for given environmental conditions. Selects one of the previous availability function depending on plant cooling type. 
        
        Not all parameters are needed depending on plant type : e.g. atmospheric parameters are not needed for a once-through plant, and hydrology is not needed for an air-cooled plant. 
        All parameters are passed to the functions anyway as keyword arguments for future flexibility (e.g. advanced efficiency calculation depending on RH for instance).

        **Input parameters:**

        - `plant` (`PowerPlant`): plant to be simulated
        - `Q` (`float`) : upstream river flow (m3/s)
        - `Tw` (`float`) : upstream river temperature (°C)
        - `Tair` (`float`) : ambient air temperature (°C)
        - `RH`(`float`) : relative humidity (%)
        
        """
        if plant.water_type == 'Seawater':
            return plant.capacity_net

        cooling_type = plant.cooling_type

        if cooling_type not in PowerPlant.COOLING_TYPE_ALIASES.keys():
            raise ValueError(f"Wrong cooling argument : expected among {PowerPlant.COOLING_TYPE_ALIASES.keys()}")

        match cooling_type:
            case 'Air cooling':
                
                return PowerPlant.powerAvailabilityAirCooling(plant,Tair=Tair, RH=RH, 
                                                              enforce_Pmin=enforce_Pmin), Q, Tw
            
            case 'Once-through':

                return PowerPlant.powerAvailabilityOnceThrough(plant, Q=Q ,Tw=Tw, Tair=Tair, RH=RH, day = day,
                                                               enforce_Pmin=enforce_Pmin)
                
            case 'Cooling tower':

                return PowerPlant.powerAvailabilityCoolingTower(plant, Q=Q, Tw = Tw, Tair= Tair, RH= RH, day = day,
                                                                dynamicTower=dynamicTower, enforce_Pmin=enforce_Pmin)
               
    #################################### ENERGY BALANCE ############################################

    def computeCoolingTowerEnergyBalance(plant,Q : float =None,Tw : float =None,Tair : float =None,RH : float =None,
                                         printBalance : bool = False):
        '''
        Computes the energy balance for the dynamic cooling tower model at a set operating point.

        **Input parameters**

        - `plant` (`PowerPlant`) : the plant to be modelled
        - `Q` (`float`) : upstream streamflow (m3/s)
        - `Tw`(`float`) : upstream water temperature (°C)
        - `Tair` (`float`) : air temperature (°C)
        - `RH` (`float`) : relative humidity (%)

        **Outputs**

        A (`k_river`,`k_evap`) tuple, where :

        - `k_river` (`float`) : the ratio of cooling load dissipated through blowdown heating of the river
        - `k_evap`(`float`) : the ratio latent heat (vaporisation enthalpy) to overall energy dissipated in the tower (ie to (1-k_river)* P_cooling )

        Notes : 
        
        - `k_river` may be negative if blowdown discharge temperature is lower than upstream water temperature. 
        - `k_evap` may be larger than 1 if the ambient air is hotter than the water : in this case, the algebraic value for sensible heat is negative and latent heat is greater than overall load to the cooling tower
        '''
        efficiency_gross = plant.efficiency_gross
        P_thermal = plant.capacity_gross/plant.efficiency_gross
        ncc = plant.ncc
        
        T_basin = discharge_temp(Tair, RH)
        gross_eff_adj = efficiency_gross * relative_gross_efficiency(T_basin)
        P_el_gross = P_thermal * gross_eff_adj

        q_evap = evap_flow(Tair, RH, P_thermal - P_el_gross)
        q_blowdown = q_evap / (ncc - 1)

        T_downstream = EnvironmentalReg.downstream_temp(Q, Tw, q_evap + q_blowdown, q_evap, T_basin)

        P_toriver = rho_w * c_p * q_blowdown * (T_basin - Tw) / 1e6
        P_tower = P_thermal - P_el_gross - P_toriver
        P_vap = rho_w * L_vap * q_evap / 1e6

        k_river = P_toriver / (P_thermal - P_el_gross)
        k_evap = P_vap / P_tower

        if printBalance:
            print('Water balance :')
            print(f'Withdrawal flow : {q_evap + q_blowdown} m3/s')
            print(f'Evaporated flow : {q_evap} m3/s')
            print(f'Blowdown flow : {q_blowdown} m3/s')
            print(f'Discharge water temperature : {T_basin} °C')
            print(f'Downstream temperature : {T_downstream} °C')
            print('\n Energy balance :')
            print(f'Primary heat : {P_thermal} MW')
            print(f'Electricity output : {P_el_gross} MW')
            print(f'Cooling load : {P_thermal-P_el_gross} MW')
            print(f'Heat discharged to river : {P_toriver} MW')
            print(f'Heat discharged to tower : {P_tower} MW')
            print(f'Heat through evaporation : {P_vap} MW')
            print(f'Sensible transfer to tower air : {P_tower-P_vap} MW')
            print('\n Tower coefficients : ')
            print(f'Fraction of heating load discharged to river k_river : {k_river}')
            print(f'Fraction of heating load discharged to tower k_tower: {1 - k_river}')
            print(f'Fraction of evaporation to tower load k_evap : {P_vap/P_tower}')

        return k_river,k_evap
