# Script Description

This script relies on object-oriented programming to simulate power availability : details are shown below, with examples at the end.

The script is modularized so that the simulation can be run with heterogeneous climate input data. The core identifier for this purpose is the `region` attribute, which describes a group of plants and their corresponding input dataset. 

## `PowerPlant` class

Power plants are represented as instances of a custom `PowerPlant` class.

An instance of this class is characterized by the following attributes (which roughly match the nomenclature of the JRC plant dataset):

### Required Attributes:
- `name: str` — Name of the unit
- `region: str` — Region (to link the plant to its input dataset)
- `lon: float` — Longitude (decimal degrees)
- `lat: float` — Latitude (decimal degrees)
- `type: str` — Plant type. Supported types are: `Nuclear`, `Gas simple cycle`, `Gas combined cycle`, `Coal`
- `water_type: str` — Water type. Options: `Freshwater` or `Seawater`
- `cooling_type: str` — Cooling technology. Options: `Air cooling`, `Cooling tower`, `Once-through`

!!! note
      For cooling technology as well as plant type, flexible input matching to the canonical type is implemented (helpful for loading from CSV for instance): see the relevant `ALIASES` dictionaries at the top of the class. Also see specificities about gas-fired plants for technology matching (simple vs combined-cycle).

- `capacity_net: float = None` — Net electrical capacity (MW)
- `capacity_gross: float = None` — Gross electrical capacity (MW)
- `efficiency_net: float = None` — Net efficiency (between 0 and 1)
- `efficiency_gross: float = None` — Gross efficiency (between 0 and 1)

!!! note
      For efficiencies and capacities, at least one capacity and one efficiency value must be provided. The rest will be automatically filled, assuming gross and net capacities coincide if no other information is provided.

### Optional Attributes:
- `P_min: float = None` — Minimal production threshold (in net MW output). Default : 20% of net capacity.
- `environmental_reg: EnvironmentalReg = None` — See class description later: environmental regulation for the plant. 

    Default: `EnvironmentalReg(DeltaT_river_max=3, T_downstream_max=28, T_outlet_max=30, gamma=0.75)`

- `ncc: float = None` — Number of cycles of concentration for cooling tower plants. Default behavior: `1.5` for Nuclear plants, `3` otherwise
- `k_tower: float = None` — When using the static cooling tower model: share of cooling load discharged via tower. Default : 0.98
- `eic_p: str = None` — EIC production unit identifier. Optional, may be useful to link units to other databases
- `eic_g: str = None` — EIC generation unit identifier. Optional, may be useful to link units to other databases
- `country: str = None` — Optional, for plant description

### Attributes to help link Plants to their input dataset:
- `TNET_reach: int = None` — ID for the corresponding reach in the TNET dataset (Loire)
- `river_km: float = None` — Kilometer on the Rhine/Elbe river


## `EnvironmentalReg` class

The `EnvironmentalReg` class encodes the set of environmental constraints for each plant in a single object, accounting for potentially exotic rules (e.g., variable thresholds based on upstream river flow, DOY, etc.).

### Standard constraints:

A standard regulation set can be defined using the following float values:

- `T_outlet_max`: Max outlet temperature (°C). Default: `np.inf`
- `T_downstream_max`: Max downstream mixed temperature (°C). Default: `np.inf`
- `DeltaT_max`: Max temperature difference (mixed downstream - upstream, °C). Default: `np.inf`
- `Q_wd_max`: Max withdrawal flow rate (m³/s). Default: `np.inf`
- `Q_evap_max`: Max water consumption rate (m³/s). Default: `np.inf`  
  _Note: No difference made between 'evaporation' and 'consumption' in general._
- `gamma`: Max fraction of upstream flow allowed to be withdrawn. Defaults:

    - `1.0` if another flow constraint is given
    - `0.75` otherwise

!!! example 
    The standard EU ruleset about temperatures, with no restriction on streamflow, can be modelled by :

    ```python
    EnvironmentalReg(T_outlet_max = 30, T_downstream_max = 28, DeltaT_max = 3, gamma = 1)

    ```

    The `gamma = 1` overrides the default mechanism about streamflow regulations. Note that the default value of 0.75 (minimum 25% downflow) matches usual minimal recommendations for aquatic life.

To build a custom ruleset instead, pass a regulation function to `self.reg_function` when initialising.

### Custom Regulation Function

A regulation function acts as a checker function, assessing if an operating state is allowed or not. The following arguments will be passed to it as keyword arguments, and it is assumed that they define the full plant operating state as far as environmental regulations are concerned.

- `Q`: upstream flow rate (m³/s)
- `Tw`: upstream water temperature (°C)
- `Q_wd`: withdrawal flow (m³/s)
- `Q_evap`: consumed water (m³/s)
- `T_outlet`: outlet water temperature (°C)
- `day` : a string in the `YYYY-MM-DD` format

Returns `True` if the operating conditions are compliant, `False` otherwise.

!!! note
    Passing a regulation function overrides the previous thresholds. However, if some thresholds are still fixed (e.g. streamflow side of regulations is standard but temperature is not), passing them during initialisation can accelerate further simulation (see `PowerPlant.checkOnceThroughCompatibility` for further details).


!!! note
    It is assumed that parameters above define the full plant operating state. To add new parameters, you may need to modify the relevant methods and callables, mainly:

    - `EnvironmentalReg.is_compliant()`
    - `PowerPlant.check_environmental_compliance()`
    - `Driver.buildTimeSeries()` and `PowerPlant.powerAvailability()`, ensuring the new argument is correctly passed.

!!! tip
    As all the arguments mentioned above will be passed to your regulation function even if they do not matter for your operating rule, it is good practice to add a `**kwargs` at the end of the signature, for more robustness


## First examples

### Example n°1 : manually initialising a regular coal-fired plant

```python
walsum = PowerPlant(name = 'WALSUM_9', region = 'Rhine', type = 'Coal', capacity_net = 370, efficiency_net = 0.3, water_type = 'Freshwater', cooling_type = 'Cooling tower', lat = 51.526, lon = 6.72, country = 'Germany' )
```
In this case, the default environmental regulation `EnvironmentalReg(DeltaT_river_max=3, T_downstream_max=28, T_outlet_max=30, gamma=0.75)` will be applied, as well as all other default parameters mentioned above.

### Example n°2 : manually initialising a custom nuclear plant

```python

def Civaux_environmental_checker(Q=np.inf,Tw=-np.inf, T_outlet=-np.inf, Q_wd =0, Q_evap=0, **kwargs):
    
    if Q_wd > 6:
        return False
    if Q_evap > 1.7:
        return False
    if Q-Q_evap < 10 :
        return False
    
    T_downstream = EnvironmentalReg.downstream_temp(Q, Tw, Q_wd, Q_evap, T_outlet)
    DeltaT = T_downstream - Tw

    if Tw < 25:
        if DeltaT > 2 or T_downstream >25:
            return False
    elif DeltaT > 1e-4 :
        return False
    return True

CIVAUX = PowerPlant(name='CIVAUX', region = 'Loire', type= 'Nuclear',
                    capacity_net= 1495*2, capacity_gross=1560*2,  efficiency_net = 0.35, 
                    lat = 46.456, lon= 0.655, country = 'France',
                    water_type= 'Freshwater', cooling_type='Cooling tower',
                    TNET_reach= 40455, 
                    environmental_reg=EnvironmentalReg(regulation_function=Civaux_environmental_checker_notemp), 
                    ncc =1.8)
```

### Example n°3 : Batch-loading a group of plants

Use the relevant `PowerPlant.loadFromCSV()` function. A practical example on how to batch load a group of plants, including ones with custom regulations, is given by the `Loire.initialiseLoirePlants()` function : 

```python
PLANTS = 'Inputs/Loire/Plants/LoirePlants.csv'

def initialiseLoirePlants():
    custom_regulations = {
        'BELLEVILLE' : EnvironmentalReg.loire_regulation(Q_wd_max=10.5),
        'CHINON' : EnvironmentalReg.loire_regulation(Q_wd_max=8.6),
        'CIVAUX' : EnvironmentalReg(regulation_function=Civaux_environmental_checker_notemp),
        'CORDEMAIS' : EnvironmentalReg(DeltaT_river_max=3, T_outlet_max=40,T_downstream_max=28,Q_wd_max=63),
        'DAMPIERRE' : EnvironmentalReg.loire_regulation(Q_wd_max=12.3),
        'ST_LAURENT' : EnvironmentalReg.loire_regulation(Q_wd_max=7)
    }

    plant_list = PowerPlant.loadFromCSV(PLANTS, sep=';')

    for plant in plant_list:
        for key in custom_regulations:
            if plant.name.upper().startswith(key.upper()):
                plant.environmental_reg = custom_regulations[key]
                break

    print('Loire plants initialised.')
    return plant_list

```
