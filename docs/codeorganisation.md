## Script Structure

### Core Simulation Logic
- `constants.py` — Physical constants used across the code.
- `CoolingTower.py` — Performance models for cooling tower and condenser, extracted from the Guenand et al. data.
- `PowerPlantSimulation.py` — Modelling of individual power plants and their max availability under given environmental conditions.
- `ClusterSimulation.py` — Handles clustering logic to simulate plant interaction.

### General Utility Files
- `general_utils.py` — Useful helper functions to work with datasets. Independent from the rest of the code.
- `StationFinding.py` — Tool used to link power plants to their attributed input environmental data  
  *(Example: find closest meteorological gridpoint to match with atmospheric data)*.  
  Linked to region-specific folders, used in PowerPlant object initialisation.
- `PostProcessing.py` — Analysis of results: calculates statistical metrics and generates relevant plots.

### Regional Utility Files
- `Rheinkilometer.py` — Tool to match geographical coordinates to a river km.
- `Elbekilometer.py` — Tool to match geographical coordinates to a river km  
  *(however, Elbe not further implemented in the code yet)*.
- `Loire.py` — Preprocessing for the Loire environmental time series, plant initialisation.
- `Rhine.py` — Preprocessing for the Rhine environmental time series, plant initialisation.

### Master File
- `Driver.py` — The overarching script that links all these modules together.

---

## Nomenclature
- **Model** — Identifier for a driving climate model (GCM-RCM pair).
- **Scenario** — Identifier for the corresponding emission scenario (`rcp26`, `rcp45`, `rcp85`).
- **Region** — Identifier for plant region (`Loire`, `Rhine`, `Elbe`).
- **Configuration** — Identifier for model parameters, used to distinguish the normal model from altered time series  *(for sensitivity analysis purposes)*.  
  A _configuration_ is defined as a tuple: 
    ```python
    (configname, dynamicTower, Q_offset, Tw_offset, Tair_offset, RH_offset)
    ```

    Where:

    - `configname` *(str)* — Configuration name, where the time series will be saved.
    - `dynamicTower` *(bool)* — Whether or not to use the dynamic cooling tower model.
    - `Q_offset` *(float)* — Multiplicative offset for streamflow.
    - `Tw_offset` *(float)* — Additive offset for water temperature (°C).
    - `Tair_offset` *(float)* — Additive offset for air temperature (°C).
    - `RH_offset` *(float)* — Additive offset for relative humidity (%).

---

## Results directory structure
```bash
Outputs/configname/region/scenario/model/Timeseries/...
```

In the code, different levels of this tree are referred to as:

- `Homedir` → `Outputs/`
- `configdir` → `Outputs/configname`
- `regiondir` → `Outputs/configname/region`