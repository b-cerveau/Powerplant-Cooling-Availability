To run the script for a new set of plants and scenario/model combinations:

1. Create a new **`region`** string identifier that will represent your group of plants.

2. Provide methods to build:

    - A list of `PowerPlant` objects.
    - Corresponding **Q, Tw, Tair, RH** files for each plant.

3. Link them to `Driver.py`.

### Recommended Implementation

1. Write a **region-specific file** that contains methods to:

    - Initialise the region’s plant list.
    - Create environmental time series for the plants.

    To build these time series, it’s usually easier to have a **relevant identifier attribute** for each plant object (e.g., a gridpoint number).  
    This can be set in the `PowerPlant.__init__(self)` function and coupled to the `StationFinding` module.

    !!! note
        Be careful about **circular imports**:  
        The part from your regional file that provides a plant initialisation method will import the`PowerPlant` module — so the `PowerPlant` module can't use any methods from your regional file. 
        This is why `Rheinkilometer` and `Elbekilometer` are separate files, as matching plant coordinates to their river km is useful to initialise the plants.

2. Link Your Methods to `Driver.py` :

    Update the main variables and functions:

    - **`REGIONS`** — Set of all regions to simulate.
    - **`VALID_COMBINATIONS`** — Dictionary mapping each region name to valid `(scenario, model)` pairs.
    - **Functions:** 
        - `preprocessing()` 
        - `initialisePlants()` : returns the created list of `PowerPlant` objects
        - `buildTimeSeries()` : creates an environmental timeseries file for a `PowerPlant` object.
        Expects output CSV to have columns `Gregorian_day`, `Q`, `Tw`, `Tair`, `RH` to follow the right naming convention (_see docstrings_). Also expected to accomodate for parameters `Q_offset`, `Tw_offset`, `Tair_offset`, `RH_offset`

### Examples
Check how `Loire.py` and `Rhine.py` are coupled to the main script for reference.