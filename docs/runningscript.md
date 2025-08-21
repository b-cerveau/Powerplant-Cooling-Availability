!!! note
    The simulation can be executed entirely from `Driver.py`.  
    It **must** be run as a module from the top of the `Simulation` directory. For example:

    ```bash
    cd /path/to/.../Simulation
    python -m Driver.py
    ```

## Before running your first simulation:

1. Make sure you fill in all required file paths to the input data (paths to the raw data, and paths to where the processed timseries will be stored). These are located in the headers of:  

    - `CoolingTower.py`
    - `Loire.py` 
    - `Rhine.py`
    - `Rheinkilometer.py` 
    - `Elbe.py` (if implemented)
    - `StationFinding.py`
  
2. Run the `preprocessing()` function at least once to clean the raw input time series. This step is not needed afterwards, unless new data or new plants have been added.

## To run a simulation:

- Select the desired configurations and regions in the file header.
- Call:
  
    ```python
    runCompleteSimulation(CONFIGURATIONS)
    ```

!!! tip
    Make sure to wrap calls in the  

    ```python
    if __name__ == "__main__":
    ```
    clause to avoid multiprocessing issues.

Postprocessing is included in the main `runCompleteSimulation()` method, but can be reran separately once all timseries exist using `fullPostprocessing()`.


Other driver functions are also available to recreate the Loire plants sensitivity analysis graphs.
