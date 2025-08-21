import os
import multiprocessing as mp
import numpy as np
import traceback

from datetime import datetime, timedelta
from Code.ClusterSimulation import PlantCluster

from Code import Loire, Rhine, Rheinkilometer
from Code import PostProcessing

################ SIMULATION CONFIGURATION #######################################

HOMEDIR = 'Outputs'
REGIONS = {'Loire'}
VALID_COMBINATIONS = {
    'Loire': Loire.VALID_COMBINATIONS
}

# Configuration, listed as : 
# config_name, dynamicTower, Q_regulation, T_regulation, Q_offset, Tw_offset, Tair_offset, RH_offset

# config_name must be a valid folder name

CONFIGURATIONS ={
    ('base', True, True, True, 1, 0, 0, 0),
    ('staticTower', False, True, True, 1, 0, 0, 0),
    ('notemp', True, True, False, 1, 0, 0, 0),
    # ('eff_only', True, False, False, 1, 0, 0, 0)

    ('Q_90', True, True, True, 0.9, 0, 0, 0),
    ('Q_925', True,True,  True, 0.925, 0,0,0),
    ('Q_95', True, True, True, 0.95, 0, 0, 0),
    ('Q_975', True,True,  True, 0.975,0,0,0),
    ('Q_1025', True,True,  True, 1.025, 0,0,0),
    ('Q_105', True, True, True, 1.05, 0, 0, 0),
    ('Q_1075', True,True,  True, 1.075, 0,0,0),
    ('Q_110', True, True, True, 1.1, 0, 0, 0),

    ('Tw_-2', True, True, True, 1, -2, 0, 0),
    ('Tw_-15', True, True, True, 1, -1.5,0,0),
    ('Tw_-1', True, True, True, 1, -1, 0, 0),
    ('Tw_-05', True,True,  True, 1, -0.5, 0,0),
    ('Tw_05', True, True, True, 1, 0.5, 0,0),
    ('Tw_1', True, True, True, 1, 1, 0, 0),
    ('Tw_15', True,True,  True, 1, 1.5,0,0),
    ('Tw_2', True, True, True, 1, 2, 0, 0),

    ('Tair_-2', True, True, True, 1, 0, -2, 0),
    ('Tair_-15', True,True,  True, 1, 0, -1.5,0),
    ('Tair_-1', True, True, True, 1, 0, -1, 0),
    ('Tair_-05', True, True, True, 1, 0, -0.5,0),
    ('Tair_05', True, True, True, 1, 0, 0.5,0),
    ('Tair_1', True, True, True, 1, 0, 1, 0),
    ('Tair_15', True, True, True, 1, 0, 1.5,0),
    ('Tair_2', True, True, True, 1, 0, 2, 0),

    ('RH_-10', True,True,  True, 1, 0, 0, -10),
    ('RH_-75', True, True, True, 1, 0, 0, -7.5),
    ('RH_-5', True, True, True, 1, 0, 0, -5),
    ('RH_-25', True,True,  True, 1, 0, 0, -2.5),
    ('RH_25', True, True, True, 1, 0, 0, 2.5),
    ('RH_5', True, True, True, 1, 0, 0, 5),
    ('RH_75', True, True, True, 1, 0, 0, 7.5),
    ('RH_10', True, True, True, 1, 0, 0, 10)
}

############ INITIALISATION STEPS ###############
def preprocessing():
    '''All preprocessing utilities for dataset cleaning etc.'''

    ## Loire region
    Loire.prepareLoireAtmosphereTimeseries()
    loire_plants = Loire.initialiseLoirePlants()
    for scenario, model in Loire.VALID_COMBINATIONS:
        Loire.getLoireHydroPath(model, scenario, plant_list = loire_plants, runExtraction=True)

    ## Germany
    Rheinkilometer.cleanRheinkilometer()
    Rhine.cleanGermanyQTimeseries()
    Rhine.cleanGermanyTwTimeseries()

def initialisePlants(region, T_regulation = True, Q_regulation = True):
    ''' Returns the plant list from the input region.'''

    print(f'Initialising plants for region : {region}')
    match region:
        case 'Loire':
            return Loire.initialiseLoirePlants(T_regulation= T_regulation, Q_regulation = Q_regulation)

        case 'Rhine':
            return Rhine.initialiseRhinePlants()
        
        case _: 
            raise ValueError(f'Unsupported region for plant initialisation : {region}')

############# SIMULATION ################
def buildTimeSeries(region, plant_list, configdir : str,
                    scenario : str, model : str, 
                    Q_offset : float = 0, Tw_offset : float = 0, Tair_offset : float = 0, RH_offset : float = 0):
    '''
    Prepares the folder structure and the plant-specific evironmental CSVs, so that they are stored under :
    configdir/region/scenario/model/Timeseries/plantname_scenario_model_timeseries.csv
    '''
    print(f'Extracting plant specific timeseries for region {region}')

    # Prepare folder structure
    region_dir = os.path.join(configdir,region)
    scenario_dir = os.path.join(region_dir,f'{scenario}')
    model_dir = os.path.join(scenario_dir, f'{model}')
    timeseries_dir = os.path.join(model_dir,'Timeseries')

    os.makedirs(region_dir, exist_ok=True)
    os.makedirs(scenario_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(timeseries_dir, exist_ok=True)
    
    # Region-specific timeseries extraction
    match region:
        case 'Loire':
            Loire.buildPlantTimeSeries(plant_list, timeseries_dir, scenario, model, 
                                 extractHydro=False, extractAtm=False,
                                 Q_offset = Q_offset, Tw_offset=Tw_offset, Tair_offset=Tair_offset, RH_offset=RH_offset)

        case 'Rhine':
            Rhine.buildRhineTimeSeries(plant_list, timeseries_dir, scenario, model,
                                       Q_offset = Q_offset, Tw_offset=Tw_offset, Tair_offset=Tair_offset, RH_offset=RH_offset)

        case _ :
            print(f'Unsupported region for plant environmental timeseries extraction : {region}')

def runSimulation(region, configdir : str,
                  scenario : str, model : str,
                  Q_regulation : bool = True, T_regulation : bool = True,
                  dynamicTower : bool = True, showProgress : bool = True,
                  Q_offset : float = 0, Tw_offset : float = 0, Tair_offset : float = 0, RH_offset : float = 0):
   
    # Initialise plants and build timeseries files
    plants = initialisePlants(region, T_regulation= T_regulation, Q_regulation= Q_regulation)
    buildTimeSeries(region, plants, 
                    configdir, scenario, model,
                    Q_offset= Q_offset, Tw_offset = Tw_offset, Tair_offset = Tair_offset, RH_offset = RH_offset)
    print(f'Timeseries built for region {region}, scenario {scenario} and model {model}')

    # Simulate plant production
    clusters = PlantCluster.clustersFromList(plants)
    for cluster in clusters:
        cluster.simulateClusterProduction(configdir, scenario, model, show_progress=showProgress, dynamicTower=dynamicTower)
    
def runSimulation_wrapper(args):
    '''
    Wrapper of runSimulation used for multiprocessing. Unpacks the arguments from a tuple and sets `showProgress` to `False`.
    Expects args in order : 
    region, configdir, scenario, model, extractHydro, extractAtm, dynamicTower, Q_offset, Tw_offset, Tair_offset, RH_offset
    '''
    region, configdir, scenario, model, dynamicTower, Q_regulation, T_regulation, Q_offset, Tw_offset, Tair_offset, RH_offset = args

    try:
        runSimulation(region, configdir, scenario, model,
                      Q_regulation = Q_regulation, T_regulation=T_regulation,
                      dynamicTower=dynamicTower, showProgress=False,
                      Q_offset= Q_offset, Tw_offset = Tw_offset, Tair_offset = Tair_offset, RH_offset = RH_offset)  
           
    except KeyError as e:
        print(f"KeyError encountered: {e} in region {region}, scenario {scenario}, model {model}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e} in region {region}, scenario {scenario}, model {model}")
        raise

############ POSTPROCESSING #####################
def individualPostProcessing(plants, regiondir, scenario, model, drawPlots = True):
    '''
    Runs the post processing steps for an individual configuration/region/model/scenario combination.
    Plots power output distribution graphs (if `drawPlots` is `True`), writes a recap timeseries file, computes CVar and its evolution across timeframes.

    Expects files under regiondir/scenario/model/Timeseries/plantname_scenario_model_timeseries.csv
    If applicable, plots saved to regiondir/scenario/model/Plots/plantname_scenario_model_capacity_days.png
    '''
    if drawPlots:
        PostProcessing.plotDaysByCapacity(plants, regiondir, model, scenario)
    
    PostProcessing.writeGeneralTimeseriesFile(plants, regiondir, scenario, model)
    PostProcessing.calculateCVar(plants,scenario,model,regiondir,p=0.05)
    PostProcessing.computeCVarEvolution(regiondir, scenario, model, plants,
                                        startdate=datetime(2010, 1, 1), timedelta=timedelta(days=365 * 45), p=0.05)
    
def regionPostprocessing(configuration, region, drawPlots = True):
    ''' Postprocessing for a given configuration/region : within the region directory, runs all the individual postprocessing steps and region-wide steps as well.'''
    plants = initialisePlants(region)
    print(f'Post processing for configuration : {configuration[0]}')
    print(f'region : {region}')
    regiondir = os.path.join(HOMEDIR, configuration[0],region)
    for scenario, model in VALID_COMBINATIONS[region]:
        individualPostProcessing(plants, regiondir, scenario, model, drawPlots=drawPlots)

    CVar_path = os.path.join(regiondir, 'pCVar.png')
    PostProcessing.plotSystemCVarTrajectories(regiondir, region, save_path=CVar_path)
    PostProcessing.buildRegionPCVarRecapCSV(regiondir, VALID_COMBINATIONS[region])

def regionPostprocessing_wrapper(args):
    ''' Wrapper for multiprocessing with safe error capture '''
    configuration, region, drawPlots = args
    try:
        regionPostprocessing(configuration, region, drawPlots=drawPlots)
        return {"status": "ok", "configuration": configuration, "region": region}
    except Exception as e:
        return {
            "status": "error",
            "configuration": configuration,
            "region": region,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    
def overallPostProcessing():
    ''' Post processing that compares configurations : saves general CVar/DR file, plots DR dispersion graph across all runs, and the final sensitivity analysis graph.'''
    PostProcessing.buildOverallPCVarRecapCSV(HOMEDIR, CONFIGURATIONS, REGIONS, VALID_COMBINATIONS)
    PostProcessing.plotDiversificationRatios(HOMEDIR, REGIONS)
    PostProcessing.plotGeneralSensitivity(HOMEDIR, REGIONS, VALID_COMBINATIONS)

def fullPostprocessing(configurations, drawPlots = True):
    '''
    Master function that performs full simulation postprocessing on individual, regional and overall scopes.
    '''
    jobs = []
    for configuration in configurations:
        for region in REGIONS:
            jobs.append((configuration, region, drawPlots))

    print('Starting multithreaded postprocessing.')
    with mp.Pool(processes=max(mp.cpu_count() - 1, 1)) as pool:
        results = pool.map(regionPostprocessing_wrapper, jobs)
    
    for res in results:
        if res["status"] == "error":
            print("\n⚠️ ERROR in worker:")
            print(f"  Configuration: {res['configuration']}, Region: {res['region']}")
            print(f"  Error: {res['error']}")
            print("  Traceback:")
            print(res["traceback"])
        else:
            print(f"✔ Completed {res['configuration']} - {res['region']}")

    overallPostProcessing()

########### FULL SIMULATION ROUTINE ############
def runCompleteSimulation(configurations):
    '''
    A configuration is a tuple containing all relevant parameters, in the form:
    config_name, dynamicTower, Q_offset, Tw_offset, Tair_offset, RH_offset

    Preprocessing needs to have been ran beforehand.
    '''

    jobs = []

    for i,configuration in enumerate(configurations):

        config_name, dynamicTower, Q_regulation, T_regulation, Q_offset, Tw_offset, Tair_offset, RH_offset = configuration

        configdir = os.path.join(HOMEDIR, config_name)
        os.makedirs(configdir, exist_ok=True)

        for region in REGIONS:
            
            for scenario, model in VALID_COMBINATIONS[region]:

                    jobs.append((
                        region, 
                        configdir, 
                        scenario, 
                        model, 
                        dynamicTower, 
                        Q_regulation, T_regulation,
                        Q_offset, Tw_offset, Tair_offset, RH_offset
                    ))

    with mp.Pool(processes=max(mp.cpu_count() - 1, 1)) as pool:
        pool.map(runSimulation_wrapper, jobs)

    fullPostprocessing(CONFIGURATIONS, drawPlots= True)
################################################

##### RECREATE REPORT PLOTS ######
def nccSensitivityLoire_worker(args):
    ''' Worker for multiprocessing in the nccSensitivity function below'''
    plant_name, ncc, configdir, scenario, model, inputpath = args

    # Rebuild plants fresh in each worker
    plants = Loire.initialiseLoirePlants()
    units = [plant for plant in plants if plant.name.upper().startswith(plant_name)]
    for p in units:
            p.ncc = ncc
            p.efficiency_net = 0.35 if plant_name == 'CIVAUX' else p.efficiency_net
            p.efficiency_gross = p.efficiency_net * p.capacity_gross/p.capacity_net

    PlantCluster.clusterAndSimulate(units, configdir, scenario, model, inputpath=inputpath, showProgress=False)
    PostProcessing.writeGeneralTimeseriesFile(units, configdir, scenario, model, input_dir=inputpath)
    PostProcessing.calculateCVar(units, scenario, model, configdir, 
                                 input_path= os.path.join(inputpath, f'general_{scenario}_{model}_timeseries.csv') , 
                                 output_dir=inputpath)

def efficiencySensitivityLoire_worker(args):
    ''' Worker for multiprocessing in the efficiencySensitivity function below'''
    plant_name, eff_net, configdir, scenario, model, inputpath = args

    # Rebuild plants fresh in each worker
    plants = Loire.initialiseLoirePlants()
    units = [plant for plant in plants if plant.name.upper().startswith(plant_name)]
    for plant in units:
        plant.efficiency_net = eff_net
        plant.efficiency_gross = plant.efficiency_net * plant.capacity_gross / plant.capacity_net

    PlantCluster.clusterAndSimulate(
        units,
        configdir,
        scenario,
        model,
        inputpath=inputpath,
        showProgress=False
    )

    PlantCluster.clusterAndSimulate(units, configdir, scenario, model, inputpath=inputpath, showProgress=False)
    PostProcessing.writeGeneralTimeseriesFile(units, configdir, scenario, model, input_dir=inputpath)
    PostProcessing.calculateCVar(units, scenario, model, configdir, 
                                 input_path= os.path.join(inputpath, f'general_{scenario}_{model}_timeseries.csv') , 
                                 output_dir=inputpath)

def nccSensitivityLoire(plant_name : str = 'CHINON', runSimulation = True):
    '''
    Testing module to run sensitivity analyses on the Loire plants. Test sensitivity relative to n_cc water consumption factor.
    For plant name, input the name without its subunit number : e.g. "CHINON" and not "CHINON 1".
    For plant name CIVAUX, will use different ncc and efficiency values (see report).
    '''
    scenario = 'rcp26'
    model = 'CNRM-CM5-LR_ALADIN63'

    plant_name = plant_name.strip().upper()

    ncc_list = np.arange(1.2, 1.8, 0.1) if plant_name != 'CIVAUX' else [1.2, 1.5, 1.8, 2, 5, 10, 100]

    config_path = os.path.join(HOMEDIR, f'ncc_sensitivity_{plant_name}')
    os.makedirs(config_path, exist_ok = True)

    plants = Loire.initialiseLoirePlants()
    units = [plant for plant in plants if plant.name.upper().startswith(plant_name)]
    
    if runSimulation:
        # Prepare simulation jobs for each ncc value
        jobs = []
        for ncc in ncc_list:
            ncc = round(ncc, 2)
            timeseries_dir = os.path.join(config_path, f'ncc_{int(10*ncc)}')
            os.makedirs(timeseries_dir, exist_ok=True)

            plants = Loire.initialiseLoirePlants()
            units = [p for p in plants if p.name.upper().startswith(plant_name)]

            Loire.buildPlantTimeSeries(units, timeseries_dir, scenario, model, extractHydro= False)
            jobs.append((plant_name, ncc, None, scenario, model, timeseries_dir))

        with mp.Pool(processes=max(mp.cpu_count() - 1, 1)) as pool:
            pool.map(nccSensitivityLoire_worker, jobs)

    refplant = [unit for unit in units if unit.name.endswith('1')][0]
    PostProcessing.plotNccSensitivity(refplant, config_path, scenario, model, ncc_list)

def efficiencySensitivityLoire(plant_name : str ='CIVAUX', runSimulation = True):
    """
    Sensitivity analysis for efficiency_net parameter for CIVAUX plants.
    Changes both efficiency_net and efficiency_gross accordingly.
    For plant name, input the name without its subunit number : e.g. "CHINON" and not "CHINON 1".
    """
    scenario = 'rcp26'
    model = 'CNRM-CM5-LR_ALADIN63'
    eff_list = [0.35, 0.36, 0.37, 0.38, 0.40, 0.45] if plant_name != 'CORDEMAIS' else [0.44, 0.45, 0.46, 0.47, 0.48, 0.49]

    plant_name = plant_name.strip().upper()

    plants = Loire.initialiseLoirePlants()
    units = [p for p in plants if p.name.upper().startswith(plant_name)]
    
    config_path = os.path.join(HOMEDIR, f'efficiency_sensitivity_{plant_name}')
    os.makedirs(config_path, exist_ok=True)

    if runSimulation:
        # Prepare simulation jobs for each efficiency value
        jobs = []
        for eff in eff_list:
            eff = round(eff, 3)  # 3 decimal places just in case
            timeseries_dir = os.path.join(config_path, f'eff_{int(eff*100)}')
            os.makedirs(timeseries_dir, exist_ok=True)

            # Build time series files for this efficiency setting
            Loire.buildPlantTimeSeries(units, timeseries_dir, scenario, model, extractHydro=False)

            jobs.append((plant_name, eff, None, scenario, model, timeseries_dir))

        # Run simulations in parallel
        with mp.Pool(processes=max(mp.cpu_count() - 1, 1)) as pool:
            pool.map(efficiencySensitivityLoire_worker, jobs)

    # Pick a reference unit (first subunit) for plotting
    plants = Loire.initialiseLoirePlants()
    refplant = [p for p in plants if p.name.upper().startswith(plant_name) and p.name.endswith('4')][0]

    PostProcessing.plotEfficiencySensitivity(refplant, config_path, scenario, model, eff_list)

if __name__ == '__main__':
    mp.freeze_support()

    # runCompleteSimulation(CONFIGURATIONS)
    # fullPostprocessing(CONFIGURATIONS, drawPlots=False)
    # overallPostProcessing()
    # efficiencySensitivityLoire(plant_name='CORDEMAIS', runSimulation=False)

plants = Loire.initialiseLoirePlants()
plants = [unit for unit in plants if unit.name in ['CIVAUX_1', 'DAMPIERRE_1', 'FR-GA-MORANT1', 'CORDEMAIS_4']]
PostProcessing.plotGroupedCapacityDays(plants, 'Outputs/base/Loire', 'rcp26', 'CNRM-CM5-LR_ALADIN63')




