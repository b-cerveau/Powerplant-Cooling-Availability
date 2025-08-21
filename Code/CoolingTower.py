import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator

from metpy.calc import wet_bulb_temperature, dewpoint_from_relative_humidity
from metpy.units import units

from Code.constants import rho_w,c_p,L_vap

################################# STATIC COOLING TOWER MODEL
_INTERPOLATOR = None
INTERPOLATOR_BUILT = False               # To build interpolator only if needed

def T_wetbulb(Tair, RH, p=1013.25):
    """
    Calculate wet-bulb temperature using MetPy.
    
    Parameters
    ----------
    Tair : float
        Dry bulb temperature in °C
    RH : float
        Relative humidity in %
    p : float, optional
        Atmospheric pressure in hPa (default is 1013.25 hPa)
    
    Returns
    -------
    float
        Wet-bulb temperature in °C
    """
    
    T = Tair * units.degC
    rh = np.clip(RH / 100.0, max= 1)
    pressure = p * units.hPa
    
    Td = dewpoint_from_relative_humidity(T, rh)
    Tw = wet_bulb_temperature(pressure, T, Td)
    
    return Tw.to('degC').m

def build_Twb_interpolator(p=1013.25):
    """
    Precompute wet bulb temperatures on a grid of (Tair, RH)
    and return a fast interpolator.
    """
    Tair_grid = np.arange(-10, 40.1, 0.1)   # °C
    RH_grid   = np.array([10, 20, 50, 80, 100])  # %

    Twb_grid = np.zeros((len(Tair_grid), len(RH_grid)))
    for i, T in enumerate(Tair_grid):
        for j, rh in enumerate(RH_grid):
            Twb_grid[i, j] = T_wetbulb(T, rh, p)

    interp = RegularGridInterpolator(
        (Tair_grid, RH_grid),
        Twb_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def fast_Twb(Tair, RH):
        pts = np.array([Tair, RH]).T
        return float(interp(pts)) if np.isscalar(Tair) else interp(pts)

    return fast_Twb

def fast_Twetbulb(Tair, RH, p=1013.25):
    """
    Lazy wrapper: build the interpolator on first call,
    then replace this function with the fast one.
    """
    global _INTERPOLATOR, INTERPOLATOR_BUILT, fast_Twetbulb
    if not INTERPOLATOR_BUILT:
        print("CoolingTower: Building wet bulb temperature interpolator...")
        _INTERPOLATOR = build_Twb_interpolator(p)
        fast_Twetbulb = _INTERPOLATOR  # overwrite with fast version (2 args only)
        INTERPOLATOR_BUILT = True
        print("CoolingTower: Interpolator built.")
    return _INTERPOLATOR(Tair, RH)

################################# DYNAMIC COOLING TOWER MODEL
print('CoolingTower : Building Golfech data interpolator')

# Load Golfech performance data
evap_20 = pd.read_csv("Inputs/Cooling tower/evap_flow_RH20.csv")
evap_100 = pd.read_csv('Inputs/Cooling tower/evap_flow_RH100.csv')
disch_20 = pd.read_csv("Inputs/Cooling tower/discharge_temp_RH20.csv")
disch_100 = pd.read_csv('Inputs/Cooling tower/discharge_temp_RH100.csv')
efficiency_decrease = pd.read_csv('Inputs/Cooling tower/efficiency_decrease.csv',sep=';')


# Create interpolation functions (with linear extrapolation)
evap_interp_20 = interp1d(evap_20["dry_temp"], evap_20["evap_flow"], kind='linear', fill_value='extrapolate')
evap_interp_100 = interp1d(evap_100["dry_temp"], evap_100["evap_flow"], kind='linear', fill_value='extrapolate')

disch_interp_20 = interp1d(disch_20["dry_temp"], disch_20["discharge_temp"], kind='linear', fill_value='extrapolate')
disch_interp_100 = interp1d(disch_100["dry_temp"], disch_100["discharge_temp"], kind='linear', fill_value='extrapolate')

efficiency_interp = interp1d(efficiency_decrease['T_basin'], efficiency_decrease['Efficiency_extrapolated'], fill_value='extrapolate')

def evap_flow(dry_temp: float, RH : float, P_cooling : float):
    """
    Bilinear interpolation of evaporated flowrate in the cooling tower based on air dry-bulb temperature, relative humidity and cooling load. 
    Uses the charts provided in Guenand et al. , assuming the cooling tower balance is linear based on cooling load and the 
    reference cooling load for the graph is : 3800 - 1330 = 2470 MW
    
    Parameters:

    - `dry_temp` (`float`) : dry-bulb air temperature in °C
    - `RH` (`float`) : relative humidity in percent (0–100)
    - `P_cooling` (`float`) : cooling load (MW)

    Returns:
    - `evap_flow` (`float`) : evaporated water flow in the cooling tower (m3/s)
    """
    P_cooling_ref = 2470

    RH = np.clip(RH, 0, 100)

    flow_20 = evap_interp_20(dry_temp)
    flow_100 = evap_interp_100(dry_temp)

    # Linear interpolation based on RH
    # RH20% corresponds to flow_20, RH100% to flow_100
    if RH >= 100:
        flow = flow_100
    else:
        weight = (RH - 20) / (100 - 20)
        flow = (1 - weight) * flow_20 + weight * flow_100

    return float(flow)*P_cooling/P_cooling_ref

def discharge_temp(dry_temp, RH):
    """
    Bilinear interpolation of blowdown discharge water temperature based on air dry bulb temperature and relative humidity, based on the charts
    provided in Guenand et al. . Assumes blowdown temperature is mostly dependent on ambient conditions and not on power level.
    
    Parameters:

    - `dry_temp` (`float`) : air dry-bulb temperature in °C
    - `RH` (`float`) : relative humidity in percent (0–100)

    Returns:
    - `discharge_temp` (`float`) : blowdown discharge water temperature in °C
    """
    # Ensure RH stays within physical bounds
    RH = np.clip(RH, 0, 100)

    # Evaluate evap_flow at both RH levels
    temp_20 = disch_interp_20(dry_temp)
    temp_100 = disch_interp_100(dry_temp)

    # Linear interpolation based on RH
    # RH20% corresponds to flow_20, RH100% to flow_100
    # Interpolate linearly between RH=20 and RH=100
    if RH >= 100:
        # RH >= 100, just use flow_100
        flow = temp_100
    else:
        # Linear interpolation between flow_20 and flow_100, extrapolation if RH<20
        weight = (RH - 20) / (100 - 20)
        flow = (1 - weight) * temp_20 + weight * temp_100

    return float(flow)

def relative_gross_efficiency(T_basin):
    '''
    Returns plant relative gross electrical efficiency decrease based on basin temperature. Quantifies condenser performance for Rankine cycles.
    Calculated from the characteristic of the Golfech plant, which has 1330 MW gross power. 
    It is assumed that condenser performance is only affected by basin temperature and not by other plant characteristics.

    In:
    T_basin : basin temperature in °C

    Out:
    relative_gross_efficiency : gross electrical efficiency, as a percentage of its nominal value
    '''
    return efficiency_interp(T_basin)
    
print('CoolingTower : Interpolators built')
#### Other utilities

def plotTowerCharacteristics(cooling_power = 2470):
    """
    Plots blowdown water discharge temperature and evaporative flow 
    based on air dry bulb and relative humidity. Recreates the charts from Guenand et al. to ensure they have been accurately copied.

    `cooling_power` (`float`) : heat load on the cooling system (MW). Defaults to 2470, value for Golfech tower.
    """

    T_db_list = np.linspace(-20, 45, 1000)
    RH_list = np.arange(0, 110, 10)

    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Blowdown discharge temperature
    plt.subplot(1, 2, 1)
    for RH in RH_list:
        T_disch_list = [discharge_temp(t, RH) for t in T_db_list]

        plt.plot(T_db_list, T_disch_list, label=f"{RH}% RH")


        T_wetbulb_list = [wet_bulb_temperature(1013 * units.hPa, t * units.degC, dewpoint_from_relative_humidity(t*units.degC, RH * units.percent)).to('degC').magnitude for t in T_db_list]
        plt.plot(T_db_list, T_wetbulb_list, label=f'{RH} RH wetbulb')

    plt.title("Blowdown Discharge Temperature")
    plt.xlabel("Dry-bulb air temperature (°C)")
    plt.ylabel("Discharge water temperature (°C)")
    plt.grid(True)
    plt.legend(title="Relative Humidity")

    # Subplot 2: Evaporated water flow
    plt.subplot(1, 2, 2)
    for RH in RH_list:
        T_evap_list = [evap_flow(t, RH,cooling_power) for t in T_db_list]
        plt.plot(T_db_list, T_evap_list, label=f"{RH}% RH")
    plt.title("Evaporated Water Flow")
    plt.xlabel("Dry-bulb air temperature (°C)")
    plt.ylabel("Evaporated water flow (m³/s)")
    plt.grid(True)
    plt.legend(title="Relative Humidity")

    # Overall title and layout
    plt.suptitle(f"Cooling Tower Characteristics for a cooling load of {cooling_power}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plotCondenserPerformanceModel():
    '''
    Plots condenser performance model created : relative gross efficiency depending on cooling water temperature.
    '''
    T_basin = np.linspace(8,40,1000)
    relative_efficiency = [relative_gross_efficiency(t) for t in T_basin]

    plt.plot(T_basin, relative_efficiency)
    plt.title("Condenser performance model")
    
    plt.xlabel("Basin temperature (°C)")
    plt.ylabel("Condenser relative efficiency")
    plt.grid(True)
    
    plt.show()

def plotGolfechRatios():
    '''
    Plots, at full power, the ratio of cooling energy discharged to the river, and the ratio of evaporation in the tower P_evap / P_tower,
    depending on atmospheric conditions (Tair, RH) for different water temperatures.
    '''

    # Constants

    P_thermal = 3800    # thermal power in MW
    ncc = 1.5           # concentration cycle number
    Tw = [10, 15, 20]   # intake water temperatures
    efficiency_gross = 0.357

    T_db_list = np.linspace(-20, 45, 1000)  # air temperatures
    RH_list = np.arange(0, 110, 10)         # relative humidity values

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)

    for i, T_w in enumerate(Tw):
        for RH in RH_list:
            k_river_list = []
            k_evap_list = []

            for Tair in T_db_list:
                T_basin = discharge_temp(Tair, RH)
                gross_eff_adj = efficiency_gross * relative_gross_efficiency(T_basin)
                P_el_gross = P_thermal * gross_eff_adj

                q_evap = evap_flow(Tair, RH, P_thermal - P_el_gross)
                q_blowdown = q_evap / (ncc - 1)

                P_toriver = rho_w * c_p * q_blowdown * (T_basin - T_w) / 1e6
                P_tower = P_thermal - P_el_gross - P_toriver
                P_vap = rho_w * L_vap * q_evap / 1e6

                k_river = P_toriver / (P_thermal - P_el_gross)
                k_evap = P_vap / P_tower

                k_river_list.append(k_river)
                k_evap_list.append(k_evap)

            axs[i, 0].plot(T_db_list, k_river_list, label=f"{RH}% RH")
            axs[i, 1].plot(T_db_list, k_evap_list)

        # Labels and titles per row
        axs[i, 0].set_title(f"Blowdown heat ratio for Tw = {T_w}°C")
        axs[i, 0].set_ylabel(r"$k_{river}$")
        axs[i, 0].grid(True)

        axs[i, 1].set_title(f"Latent heat ratio in tower for Tw = {T_w}°C")
        axs[i, 1].set_ylabel(r"$k_{evap}$")
        axs[i, 1].grid(True)

    # X-axis labels for bottom row
    for ax in axs[2, :]:
        ax.set_xlabel("Dry-bulb air temperature (°C)")

    # Legend only once for all RH curves (on leftmost plots)
    axs[0, 0].legend(title="Relative Humidity", bbox_to_anchor=(1.05, 1.0), loc='upper left')

    # Improve layout
    plt.suptitle(f"Cooling Tower Characteristics for Golfech at Max Power, assuming ncc = {ncc}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])  # reserve space for suptitle and legend
    plt.savefig('/Users/barnabecerveau/Documents/Travail/X/3A/TU Berlin/Input data/Cooling tower/Tower_ratios.png')
    plt.show()

if __name__ == '__main__':
    plotGolfechRatios()
    plotCondenserPerformanceModel()
    plotTowerCharacteristics(cooling_power=2709*2)