import numpy as np
import pandas as pd
from datetime import datetime, timedelta as td
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

from datetime import timedelta

from Code import Loire

################## Capacity plotting
CATEGORIES = [
        '> 95% Capacity',
        '90%–95% Capacity',
        '75%–90% Capacity',
        '50%–75% Capacity',
        '> 0%–50% Capacity',
        '0% Capacity (Shutdown)'
    ]

COLORS = {
        '> 95% Capacity': '#1a9850',
        '90%–95% Capacity': '#66bd63',
        '75%–90% Capacity': '#fee08b',
        '50%–75% Capacity': '#f46d43',
        '> 0%–50% Capacity': '#d73027',
        '0% Capacity (Shutdown)': '#7f0000'
    }

# Capacity category classification
def categorise_capacity(power, nominal_capacity):
    if power < 1e-3:
        return '0% Capacity (Shutdown)'
    pct = power / nominal_capacity
    if pct > 0.95:
        return '> 95% Capacity'
    elif pct > 0.90:
        return '90%–95% Capacity'
    elif pct > 0.75:
        return '75%–90% Capacity'
    elif pct > 0.50:
        return '50%–75% Capacity'
    elif pct > 0:
        return '> 0%–50% Capacity'
    else:
        return '0% Capacity (Shutdown)'

def plotDaysByCapacity(plant_list, regiondir, gcm_rcm, scenario):
    """
    Plots and saves the number of days per year in various capacity retention categories for each plant.
    Expects timeseries files under : regiondir/scenario/gcm_rcm/Timeseries/plantname_scenario_gcm_rcm_timeseries.csv
    Saves plot under : regiondir/scenario/gcm_rcm/Plots/plantname_scenario_gcm_rcm_capacity_days.png

    Parameters:
        plant_list : list of PowerPlant instances
        base_dir : Base path containing input and output directories
        gcm_rcm : String for the climate model
        scenario : Emissions scenario
    """
    input_dir = os.path.join(regiondir, f'{scenario}', f'{gcm_rcm}', 'Timeseries')
    output_dir = os.path.join(regiondir, f'{scenario}', f'{gcm_rcm}', 'Plots')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    for plant in plant_list:
        input_filename = f'{plant.name}_{scenario}_{gcm_rcm}_timeseries.csv'
        csv_path = os.path.join(input_dir, input_filename)

        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, parse_dates=["Gregorian_day"])
        nominal_capacity = plant.capacity_net
        df['Year'] = df['Gregorian_day'].dt.year

        df['Capacity_Category'] = df['Power'].apply(categorise_capacity, args=(nominal_capacity,))

        # Count days in each category per year
        grouped = df.groupby(['Year', 'Capacity_Category']).size().unstack()

        # Ensure all categories and years are present
        for cat in CATEGORIES:
            if cat not in grouped.columns:
                grouped[cat] = 0

        all_years = sorted(df['Year'].unique())
        for year in all_years:
            if year not in grouped.index:
                grouped.loc[year] = 0

        grouped = grouped.fillna(0)
        grouped = grouped[CATEGORIES].sort_index()

        # Plot stacked bars with spacing and edge outlines
        fig, ax = plt.subplots(figsize=(max(10, len(grouped) * 0.1), 6))

        # Tighter spacing between bars
        x = np.linspace(0, len(grouped) - 1, len(grouped)) * 0.05
        bar_width = 0.03
        bottoms = np.zeros(len(grouped))

        for cat in CATEGORIES:
            values = grouped[cat].values
            ax.bar(
                x,
                values,
                bottom=bottoms,
                color=COLORS[cat],
                edgecolor='black',
                linewidth=1,
                width=bar_width,
                label=cat
            )
            bottoms += values

        ax.set_title(f"Days per Year by Capacity Category ({nominal_capacity} MW): {plant.name}\nScenario: {scenario}, Model: {gcm_rcm}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Days")
        ax.set_ylim(top= 400)
        ax.legend(title="Capacity Category", loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        xtick_labels = [str(year) if year % 5 == 0 else '' for year in grouped.index]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=90)

        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{plant.name}_{scenario}_{gcm_rcm}_capacity_days.png")
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        print(f"Saved plot for {plant.name} to {plot_filename}")

def plotGroupedCapacityDays(plants, regiondir, scenario, gcm_rcm, savepath=None):
    """
    Plots and saves the number of days per year in various capacity retention categories
    for each plant, arranged in a 2-row grid of subplots.

    Expects timeseries files under:
        regiondir/scenario/gcm_rcm/Timeseries/plantname_scenario_gcm_rcm_timeseries.csv

    Saves plot under:
        regiondir/scenario/gcm_rcm/Plots/grouped_scenario_gcm_rcm_capacity_days.png
        or savepath if defined
    """
    input_dir = os.path.join(regiondir, scenario, gcm_rcm, "Timeseries")
    output_dir = os.path.join(regiondir, scenario, gcm_rcm, "Plots")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    n_plants = len(plants)
    n_cols = int(np.ceil(n_plants / 2))  # 2 rows
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10), sharey=True)

    # Flatten axes array to index easily, even if n_cols == 1
    axes = axes.flatten()

    for i, plant in enumerate(plants):
        ax = axes[i]
        input_filename = f"{plant.name}_{scenario}_{gcm_rcm}_timeseries.csv"
        csv_path = os.path.join(input_dir, input_filename)

        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            ax.set_visible(False)
            continue

        df = pd.read_csv(csv_path, parse_dates=["Gregorian_day"])
        nominal_capacity = plant.capacity_net
        df["Year"] = df["Gregorian_day"].dt.year
        df["Capacity_Category"] = df["Power"].apply(categorise_capacity, args=(nominal_capacity,))

        # Count days in each category per year
        grouped = df.groupby(["Year", "Capacity_Category"]).size().unstack()

        # Ensure all categories and years are present
        for cat in CATEGORIES:
            if cat not in grouped.columns:
                grouped[cat] = 0

        all_years = sorted(df["Year"].unique())
        for year in all_years:
            if year not in grouped.index:
                grouped.loc[year] = 0

        grouped = grouped.fillna(0)
        grouped = grouped[CATEGORIES].sort_index()

        # Plot stacked bars with spacing and edge outlines
        x = np.linspace(0, len(grouped) - 1, len(grouped)) * 0.05
        bar_width = 0.03
        bottoms = np.zeros(len(grouped))

        for cat in CATEGORIES:
            values = grouped[cat].values
            ax.bar(
                x,
                values,
                bottom=bottoms,
                color=COLORS[cat],
                edgecolor="black",
                linewidth=0.3,
                width=bar_width,
                label=cat
            )
            bottoms += values

        ax.set_title(f"{plant.name} ({nominal_capacity} MW)")
        ax.set_ylim(top=400)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        xtick_labels = [str(year) if year % 5 == 0 else "" for year in grouped.index]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=90)

        if i % n_cols == 0:  # left column
            ax.set_ylabel("Days")

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    # One legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Capacity Category", loc="center right")

    # Global title
    fig.suptitle(
        f"Simulation results for scenario {scenario}, model {gcm_rcm}",
        fontsize=18,
        y=0.98
    )

    fig.tight_layout(h_pad=4, w_pad=4)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend
    if savepath is None:
        savepath = os.path.join(output_dir, f"grouped_{scenario}_{gcm_rcm}_capacity_days.png")

    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"Saved grouped plot to {savepath}")

################# Cooling tower analysis 

def plotTowerParameters(plant, regiondir, scenario, model,
                                  savepath = 'Inputs/Cooling tower'):

    # Load CSV
    timeseries_path = os.path.join(regiondir, scenario, model, 'Timeseries', f'{plant.name}_{scenario}_{model}_timeseries.csv')
    df = pd.read_csv(timeseries_path)

    # Ensure 'Gregorian_day' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Gregorian_day']):
        df['Gregorian_day'] = pd.to_datetime(df['Gregorian_day'])

    # Compute energy balance parameters
    expected_args = ['Q', 'Tw', 'Tair', 'RH']
    def compute_balance(row):
        kwargs = {arg: row[arg] for arg in expected_args if arg in df.columns}
        return plant.computeCoolingTowerEnergyBalance(**kwargs)

    balance_df = df.apply(compute_balance, axis=1, result_type='expand')
    balance_df.columns = ['k_river', 'k_evap']
    df = pd.concat([df, balance_df], axis=1)

    # Plot: two vertically stacked subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(df['Gregorian_day'], df['k_river'], color='blue', label='k_river')
    axs[0].set_ylabel('k_river')
    axs[0].set_title(f'{plant.name} Cooling Tower Energy Balance\nScenario: {scenario} | Model: {model}')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(df['Gregorian_day'], df['k_evap'], color='green', label='k_evap')
    axs[1].set_ylabel('k_evap')
    axs[1].set_xlabel('Date')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f'{plant.name}_tower_1.png'))
    plt.show()

    # Create day-of-year column
    df['doy'] = df['Gregorian_day'].dt.dayofyear

    # Compute daily stats
    seasonal_stats = df.groupby('doy')[['k_river', 'k_evap']].agg(['mean', 'std', 'min', 'max', 'quantile'])

    # Compute percentiles manually
    seasonal_kriver = df.groupby('doy')['k_river']
    seasonal_kevap  = df.groupby('doy')['k_evap']

    kriver_mean = seasonal_kriver.mean()
    kriver_p25  = seasonal_kriver.quantile(0.25)
    kriver_p75  = seasonal_kriver.quantile(0.75)

    kevap_mean = seasonal_kevap.mean()
    kevap_p25  = seasonal_kevap.quantile(0.25)
    kevap_p75  = seasonal_kevap.quantile(0.75)

    # Plot with shaded percentile range
    fig2, axs2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # k_river
    axs2[0].plot(kriver_mean.index, kriver_mean, color='blue', label='Mean k_river')
    axs2[0].fill_between(kriver_mean.index, kriver_p25, kriver_p75, color='blue', alpha=0.2, label='25–75th percentile')
    axs2[0].set_ylabel('k_river')
    axs2[0].legend()
    axs2[0].grid(True)
    axs2[0].set_title(f'Seasonal Distribution – {plant.name} Tower ({scenario} / {model})')

    # k_evap
    axs2[1].plot(kevap_mean.index, kevap_mean, color='green', label='Mean k_evap')
    axs2[1].fill_between(kevap_mean.index, kevap_p25, kevap_p75, color='green', alpha=0.2, label='25–75th percentile')
    axs2[1].set_ylabel('k_evap')
    axs2[1].set_xlabel('Day of Year')
    axs2[1].legend()
    axs2[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f'{plant.name}_tower_2.png'))
    plt.show()

################## CVar analysis

def writeGeneralTimeseriesFile(plant_list, regiondir, scenario, gcm_rcm, input_dir = None, output_dir = None):
    '''
    Aggregates all plant-level power output timeseries into a single file with one column per plant
    and an additional column 'Total_Power' for the sum across all plants.

    Standard naming can be overriden using kwargs input_dir and output_dir:

    Looks for timeseries under regiondir/scenario/model/Timeseries/ or under input_dir/

    Saves the combined DataFrame to:

    regiondir/scenario/gcm_rcm/Timeseries/general_{scenario}_{gcm_rcm}_timeseries.csv or 
    output_dir/general_{scenario}_{gcm_rcm}_timeseries.csv
    
    '''
    
    timeseries_dir = input_dir if input_dir else os.path.join(regiondir, scenario, gcm_rcm, 'Timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)

    merged_df = None

    for plant in plant_list:
        filename = f"{plant.name}_{scenario}_{gcm_rcm}_timeseries.csv"
        filepath = os.path.join(timeseries_dir, filename)

        if not os.path.isfile(filepath):
            print(f"[Warning] File not found: {filepath}")
            continue
        
        try:
            df = pd.read_csv(filepath, parse_dates=['Gregorian_day'])
            df = df[['Julian_day', 'Gregorian_day', 'Power']].copy()
            df.rename(columns={'Power': plant.name}, inplace=True)
        except KeyError as e:
                print(f"KeyError encountered: {e} in directory {regiondir}, scenario {scenario}, model {gcm_rcm}, for plant {plant.name}")
                print(f"For plant : {plant}")
                print(f'Input dir overriden : {input_dir}')
                print(f'Output dir overriden : {output_dir}')
                sys.stdout.flush()
                raise
        
        except Exception as e:
            print(f"Unexpected error: {e} in directory {regiondir}, scenario {scenario}, model {gcm_rcm}, when simulating plant {plant.name}.")
            raise

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=['Julian_day', 'Gregorian_day'], how='outer')

    if merged_df is not None:
        merged_df.sort_values(by='Julian_day', inplace=True)
        merged_df['Total_Power'] = merged_df[[plant.name for plant in plant_list if plant.name in merged_df.columns]].sum(axis=1)

        output_filename = f"general_{scenario}_{gcm_rcm}_timeseries.csv"
        outputdir = output_dir if output_dir else timeseries_dir

        output_path = os.path.join(outputdir, output_filename)
        merged_df.to_csv(output_path, index=False)
        print(f"[Info] General timeseries saved: {output_path}")
    else:
        print("[Error] No valid timeseries files found to merge.")

def calculateCVar(plant_list, scenario, gcm_rcm, regiondir, p=0.05, 
                  input_path = None, output_dir = None):
    """
    Robust pCVar/DR calculation using exact top-k selection (no quantile-tie ambiguity).
    Standard naming can be overriden using kwargs input_path and output_dir

    Searches for general timeseries file under 
    
    regiondir/scenario/model/Timeseries/general_{scenario}_{gcm_rcm}_timeseries.csv
    or input_path

    Saves pCVar_{scenario}_{gcm_rcm}.csv under regiondir/scenario/gcm_rcm/pCVar/.

    
    """

    timeseries_path = input_path if input_path else os.path.join(regiondir, scenario, gcm_rcm, 'Timeseries', f'general_{scenario}_{gcm_rcm}_timeseries.csv')
    if not os.path.isfile(timeseries_path):
        print(f"[Error] Timeseries file not found: {timeseries_path}")
        return

    df = pd.read_csv(timeseries_path, parse_dates=['Gregorian_day'])

    # Ensure plant order and presence
    plants = [pl for pl in plant_list if pl.name in df.columns]
    plant_names = [pl.name for pl in plants]
    N = len(df)
    k = max(1, int(np.ceil(p * N)))  # exact tail size

    # Fill any missing plant values with 0 (or decide other policy)
    df_plants = df[plant_names].fillna(0)

    # Precompute shortfalls in MW: shortfall = capacity - output
    shortfalls = pd.DataFrame(index=df.index, columns=plant_names, dtype=float)
    for pl in plants:
        shortfalls[pl.name] = pl.capacity_net - df_plants[pl.name]
        # clip to [0, capacity] if needed
        shortfalls[pl.name] = shortfalls[pl.name].clip(lower=0.0, upper=pl.capacity_net)

    # For each plant: take top-k shortfalls (largest shortfalls)
    plant_pCVar_frac = {}
    plant_sum_topk = {}
    for pl in plants:
        vals = shortfalls[pl.name].values
        if k >= len(vals):
            topk = vals.copy()
        else:
            # partition is faster than sort for large arrays
            idx = np.argpartition(vals, -k)[-k:]
            topk = vals[idx]
        plant_sum_topk[pl.name] = np.sum(topk)                # sum of MW shortfalls on plant's worst k days
        plant_pCVar_frac[pl.name] = np.mean(topk) / pl.capacity_net  # average shortfall fraction over plant's worst k days

    # System: compute total shortfall time series and take top-k days
    total_capacity = sum(pl.capacity_net for pl in plants)
    total_shortfall_ts = shortfalls.sum(axis=1).values  # MW shortfall of the system each day
    if k >= len(total_shortfall_ts):
        sys_topk = total_shortfall_ts.copy()
    else:
        sys_idx = np.argpartition(total_shortfall_ts, -k)[-k:]
        sys_topk = total_shortfall_ts[sys_idx]
    system_sum_topk = np.sum(sys_topk)
    system_pCVar_frac = np.mean(sys_topk) / total_capacity  # fraction

    # Weighted average from plant sums (equivalent to capacity-weighted average of plant fractions)
    # Numerator = sum_i plant_sum_topk[i]  (sum of MW shortfalls on each plant's own worst k days)
    numerator_mw = sum(plant_sum_topk.values())
    denominator_mw = system_sum_topk  # sum of MW shortfalls on system worst k days
    diversification_ratio = (numerator_mw / denominator_mw) if denominator_mw > 0 else None

    # Prepare output rows (keep percent outputs consistent with your format)
    rows = []
    for pl in plants:
        rows.append({
            'Plant': pl.name,
            'Capacity': pl.capacity_net,
            'pCVar': round(plant_pCVar_frac[pl.name] * 100, 2)
        })
    rows.append({
        'Plant': 'System_Total',
        'Capacity': total_capacity,
        'pCVar': round(system_pCVar_frac * 100, 2)
    })
    # Weighted average in percent (use sums so no rounding issues)
    weighted_avg_frac = numerator_mw / total_capacity / k  # because numerator_mw is sum over k-days-per-plant, divide by k then by total_capacity
    rows.append({
        'Plant': 'Weighted_Average',
        'Capacity': total_capacity,
        'pCVar': round(weighted_avg_frac * 100, 2)
    })
    rows.append({
        'Plant': 'Diversification_Ratio',
        'Capacity': '',
        'pCVar': round(diversification_ratio, 6) if diversification_ratio is not None else None
    })

    outputdir = output_dir if output_dir else os.path.join(regiondir, scenario, gcm_rcm, 'pCVar')
    os.makedirs(outputdir, exist_ok=True)

    output_path = os.path.join(outputdir, f'pCVar_{scenario}_{gcm_rcm}.csv') 
    pd.DataFrame(rows).to_csv(output_path, index=False)

    # Also save a small diagnostic CSV listing:
    # - plant name, k, plant_sum_topk, plant_avg_frac, plus system_sum_topk
    diag = []
    for pl in plants:
        diag.append({
            'Plant': pl.name,
            'k_tail_days': k,
            'plant_sum_topk_MW': plant_sum_topk[pl.name],
            'plant_avg_topk_fraction': plant_pCVar_frac[pl.name]
        })
    diag.append({
        'Plant': 'System_Total',
        'k_tail_days': k,
        'plant_sum_topk_MW': system_sum_topk,
        'plant_avg_topk_fraction': system_pCVar_frac
    })
    pd.DataFrame(diag).to_csv(os.path.join(outputdir, f'pCVar_{scenario}_{gcm_rcm}_diagnostic.csv'), index=False)

    print(f"[Info] pCVar results saved to: {output_path}")
    print(f"[Info] Diagnostic saved to: {os.path.join(outputdir, f'pCVar_{scenario}_{gcm_rcm}_diagnostic.csv')}")
    print(f"[Info] Diversification ratio (top-k) = {diversification_ratio}")

def computeCVarEvolution(regiondir : str, scenario : str, model : str, plant_list, 
                         startdate : datetime, timedelta, p : float =0.05):
    '''
    Computes individual and system-wide pCVar over successive periods (starting from startdate)
    and saves the results to a CSV.

    Parameters:
        - regiondir: root directory of timeseries
        - scenario: climate scenario name
        - model: GCM-RCM model name
        - plant_list: list of plant objects (with .name and .capacity_net)
        - startdate: datetime object (start of first window)
        - timedelta: timedelta object (length of each period)
        - p: tail probability for CVaR (e.g. 0.05 for 5% worst cases)
    '''
    try: 
        ts_path = os.path.join(regiondir, scenario, model, 'Timeseries', f'general_{scenario}_{model}_timeseries.csv')
    except TypeError as e:
        print(f"Encountered type error when computing CVar evolution, for scenario {scenario} and model {model}.")
        print(f'Scenario : {scenario}')
        print(f'Model : {model}')
        print(f'Regiondir : {regiondir}')
        print(f'Attempted file path : {regiondir}/{scenario}/{model}/Timeseries/general_{scenario}_{model}_timeseries.csv')

    if not os.path.isfile(ts_path):
        print(f"[Error] Timeseries file not found: {ts_path}")
        return

    df = pd.read_csv(ts_path, parse_dates=['Gregorian_day'])

    results = []

    current_start = startdate
    max_date = df['Gregorian_day'].max()

    while current_start <= max_date:
        # Last time window is usually incomplete : if complete by more than 50% then it is kept
        if max_date - current_start < timedelta:
            if max_date - current_start > timedelta/2 :
                print(f'Warning : shorter last time window for pCvar calculation under scenario {scenario} and model {model}.')
                print(f'Last time window starting at {current_start}, for an end date of {max_date} : difference {max_date - current_start}, expected {timedelta}. Still included in results.')
            else :
                print(f'Warning : shorter last time window for pCvar calculation under scenario {scenario} and model {model}.')
                print(f'Last time window starting at {current_start}, for an end date of {max_date} : difference {max_date - current_start}, expected {timedelta}. Excluded from results.')
                break
        current_end = current_start + timedelta
        window_df = df[(df['Gregorian_day'] >= current_start) & (df['Gregorian_day'] < current_end)]

        if window_df.empty:
            current_start += timedelta
            continue

        # Individual plant pCVaRs
        valid_plants = []
        plant_names = []
        for plant in plant_list:
            if plant.name not in window_df.columns:
                print(f'No timeseries found for {plant.name}')
                continue
            unavailability = 1 - (window_df[plant.name] / plant.capacity_net)
            unavailability = unavailability.clip(lower=0, upper=1)
            vals = unavailability.values
            k = max(1, int(np.ceil(p * len(vals))))
            if k >= len(vals):
                topk = vals.copy()
            else:
                idx = np.argpartition(vals, -k)[-k:]
                topk = vals[idx]
            pCVar = np.mean(topk)
            results.append({
                'Period_Start': current_start.date(),
                'Period_End': current_end.date(),
                'Plant': plant.name,
                'Capacity': plant.capacity_net,
                'pCVar': round(pCVar * 100, 2)
            })
            valid_plants.append((plant.name, plant.capacity_net, unavailability))
            plant_names.append(plant.name)

        # System-wide pCVar
        total_capacity = sum(cap for _, cap, _ in valid_plants)
        if total_capacity > 0:
            total_power = window_df[plant_names].sum(axis=1)
            total_unavailability = 1 - (total_power / total_capacity)
            total_unavailability = total_unavailability.clip(lower=0, upper=1)
            vals = total_unavailability.values
            k = max(1, int(np.ceil(p * len(vals))))
            if k >= len(vals):
                topk = vals.copy()
            else:
                idx = np.argpartition(vals, -k)[-k:]
                topk = vals[idx]
            system_pCVar = np.mean(topk)

            results.append({
                'Period_Start': current_start.date(),
                'Period_End': current_end.date(),
                'Plant': 'System_Total',
                'Capacity': total_capacity,
                'pCVar': round(system_pCVar * 100, 2)
            })


        current_start += timedelta

    output_dir = os.path.join(regiondir, scenario, model, 'pCVar')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'pCVar_evolution_{scenario}_{model}.csv')

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"[Info] pCVar evolution saved to: {output_path} \n")

def compute_allPCVar(plant_list, regiondir, valid_combinations, n_years= 45, p=0.05):
    ''' 
    Computes all pCVar files for a given plant list

    - `plant_list` : list of `PowerPlant` instances representing plants to simulate
    - `regiondir` (`str`) : path to home simulation directory
    - `p` (`float`) : quantile for p-CVar calculation, set to 0.05 by default
    '''
    os.makedirs(regiondir, exist_ok=True)

    for scenario, model in valid_combinations:
        calculateCVar(plant_list,scenario,model,regiondir,p)
        computeCVarEvolution(plant_list=plant_list, regiondir=regiondir, scenario=scenario, model=model, 
                                 startdate=datetime(2010, 1, 1), timedelta=timedelta(days=365 * n_years), p=p)

def plotSystemCVarTrajectories(regiondir : str, region : str, save_path=None):
    '''
    Plots the evolution of system-wide pCVar for all scenario/model combinations.
   
    If save_path is provided, plot is saved to file.
    
    Looks for files under regiondir/scenario/model/pCvar/pCvar_evolution_scenario_model.csv

    **Inputs**

    - `regiondir`(`str`) : path to home directory from which individual simulation files are retrieved
    - `savepath`(`str`) : path to which save the figure if wanted
    '''
    ###### Models dictionary to update when using new data!!!
    models = {
        'CNRM-CM5-LR_ALADIN63': '-',
        'IPSL-CM5A-MR_WRF381P': '-.',
        'HadGEM2_CCLM4-8-17': '--'
    }
    scenarios = {
        'rcp26': 'green',
        'rcp45': 'orange',
        'rcp85': 'red'
    }

    plt.figure(figsize=(10, 6))

    match region :
        case 'Loire':
            valid_comb = Loire.VALID_COMBINATIONS
        case 'Germany':
            valid_comb = None
        case _ :
            raise ValueError(f'Unsupported region for CVar trajectory processing : {region}')
        
    for scenario, model in valid_comb:
        csv_path = os.path.join(
            regiondir, scenario, model, 'pCVar', f'pCVar_evolution_{scenario}_{model}.csv'
        )

        if not os.path.isfile(csv_path):
            print(f"[Warning] Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path, parse_dates=['Period_Start', 'Period_End'])
        df = df[df['Plant'] == 'System_Total']

        if df.empty:
            continue

        plt.plot(
            df['Period_Start'],
            df['pCVar'],
            label=f"{scenario} - {model}",
            linestyle=models[model],
            color=scenarios[scenario]
        )

    plt.xlabel("Period Start")
    plt.ylabel("System pCVar (%)")
    plt.title("System-wide pCVar Evolution by Scenario and Model")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Info] Plot saved to {save_path}")
    else:
        plt.show()

def buildRegionPCVarRecapCSV(regiondir, valid_combinations):
    ''' 
    Makes a recap CSV with plant/system wide pCVar and DR values under each scenario/model.
    Produces a wide-format table: one column per scenario/model.
    
    Input:
    - regiondir (`str`) : home directory for simulation output
    - valid_combinations : set of supported (scenario,model) for that region
    
    Output:
    - Saves to regiondir/pCVar_DR_recap.csv
    - Returns recap_df (pd.DataFrame)
    '''
    
    recap_df = None
    
    for scenario, model in valid_combinations:
        file_path = os.path.join(regiondir, scenario, model, "pCVar", f"pCVar_{scenario}_{model}.csv")
        if not os.path.isfile(file_path):
            print(f"[Warning] File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)

            # Keep only Plant, Capacity, and pCVar
            df = df[['Plant', 'Capacity', 'pCVar']].copy()

            # Rename pCVar column -> "scenario -- model"
            scenario_model_col = f"{scenario} -- {model}"
            df.rename(columns={'pCVar': scenario_model_col}, inplace=True)

            if recap_df is None:
                # Initialize recap_df with Plant, Capacity
                recap_df = df
            else:
                # Merge on Plant + Capacity
                recap_df = pd.merge(recap_df, df, on=['Plant', 'Capacity'], how='outer')

        except Exception as e:
            print(f"[Error] Failed to process {file_path}: {e}")
            continue
    
    # Custom ordering : plants alphabetically then the final recap rows
    specials = ["System_Total", "Weighted_Average", "Diversification_Ratio"]
    plant_rows = recap_df[~recap_df['Plant'].isin(specials)].sort_values(by="Plant")
    special_rows = [recap_df[recap_df['Plant'] == s] for s in specials]
    recap_df = pd.concat([plant_rows] + special_rows, ignore_index=True)

    # Save final recap table
    output_path = os.path.join(regiondir, "pCVar_DR_recap.csv")
    recap_df.to_csv(output_path, index=False)
    print(f"Recap CSV saved to {output_path}")
    
    return recap_df

def buildOverallPCVarRecapCSV(homedir, configurations, regions, scenario_model_dict):
    ''' 
    Makes a recap CSV with the system pCVar and DR value under each configuration/region/scenario/model.
    Retrieves data from homedir/configname/region/scenario/model/pCVar/pCVar_scenario_model.csv
    CSV is saved to homedir/pCVar_DR_recap.csv

    homedir (`str`) : home directory for simulation outputs
    configurations : a list of configurations. A configuration is a tuple whose first attribute is its name (`str`)
    regions : list of `region` (`str`)
    scenario_model_dict : a {region : valid_combinations} dictionary, where valid_combinations is the set of supported (scenario,model) for that region
    '''
    
    records = []

    for config in configurations:
        config_name = config[0]
        for region in regions:
            valid_combos = scenario_model_dict.get(region, [])
            for scenario, model in valid_combos:
                file_path = os.path.join(
                    homedir, config_name, region, scenario, model, "pCVar",
                    f"pCVar_{scenario}_{model}.csv"
                )
                if not os.path.isfile(file_path):
                    continue

                try:
                    df = pd.read_csv(file_path)

                    # System pCVar
                    sys_row = df[df['Plant'] == "System_Total"]
                    sys_pCVar = float(sys_row['pCVar'].values[0]) if not sys_row.empty else np.nan

                    # DR value
                    dr_row = df[df['Plant'] == "Diversification_Ratio"]
                    dr_value = float(dr_row['pCVar'].values[0]) if not dr_row.empty else np.nan

                    records.append({
                        'region': region,
                        'configuration': config_name,
                        'scenario': scenario,
                        'model': model,
                        'pCVar': sys_pCVar,
                        'DR': dr_value
                    })

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    recap_df = pd.DataFrame(records)
    output_path = os.path.join(homedir, "pCVar_DR_recap.csv")
    recap_df.to_csv(output_path, index=False)
    print(f"Recap CSV saved to {output_path}")

    return recap_df

################### Sensitivity analysis

def plotDiversificationRatios(homedir, regions):
    ''' 
    Makes a scatter plot describing dispersion of the diversification ratio metric, for each region.
    Retrieves DR data from the CSV built by `buildOverallPCVarRecapCSV`, under homedir/pCVar_DR_recap.csv
    Plot is saved to homedir/diversification_ratio.png

    homedir (`str`) : home directory for simulation outputs
    configurations : a list of configurations. A configuration is a tuple whose first attribute is its name (`str`)
    regions : list of `region` (`str`)
    scenario_model_dict : a {region : valid_combinations} dictionary, where valid_combinations is the set of supported (scenario,model) for that region
    '''

    recap_df = pd.read_csv(os.path.join(homedir, 'pCVar_DR_recap.csv'))

    # Prepare plot data: gather DR values for each region
    dr_data_plot = []
    for region in regions:
        dr_values = recap_df.loc[recap_df['region'] == region, 'DR'].dropna().tolist()
        dr_data_plot.append(dr_values)

    # Plot boxplots
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(
        dr_data_plot,
        labels=regions,
        patch_artist=True,
        showfliers=True,   # show outliers as points
        widths=0.6
    )

    # Style boxplots
    colors = plt.cm.Set3.colors  # pick a nice palette
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel("Diversification Ratio")
    plt.title("Distribution of Diversification Ratio across Regions")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(homedir, "diversification_ratio.png"), dpi=300)
    print(f'Diversification ratio plot saved to {homedir}/diversification_ratio.png')
    plt.close()

def plotGeneralSensitivity(homedir, regions, scenario_model_dict):
    ''' 
    Plots sensitivity of results to climatic input parameters.
    Retrieves pCVar data from homedir/pCVar_DR_recap.csv
    Plot is saved to homedir/region_pCVar_sensitivity.png

    homedir (`str`) : home directory for simulation outputs
    configurations : a list of configurations. A configuration is a tuple whose first attribute is its name (`str`)
    regions : list of `region` (`str`)
    scenario_model_dict : a {region : valid_combinations} dictionary, where valid_combinations is the set of supported (scenario,model) for that region
    '''

    # Scenario colour mapping
    scenario_colors = {
        'rcp26': 'green',
        'rcp45': 'orange',
        'rcp85': 'red'
    }

    # Distinct line styles for models
    model_linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]

    # Categories for subplots (order matters)
    sensitivity_categories = {
        "StaticTower": ["base", "staticTower"],
        "Streamflow": ["Q_90", "Q_925","Q_95","Q_975", "base","Q_1025", "Q_105","Q_1075", "Q_110"],
        "Air Temperature": ["Tair_-2","Tair_-15", "Tair_-1", "Tair_-05", "base", "Tair_05", "Tair_1","Tair_15", "Tair_2"],
        "Water Temperature": ["Tw_-2","Tw_-15", "Tw_-1", "Tw_-05", "base", "Tw_05", "Tw_1","Tw_15", "Tw_2"],
        "Relative Humidity": ["RH_-10", "RH_-75", "RH_-5","RH_-25", "base","RH_25", "RH_5","RH_75", "RH_10"]
    }

    # Readable tick labels
    config_labels = {
        # Static tower
        "base": "Base",
        "staticTower": "Static",

        # Streamflow (percent of baseline)
        "Q_90": "90%", "Q_925": "92.5%", "Q_95": "95%", "Q_975": "97.5%",
        "Q_1025": "102.5%", "Q_105": "105%", "Q_1075": "107.5%", "Q_110": "110%",

        # Air temp (°C deviations)
        "Tair_-2": "-2", "Tair_-15": "-1.5", "Tair_-1": "-1", "Tair_-05": "-0.5",
        "Tair_05": "+0.5", "Tair_1": "+1", "Tair_15": "+1.5", "Tair_2": "+2",

        # Water temp (°C deviations)
        "Tw_-2": "-2", "Tw_-15": "-1.5", "Tw_-1": "-1", "Tw_-05": "-0.5",
        "Tw_05": "+0.5", "Tw_1": "+1", "Tw_15": "+1.5", "Tw_2": "+2",

        # Relative Humidity (% deviations)
        "RH_-10": "-10", "RH_-75": "-7.5", "RH_-5": "-5", "RH_-25": "-2.5",
        "RH_25": "+2.5", "RH_5": "+5", "RH_75": "+7.5", "RH_10": "+10",
    }

    category_units = {
        "StaticTower": "",  # no units
        "Streamflow": " (% of baseline)",
        "Air Temperature": " (°C)",
        "Water Temperature": " (°C)",
        "Relative Humidity": " (% points)"
    }
    
    recap_path = os.path.join(homedir, "pCVar_DR_recap.csv")
    if not os.path.isfile(recap_path):
        raise FileNotFoundError(f"Recap CSV not found: {recap_path}")

    recap_df = pd.read_csv(recap_path)

    for region in regions:
        region_df = recap_df[recap_df['region'] == region].copy()

        if region_df.empty:
            print(f"No data for region {region}, skipping.")
            continue

        # Prepare figure as 2 rows x 3 columns
        nrows, ncols = 2, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), sharey=True)

        # Flatten axes for easy iteration
        axes = axes.flatten()

        models_seen_global = {}

        # Now zip only the first 5 axes with your categories
        for ax, (cat_name, config_order) in zip(axes[:len(sensitivity_categories)], sensitivity_categories.items()):
            # Map codes to pretty labels
            tick_labels = [config_labels.get(cfg, cfg) for cfg in config_order]

            ax.set_xticks(range(len(config_order)))
            ax.set_xticklabels(tick_labels, rotation=45)

            # Add units to title if available
            ax.set_title(cat_name + category_units.get(cat_name, ""))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            valid_combos = scenario_model_dict.get(region, [])
            for scenario, model in valid_combos:
                base_row = region_df[(region_df['configuration'] == "base") &
                                    (region_df['scenario'] == scenario) &
                                    (region_df['model'] == model)]
                if base_row.empty or pd.isna(base_row['pCVar'].values[0]):
                    continue
                base_val = base_row['pCVar'].values[0]

                y_vals = []
                for cfg in config_order:
                    cfg_row = region_df[(region_df['configuration'] == cfg) &
                                        (region_df['scenario'] == scenario) &
                                        (region_df['model'] == model)]
                    if not cfg_row.empty and pd.notna(cfg_row['pCVar'].values[0]):
                        y_vals.append(cfg_row['pCVar'].values[0] / base_val)
                    else:
                        y_vals.append(None)

                if model not in models_seen_global:
                    models_seen_global[model] = model_linestyles[len(models_seen_global) % len(model_linestyles)]

                ax.plot(range(len(config_order)),
                        y_vals,
                        color=scenario_colors.get(scenario, 'gray'),
                        linestyle=models_seen_global[model],
                        marker='o',
                        alpha=0.8)

        # Hide the 6th subplot (bottom-right) and use it for legends
        axes[-1].axis('off')

        # Create legend handles
        scenario_legend_handles = [
            mlines.Line2D([], [], color=color, marker='o', linestyle='-', label=scenario)
            for scenario, color in scenario_colors.items()
        ]
        model_legend_handles = [
            mlines.Line2D([], [], color='black', linestyle=style, label=model)
            for model, style in models_seen_global.items()
        ]

        # First legend: Scenarios (put higher up in the blank subplot)
        legend1 = axes[-1].legend(
            handles=scenario_legend_handles,
            title="Scenario (Color)",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.65)   # shift upward
        )
        axes[-1].add_artist(legend1)

        # Second legend: Models (put lower down in the blank subplot)
        legend2 = axes[-1].legend(
            handles=model_legend_handles,
            title="Model (Line style)",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.25)   # shift downward
        )
        fig.suptitle(f"pCVar Sensitivity - {region}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(homedir, f"{region}_pCVar_sensitivity.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sensitivity plot saved: {output_path}")

def plotNccSensitivity(refplant, configdir, scenario, model, ncc_list):
    """
    Plots a sensitivity analysis of the n_CC parameter for a given reference plant.
    Creates a grid of subplots (2x4) showing the capacity category distribution
    for each ncc value, with the last subplot reserved for the legend.

    Parameters:
        refplant : PowerPlant instance (reference subunit)
        configdir : Directory containing the ncc_sensitivity_* folders
        scenario : Emissions scenario
        model : GCM/RCM model string
        ncc_list : iterable of ncc values (float)
    """

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.flatten()

    nominal_capacity = refplant.capacity_net
    pCVar_values = []

    # Loop through first 6 ncc values
    for idx, ncc in enumerate(ncc_list[:6]):
        ax = axes[idx]

        foldername = f"ncc_{int(ncc*10)}"
        timeseries_dir = os.path.join(configdir, foldername)
        input_filename = f"{refplant.name}_{scenario}_{model}_timeseries.csv"
        csv_path = os.path.join(timeseries_dir, input_filename)

        if not os.path.exists(csv_path):
            ax.set_visible(False)
            continue

        df = pd.read_csv(csv_path, parse_dates=["Gregorian_day"])
        df['Year'] = df['Gregorian_day'].dt.year
        df['Capacity_Category'] = df['Power'].apply(categorise_capacity, args=(nominal_capacity,))
        grouped = df.groupby(['Year', 'Capacity_Category']).size().unstack()

        for cat in CATEGORIES:
            if cat not in grouped.columns:
                grouped[cat] = 0
        grouped = grouped.fillna(0)[CATEGORIES].sort_index()

        x = np.linspace(0, len(grouped) - 1, len(grouped)) * 0.05
        bar_width = 0.03
        bottoms = np.zeros(len(grouped))

        for cat in CATEGORIES:
            values = grouped[cat].values
            ax.bar(
                x,
                values,
                bottom=bottoms,
                color=COLORS[cat],
                edgecolor='black',
                linewidth=0.4,
                width=bar_width,
                label=cat if idx == 0 else None
            )
            bottoms += values

        ax.set_title(f"ncc = {ncc:.1f}", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Days")
        ax.set_ylim(top=400)
        xtick_labels = [str(year) if year % 5 == 0 else '' for year in grouped.index]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Retrieve pCVar value from dedicated pCVar file
        pCVar_file = os.path.join(timeseries_dir, f"pCVar_{scenario}_{model}.csv")
        if os.path.exists(pCVar_file):
            df_pCVar = pd.read_csv(pCVar_file)
            val = df_pCVar.loc[df_pCVar['Plant'] == 'Weighted_Average', 'pCVar'].values
            if len(val) > 0:
                pCVar_values.append(val[0])
            else:
                pCVar_values.append(np.nan)
        else:
            pCVar_values.append(np.nan)

    # Last two subplots:
    # -2 → pCVar vs ncc
    # -1 → legend
    pCVar_ax = axes[-2]
    pCVar_ax.plot(ncc_list[:6], pCVar_values, marker='o', linestyle='-', color='b')
    pCVar_ax.set_title(f"{refplant.name} pCVar vs ncc", fontsize=14)
    pCVar_ax.set_xlabel("ncc")
    pCVar_ax.set_ylabel("pCVar")
    pCVar_ax.grid(True)

    legend_ax = axes[-1]
    legend_ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, title="Capacity Category", loc="center")

    # Global title
    fig.suptitle(
        f"Sensitivity analysis for parameter ncc\nPlant {refplant.name}",
        fontsize=18,
        y=0.98
    )

    fig.tight_layout(h_pad=4, w_pad=4)
    save_path = os.path.join(configdir, f"{refplant.name}_ncc_sensitivity.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Sensitivity plot saved to {save_path}")

def plotEfficiencySensitivity(refplant, configdir, scenario, model, eff_list):
    """
    Plots a sensitivity analysis of the efficiency_net parameter for a given reference plant.
    Creates a grid of subplots showing the capacity category distribution
    for each efficiency_net value.

    Parameters:
        refplant : PowerPlant instance (reference subunit)
        configdir : Directory containing the efficiency_sensitivity_* folders
        scenario : Emissions scenario
        model : GCM/RCM model string
        eff_list : iterable of efficiency_net values (float)
    """
    n_plots = len(eff_list)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axes = axes.flatten()

    pCVar_values = []

    nominal_capacity = refplant.capacity_net

    for idx, eff in enumerate(eff_list):
        ax = axes[idx]

        foldername = f"eff_{int(eff*100)}"
        timeseries_dir = os.path.join(configdir, foldername)

        # Collect pCVar
        pCVar_file = os.path.join(timeseries_dir, f"pCVar_{scenario}_{model}.csv")
        if os.path.exists(pCVar_file):
            df_pCVar = pd.read_csv(pCVar_file)
            val = df_pCVar.loc[df_pCVar['Plant'] == 'Weighted_Average', 'pCVar'].values
            if len(val) > 0:
                pCVar_values.append(val[0])
            else:
                pCVar_values.append(np.nan)
        else:
            pCVar_values.append(np.nan)

        # Draw other plots
        input_filename = f"{refplant.name}_{scenario}_{model}_timeseries.csv"
        csv_path = os.path.join(timeseries_dir, input_filename)

        if not os.path.exists(csv_path):
            ax.set_visible(False)
            continue

        df = pd.read_csv(csv_path, parse_dates=["Gregorian_day"])
        df['Year'] = df['Gregorian_day'].dt.year
        df['Capacity_Category'] = df['Power'].apply(categorise_capacity, args=(nominal_capacity,))

        grouped = df.groupby(['Year', 'Capacity_Category']).size().unstack()

        # Ensure all categories exist
        for cat in CATEGORIES:
            if cat not in grouped.columns:
                grouped[cat] = 0

        grouped = grouped.fillna(0)
        grouped = grouped[CATEGORIES].sort_index()

        x = np.linspace(0, len(grouped) - 1, len(grouped)) * 0.05
        bar_width = 0.03
        bottoms = np.zeros(len(grouped))

        for cat in CATEGORIES:
            values = grouped[cat].values
            ax.bar(
                x,
                values,
                bottom=bottoms,
                color=COLORS[cat],
                edgecolor='black',
                linewidth=0.4,
                width=bar_width,
                label=cat if idx == 0 else None
            )
            bottoms += values

        ax.set_title(f"efficiency_net = {eff:.2f}", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Days")
        ax.set_ylim(top=400)
        xtick_labels = [str(year) if year % 5 == 0 else '' for year in grouped.index]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.7)


    # Hide any unused subplots except the last two (reserved for pCVar plot and legend)
    for ax in axes[n_plots:-2]:
        ax.set_visible(False)

    # Second to last subplot: pCVar vs efficiency
    pCVar_ax = axes[-2]
    pCVar_ax.plot(eff_list[:6], pCVar_values, marker='o', linestyle='-', color='b')
    pCVar_ax.set_title(f"{refplant.name} pCVar vs efficiency", fontsize=14)
    pCVar_ax.set_xlabel("efficiency_net")
    pCVar_ax.set_ylabel("pCVar")
    pCVar_ax.grid(True)

    # Last subplot for legend
    legend_ax = axes[-1]
    legend_ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, title="Capacity Category", loc="center")

    # Global title
    fig.suptitle(
        f"Sensitivity analysis for parameter efficiency_net\nPlant {refplant.name}",
        fontsize=18,
        y=0.98
    )

    fig.tight_layout(h_pad=4, w_pad=4)
    save_path = os.path.join(configdir, f"{refplant.name}_efficiency_sensitivity.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Sensitivity plot saved to {save_path}")
