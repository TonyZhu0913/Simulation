import pandas as pd
import numpy as np
from amplpy import AMPL, modules
import datetime 

import datetime 
import pandas as pd
import numpy as np
from amplpy import AMPL, modules
import datetime 

import os 
uuid = os.environ.get("4b3bc435-ad7b-4678-9d85-829c70ca3dcf")  # Use a free https://ampl.com/ce license
if uuid is not None:
    modules.activate(uuid)  # activate your license


def run_simulation(bat, df, start, end, forecasted=True, frame_size=14, update_period=1, forecasting_model = None ):
    """
    Run a simulation starting from the start-th day of the dataframe.
    For every day of the simulation, a schedule is generated (either based on true prices or prediction) and different 
    metrics are recorded.
    We return a dataframe containing the results of the simulation
    """
    
    try :
        end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=1)
        start_index = df.index.get_loc(df.index[df.timestamp==start][0]) 
        end_index = df.index.get_loc(df.index[df.timestamp==end][0])
    except : raise ValueError("The dataframe does not contain all the data between the start and end dates")
    start_df = start_index - 24 * frame_size if forecasted else start_index
    if  start_df < 0 : raise ValueError("The dataframe does not contain enough data for the price prediction relying on the {} last days to be computed".format(frame_size))
    
    df = df.iloc[start_df: end_index+1, :]
    n_hours = (end_index-start_df) + 1
    
    if  n_hours% 24 != 0:
        raise Exception(
            "The dataframe should contain only full days (24 hours)")


    bat.reset()  # start with a new battery, and get the max SOC change when charging and discharging
    G_c, G_d = bat.max_SOC_change_charge, bat.min_SOC_change_discharge

    n_cycles_list = np.zeros(n_hours)
    eff_list = np.zeros(n_hours)
    NEC_list = np.zeros(n_hours)
    price_forecast_list = np.zeros(n_hours)
    schedule = np.zeros(n_hours)





    # optimization done for each day :
    for i, day in enumerate(range((frame_size if forecasted else 0), n_hours//24)):

        day_indices = slice(day*24, (day+1)*24)

        # if using forecasted prices, get new forecast evert update_period iterations :
        if forecasted and (i % update_period == 0):
            if forecasting_model :
                prices = forecasting_model(df.iloc[(day-frame_size)*24:day*24, :][0])
            else : prices = df.iloc[(day-frame_size)*24:day*24, :].groupby(
                df.timestamp.dt.hour).price_euros_wh.mean().to_numpy()
            

        # Otherwise, use the true prices for the current day
        if not forecasted:
            prices = df.iloc[day_indices].price_euros_wh.to_numpy()

        # get the variable grid cost
        vgc = df.vgc.iloc[day*24:(day+1)*24].to_numpy()

        # get the fixed grid cost
        fgc = df.fgc.iloc[day*24:(day+1)*24].to_numpy()

        # store battery state
        n_cycles_list[day_indices] = bat.n_cycles
        eff_list[day_indices] = bat.eff
        NEC_list[day_indices] = bat.NEC
        price_forecast_list[day_indices] = prices

        # get optimized schedule
        schedule[day_indices] = get_daily_schedule(
            prices, vgc, fgc, bat, G_c, G_d)

    ## store simulation results 
    df = df.assign(n_cycles=n_cycles_list,
                   eff=eff_list, 
                   NEC=NEC_list,
                   price_forecast=price_forecast_list,
                   schedule=schedule,
                   capacity=np.hstack(
                       (np.array([0]), np.cumsum(schedule)[:-1])),
                   SOC=lambda x: 100 * x.capacity/x.NEC,
                   charge_energy=lambda x: x.schedule.mask(x.schedule < 0, 0), ## energy delivered to the battery
                   discharge_energy=lambda x: -
                   x.schedule.mask(x.schedule > 0, 0) * x.eff, ## energy obtained from the battery (taking into account the discharge efficiency)
                   electricity_revenue=lambda x: x.price_euros_wh * ## net revenue from electricity trading (before grid costs)
                   (x.discharge_energy - x.charge_energy),
                   grid_cost=lambda x: x.vgc * ## grid costs
                   (x.discharge_energy + x.charge_energy) +
                   x.fgc * (abs(x.schedule) > 10**-5),
                    variable_grid_cost=lambda x: x.vgc * ## grid costs
                   (x.discharge_energy + x.charge_energy),
                   fixed_grid_cost = lambda x: x.fgc * (abs(x.schedule) > 10**-5),
                   hourly_profit=lambda x: x.electricity_revenue - x.grid_cost ## profits
                   )

    return df.iloc[(frame_size if forecasted else 0) * 24:]


def get_daily_schedule(prices, vgc, fgc, bat, G_c, G_d):
    """
    Obtain schedule given the battery model, prices, vgc and fgc.
    """

    ## the arrays have to contain the data for the 24 hours of the day
    if not (len(prices == 24) and len(vgc) == 24 and len(fgc) == 24) :
        raise Exception(
            "The arrays should contain the data for a full day (24 hours)")

    ## instantiate AMPL object and load the model
    modules.load()  # load all AMPL modules
    ampl = AMPL()  
    # ampl.read("ampl/ampl.mod")  
    ampl.eval(r'''
############################# PARAMS AND SETS ##################


# hours in a day
set H ordered := {0..23};

# ----------- Battery specs -------------
# nominal energy capacity
param NEC >= 0 default 100000;
# initial energy capacity
param EC_init >= 0, <= NEC default 0;
# degradation and efficiency coefficient
param eff >0, <=1 default 1.00;

# ----------- constants -------------
# eps 
param eps := 0.01;
# big M to compute is_charging and is_discharging
param M := NEC;



# ----------- Charge/ discharge curve discretization -------------

# number of intervals
param Nint >= 1, <= 100 default 5;
# incremental - number of intervals
param inc := 1/Nint;
# interval indices 
set I ordered := {1..Nint};
# cut points indices 
set C ordered := {0..Nint};
# cut points
param S{i in C} := i*inc;
# max negative change in energy capacity (discharge)
param G_d{C} >= -NEC, <= 0 default -NEC/2;
# max positive change in energy capacity (charge)
param G_c{C} >=0, <= NEC default NEC/2;





# ----------- Price and costs -------------
# price of electricity per hour
param p{H};
# fixed grid cost per hour
param fgc{H} >= 0 default 0;
# variable grid cost per hour
param vgc{H} >= 0 default 0;


# ----------- Availability Constraints -------------
# minimum SOC required 
param min_SOC{H} >= 0, <= 1 default 0;
# max SOC required 
param max_SOC{H} >= 0, <= 1 default 1;

# ----------- UNUSED -------------
# current cycle count 
param CCC{H} >= 0 default 0;
param fc >=0 default 0;


############################# VARIABLES ##################


# ----------- Decision variables --------------
# energy "in" per hour
var x{H} >= -NEC, <= NEC;




# ----------- operation booleans -------------
# boolean indicating if we are charging or discharging (1), or holding (0)
var is_charging_or_discharging{H} binary; 



# ----------- Charge/ discharge curve discretization -------------
# interval indicator
var in_interval{I,H} binary;
# convex combination weights
var interval_start_w{I,H} >=0, <= 1 default 1;
var interval_end_w{I,H} >= 0, <= 1 default 0;

# ----------- positive and negative parts -------------

var x_p{H} >=0, <=NEC;
var x_n{H} >=0, <=NEC;
var y{H} binary;



############################# CONSTRAINTS ##################



#------------------- Auxiliary variables ----------------
subject to x_decomposition {i in H} :
    x[i] == x_p[i] - x_n[i];

subject to positive_part {i in H} :
    x_p[i] <= M * y[i];

subject to negative_part {i in H} :
    x_n[i] <= M * (1-y[i]);



# ---------- charging/discharging or no action on the battery -------
subject to is_charging_or_discharging_right {i in H} : 
    x_n[i] + x_p[i] >= eps*is_charging_or_discharging[i];

subject to is_charging_or_discharging_left {i in H} : 
    x_n[i] + x_p[i] <= M * is_charging_or_discharging[i];



#------------------- Availability constraints ----------------
# keep SOC within bounds 
subject to availability_constraint {i in H}:
    min_SOC[i] <= EC_init/NEC + sum{t in 0..i} x[t] / NEC <= max_SOC[i];



# ------------------- Find discretization parameters of SOC ------------
subject to find_weights_for_each_interval {i in 1..23} : 
    sum{k in I} (interval_start_w[k,i]*S[k-1]+ interval_end_w[k,i]*S[k]) == EC_init/NEC + sum{t in 0..i-1} x[t] / NEC;


subject to find_weights_for_each_interval_0 : 
    sum{k in I} (interval_start_w[k,0]*S[k-1]+ interval_end_w[k,0]*S[k]) == EC_init/NEC;


subject to comvex_combination_constraint {k in I, i in H} : 
    interval_start_w[k,i] + interval_end_w[k,i] == in_interval[k,i];

subject to select_one_interval {i in H} : 
    sum{k in I} in_interval[k,i] == 1;



#-------------------Charge/ Discharge rate constraints ----------------
# energy increment should be below maximum increment 
subject to energy_increment {i in H}:
    sum{k in 1..Nint} (interval_start_w[k,i] * G_c[k-1] + interval_end_w[k,i] * G_c[k]) >= x[i];

# energy increment should be above minimum increment 
subject to energy_decrease {i in H}:
    sum{k in 1..Nint} (interval_start_w[k,i] * G_d[k-1] + interval_end_w[k,i] * G_d[k]) <= x[i];


############################# OBJECTIVE FUNCTION ##################

maximize profit :
    - sum{i in H} (
        (x_p[i]-x_n[i]*eff) * p[i] 
        + (x_p[i]+x_n[i]*eff) * vgc[i] 
        + is_charging_or_discharging[i] * fgc[i]
        );
    
    ''')

    ## set parameters 
    ampl.get_parameter("vgc").set_values(vgc)
    ampl.get_parameter("fgc").set_values(fgc)
    ampl.get_parameter("p").set_values(prices)
    ampl.get_parameter("eff").set_values([bat.eff])
    ampl.get_parameter("Nint").set_values([bat.Nint])
    ampl.get_parameter("max_SOC").set_values([1]*23 + [0])
    ampl.get_parameter("G_c").set_values(np.array(G_c)*bat.NEC)
    ampl.get_parameter("G_d").set_values(np.array(G_d)*bat.NEC)
    ampl.get_parameter("NEC").set_values([bat.NEC])

    ## solve and get optimization solution
    ampl.option["solver"] = "gurobi"
    ampl.solve()
    daily_schedule = ampl.get_variable('x').get_values().to_pandas()[
        "x.val"].to_numpy()
    

    # print(ampl.get_variable('x').get_values().to_pandas()[
    #     "x.val"].to_numpy())
    
    # print(ampl.get_variable('is_charging_or_discharging').get_values().to_pandas()[
    # "is_charging_or_discharging.val"].to_numpy())
    # ampl.reset()

    ## update battery state
    bat.n_cycles += abs(daily_schedule).sum()/(2*bat.init_NEC)

    return daily_schedule
