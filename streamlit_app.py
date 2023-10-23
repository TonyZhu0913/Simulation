from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

from battery import Battery
from optimize import run_simulation
from load_data import load_data
from plot import display_schedule, display_profit, get_stats

import streamlit as st

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd

import gdown

file_id = "1G0V9oNUkeAT4ihhKAuZEXr8yPgiMr9iO"  # Replace this with your file's ID
output_file = "european_wholesale_electricity_price_data_hourly.csv"  # Replace "data_file.ext" with the desired output filename and extension

gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file)

st.set_page_config(layout="wide")

"""
## BESS Simulation
**Run BESS simulation on Europe wholesale electricity market**
"""

col1, col2 = st.columns([1, 2], gap="large")

col1.write("### Settings")
start_date = str(col1.date_input("Start date", value=None))
end_date = str(col1.date_input("End date", value=None))
init_NEC = int(col1.number_input("Initial Battery Capacity (Wh)", value=1000000))
init_eff = float(col1.number_input("Initial Efficiency", value=0.99))
max_cycles = int(col1.number_input("Max Cycles", value=4000))

country = col1.selectbox(
    'Country',
    ('Austria',
    'Belgium',
    'Bulgaria',
    'Switzerland',
    'Czechia',
    'Germany',
    'Denmark',
    'Spain',
    'Estonia',
    'Finland',
    'France',
    'Greece',
    'Croatia',
    'Hungary',
    'Ireland',
    'Italy',
    'Lithuania',
    'Luxembourg',
    'Latvia',
    'North Macedonia',
    'Netherlands',
    'Norway',
    'Poland',
    'Portugal',
    'Romania',
    'Serbia',
    'Slovakia',
    'Slovenia',
    'Sweden'))

col2.write("### Results")
if col1.button('Check availability'):

    SOC = np.array([0, 0.01, 0.85, 1.0])

    # Charging curve
    CR = np.array([0.25, 0.5, 0.5, 0.1])

    SOC_to_CR_function = interp1d(SOC, CR)

    # Discharging curve
    SOC = np.array([0.0, 0.15, 0.99, 1.0])
    DR = np.array([0.1, 0.5, 0.5, 0.25])

    SOC_to_DR_function = interp1d(SOC, DR)

    bat = Battery(SOC_to_CR_function, SOC_to_DR_function, init_NEC=init_NEC, init_eff=init_eff, max_cycles=max_cycles)

    frame_size_forecast = 28
    update_period = 1

    start_date = start_date + " 00:00:00"
    end_date = end_date + " 00:00:00"

    print('Start Date:', start_date)
    print("End Date:", end_date)

    df = load_data(country=country)

    df_optim = run_simulation(
        bat,
        start=start_date,
        end=end_date,
        df=df,
        forecasted=False,
        frame_size=frame_size_forecast,
        update_period=update_period,
    )

    fig = display_profit(df_optim, name=country + "_optim_" + str(frame_size_forecast))
    col2.plotly_chart(fig)

    csv_file = df_optim.to_csv(index=False).encode('utf-8')
    col2.download_button(
        "Press to Download",
        csv_file,
        "file.csv",
        "text/csv",
        key='download-csv'
    )

# # display_schedule(df_optim, name=country + "_optim_" + str(frame_size_forecast))