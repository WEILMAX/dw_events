# Imports
import os
import pandas as pd
from dw_events.temperature_compensation.particle_filter import ParticleFilter
from dw_events.data.make_dataset import DataGetter
from dw_events.data.utils import make_dt_list, get_dataframe_str_subset
import datetime
import matplotlib.pyplot as plt
import numpy as np

start = datetime.datetime(2022, 9, 1)
end = datetime.datetime(2022, 9, 2)
data_path = \
    "_".join(
        [
            "../../data/raw/strain_data",
            str(start.year),
            str(start.month),
            str(start.day),
            str(end.year),
            str(end.month),
            str(end.day),
            ".parquet"
        ]
    )
datagetter = DataGetter(start, end)
try:
    strain_data = pd.read_parquet(data_path)
    datagetter.merged_signals = strain_data
except:
    strain_data = datagetter.get_strain_data(destination = "../../data/raw/SCB_ALM")
    #strain_data.to_parquet(data_path)

# Get strain line data BCN
strain_line = 'TCN'
temperature_sensor = 'TCN'
strain_line_signals = datagetter.get_dataframe_str_subset(strain_line)

# Temperature data
temperature_data = datagetter.get_dataframe_str_subset('TFBG')
temperature_data = temperature_data.filter(regex=temperature_sensor)

num_particles = 2000
r_measurement_noise = 1e3
q_process_noise = np.array([2e-1, 1e-1])
scale = 1
loc = -0.5
loading = 'compression'

for sensor in range(len(strain_line_signals.columns)):
    Tb = temperature_data
    delta_Tb = (Tb - Tb.shift(int(20*60)))/60

    measurements = pd.DataFrame(
        {
            strain_line_signals.columns[sensor]: strain_line_signals.iloc[:, sensor].values
        },
        index = strain_line_signals.index)

    inputs = pd.DataFrame(
        {
            'Tb': Tb.values[:,0],
            'delta_Tb': delta_Tb.values[:,0]
        }, index = measurements.index)

    measurements = measurements.resample('1s').mean().dropna()
    inputs = inputs.dropna().resample('1s').mean().dropna().loc[measurements.index]


    pf = ParticleFilter(
        num_particles = num_particles,
        r_measurement_noise = r_measurement_noise,
        q_process_noise = q_process_noise,
        scale = scale,
        loc = loc
    )

    filtered_data = pf.filter(measurements.values, inputs.values, loading = loading)
    filtered_data = pd.DataFrame(
        filtered_data,
        index = measurements.index,
        columns = [strain_line_signals.columns[sensor] + '_filtered']
    )
    filtered_data.to_parquet(
        "../../data/raw/filtered_data/" + strain_line_signals.columns[sensor] + "_filtered.parquet",
        compression="gzip")
