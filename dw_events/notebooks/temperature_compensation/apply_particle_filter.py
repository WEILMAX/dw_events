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
strain_line = 'BCN'
temperature_sensor = '_B'
strain_line_signals = datagetter.get_dataframe_str_subset(strain_line)

# Temperature data
temperature_data = datagetter.get_dataframe_str_subset('TFBG')
bottom_temperature_data = temperature_data.filter(regex=temperature_sensor)

num_particles = 100
r_measurement_noise = 1e2
q_process_noise = np.array([5e-3, 1])
scale = 1e-3


for sensor in range(len(strain_line_signals.columns)):
    Tb = bottom_temperature_data
    delta_Tb = Tb - Tb.shift(int(1))

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


    pf = ParticleFilter(
        num_particles = num_particles,
        r_measurement_noise = r_measurement_noise,
        q_process_noise = q_process_noise,
        scale = scale
    )

    filtered_data = pf.filter(measurements.values, inputs.values)
    filtered_data = pd.DataFrame(
        filtered_data,
        index = measurements.index,
        columns = [strain_line_signals.columns[sensor] + '_filtered']
    )
    filtered_data.to_parquet(
        "../../data/raw/filtered_data/" + strain_line_signals.columns[sensor] + "_filtered.parquet",
        compression="gzip")
