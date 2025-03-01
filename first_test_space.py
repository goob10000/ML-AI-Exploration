import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pl.read_csv("ML-AI-Exploration\dataexport_20250301T010932.csv",skip_rows=9)
# df.write_parquet("SFO_weather_data_2008to2025.parquet")
df = pl.read_parquet("SFO_weather_data_2008to2025.parquet")
df.insert_column(1, df['timestamp'].str.slice(0,4).alias("year"))
df.insert_column(1, df['timestamp'].str.slice(4,2).alias("month"))
df.insert_column(1, df['timestamp'].str.slice(6,2).alias("day"))
df.insert_column(1, df['timestamp'].str.slice(9,4).alias("time"))
df["time"]

df