import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pl.read_csv("ML-AI-Exploration\dataexport_20250301T010932.csv",skip_rows=9)
df.write_parquet("SFO_weather_data_2008to2025.parquet")
df