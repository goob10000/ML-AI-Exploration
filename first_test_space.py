import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss




# df = pl.read_csv("ML-AI-Exploration\dataexport_20250301T010932.csv",skip_rows=9)
# df.write_parquet("SFO_weather_data_2008to2025.parquet")
df = pl.read_parquet("SFO_weather_data_2008to2025.parquet")
df
# df.insert_column(1, df['timestamp'].str.slice(0,4).alias("year"))
# df.insert_column(1, df['timestamp'].str.slice(4,2).alias("month"))
# df.insert_column(1, df['timestamp'].str.slice(6,2).alias("day"))
# df.insert_column(1, df['timestamp'].str.slice(9,4).alias("time"))

# df.columns
# df["month"].unique()

zipped = zip(["f"+ str(x) for x in range(0,36)],['Basel Temperature [2 m elevation corrected]', 'Basel Growing Degree Days [2 m elevation corrected]', 
               'Basel Temperature [850 mb]', 'Basel Temperature [700 mb]', 'Basel Sunshine Duration', 
               'Basel Shortwave Radiation', 'Basel Direct Shortwave Radiation', 'Basel Diffuse Shortwave Radiation', 
               'Basel Precipitation Total', 'Basel Snowfall Amount', 'Basel Relative Humidity [2 m]', 
               'Basel Cloud Cover Total', 'Basel Cloud Cover High [high cld lay]', 
               'Basel Cloud Cover Medium [mid cld lay]', 'Basel Cloud Cover Low [low cld lay]', 
               'Basel CAPE [180-0 mb above gnd]', 'Basel Mean Sea Level Pressure [MSL]', 
               'Basel Geopotential Height [1000 mb]', 'Basel Geopotential Height [850 mb]', 
               'Basel Geopotential Height [700 mb]', 'Basel Geopotential Height [500 mb]', 
               'Basel Evapotranspiration', 'Basel FAO Reference Evapotranspiration [2 m]', 
               'Basel Temperature', 'Basel Vapor Pressure Deficit [2 m]', 'Basel Wind Speed [10 m]', 
               'Basel Wind Direction [10 m]', 'Basel Wind Gust', 'Basel Wind Speed [900 mb]', 
               'Basel Wind Direction [900 mb]', 'Basel Wind Speed [850 mb]', 'Basel Wind Direction [850 mb]', 
               'Basel Wind Speed [700 mb]', 'Basel Wind Direction [700 mb]', 'Basel Wind Speed [500 mb]', 
               'Basel Wind Direction [500 mb]'])

col_dict = dict(zipped)

X = df.select(['Basel Temperature [2 m elevation corrected]', 'Basel Growing Degree Days [2 m elevation corrected]', 
               'Basel Temperature [850 mb]', 'Basel Temperature [700 mb]', 'Basel Sunshine Duration', 
               'Basel Shortwave Radiation', 'Basel Direct Shortwave Radiation', 'Basel Diffuse Shortwave Radiation', 
               'Basel Precipitation Total', 'Basel Snowfall Amount', 'Basel Relative Humidity [2 m]', 
               'Basel Cloud Cover Total', 'Basel Cloud Cover High [high cld lay]', 
               'Basel Cloud Cover Medium [mid cld lay]', 'Basel Cloud Cover Low [low cld lay]', 
               'Basel CAPE [180-0 mb above gnd]', 'Basel Mean Sea Level Pressure [MSL]', 
               'Basel Geopotential Height [1000 mb]', 'Basel Geopotential Height [850 mb]', 
               'Basel Geopotential Height [700 mb]', 'Basel Geopotential Height [500 mb]', 
               'Basel Evapotranspiration', 'Basel FAO Reference Evapotranspiration [2 m]', 
               'Basel Temperature', 'Basel Vapor Pressure Deficit [2 m]', 'Basel Wind Speed [10 m]', 
               'Basel Wind Direction [10 m]', 'Basel Wind Gust', 'Basel Wind Speed [900 mb]', 
               'Basel Wind Direction [900 mb]', 'Basel Wind Speed [850 mb]', 'Basel Wind Direction [850 mb]', 
               'Basel Wind Speed [700 mb]', 'Basel Wind Direction [700 mb]', 'Basel Wind Speed [500 mb]', 
               'Basel Wind Direction [500 mb]'])
y = pl.DataFrame(df["month"].cast(pl.Int16)-1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y["month"].value_counts()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)



clf = xgb.XGBClassifier(n_estimators=125, max_depth=9, learning_rate=0.5, min_samples_leaf = 5, 
                        min_samples_split=9, verbose = 2, enable_categorical=True)


params = {
    "objective":"multi:softmax",
    "eval_metric":"mlogloss",
    "num_class":12,
    "max_depth":6,
    "eta":0.5,
    "colsample_bytree":0.8,
    "seed":42
}

tree = xgb.train(params, dtrain, num_boost_round=100)

preds = tree.predict(dtest)

mse = mean_squared_error(y_test, preds)
mse
# loss = log_loss(y_test, preds)
y_test
preds

def month_wrap(a, b):
    case1 = abs(a-b)
    case2 = 12-case1
    if(case1>case2):
        return case2
    else:
        return case1
    
def feature_reference(a):
    out = list(a)
    out[0] = col_dict[a[0]]
    return tuple(out)

plt.scatter(np.arange(500),list(map(month_wrap,preds[:500],y_test[:500,0])),s=0.2)
plt.show()

abs(preds[:500]-y_test[:500,0])

features_d = tree.get_score(importance_type="gain")
features_d_sorted = list(features_d.items())
features_d_sorted.sort(reverse=True,key = lambda x : x[1])
feat_d_sorted_titled = list(map(feature_reference,features_d_sorted))
feat_d_sorted_titled
for element in feat_d_sorted_titled:
    print(f"{element[1]}\t{element[0]}")
    

#create a violin plot for the first column in X sorted by month


plt.scatter(np.arange(X['Basel Snowfall Amount'].shape[0]),X['Basel Snowfall Amount'],s=0.2)
plt.scatter(np.arange(X['Basel Precipitation Total'].shape[0]),X['Basel Precipitation Total'],s=0.2)
plt.scatter(np.arange(X["Basel Temperature [2 m elevation corrected]"].shape[0]),X["Basel Temperature [2 m elevation corrected]"],s=0.2)
plt.show()

'''
Next Steps:
- Principal component analysis, look for redundant and not useful features
    - Graph each variable against the others for chorrelation
- Look at the tree!!
    - Explainer!!
- Entropy?



'''