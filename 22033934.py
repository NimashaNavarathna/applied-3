# -*- coding: utf-8 -*-
"""
Created on Thu May 11 02:59:57 2023

@author: Nimasha
"""

import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.datasets as skdat
import sklearn.metrics as skmet
from sklearn.preprocessing import StandardScaler
import cluster_tools as ct
from scipy.optimize import curve_fit

# Read the csv file

df_climate = pd.read_csv("world_bank_data.csv", skiprows=(4))
print(df_climate)

# Transpose the dataset and clean it
df_climated = df_climate.T
df_climated = df_climated.dropna

# create new data frames with only data on Population growth and CO2 emission
df_pop = df_climate[(df_climate["Indicator Name"] ==
                     "Population growth (annual %)")]
df_co2 = df_climate[(df_climate["Indicator Name"] ==
                     "CO2 emissions (metric tons per capita)")]
print(df_pop)
print(df_co2)
# describe the dataframe
print(df_pop.describe())
print(df_co2.describe())

# drop rows with nan's in 2020,2021
df_co2 = df_co2[df_co2["2018"].notna()]
print(df_co2.describe())

# alternative way of targetting one or more columns
df_pop = df_pop.dropna(subset=["2018"])
print(df_pop.describe)

df_co22020 = df_co2[["Country Name", "Country Code", "2018"]].copy()
df_pop2020 = df_pop[["Country Name", "Country Code", "2018"]].copy()

print(df_co22020.describe())
print(df_pop2020.describe())

df_2020 = pd.merge(df_co22020, df_pop2020, on="Country Name", how="outer")
print(df_2020.describe())
df_2020 = df_2020.dropna()
print()
print(df_2020.describe())
# rename columns
df_2020 = df_2020.rename(
    columns={"2018_x": "CO2 Emission", "2018_y": "Population Growth"})
print(df_2020.describe())

# visualize the data distribution
def plot_scat():
    """ Plot the scatter chart using above data frames """
plt.scatter(df_2020['CO2 Emission'], df_2020['Population Growth'])
plt.xlabel("CO2 Emission")
plt.ylabel("Population Growth")
# call the function plot_scat
plot_scat()

pd.plotting.scatter_matrix(df_2020, figsize=(12, 12), s=5, alpha=0.8)

# print correlation between two indicators
print(df_2020.corr())

df_cluster = df_2020[["CO2 Emission", "Population Growth"]].copy()

# normalise,store minimum and maximum
df_cluster, df_min, df_max = ct.scaler(df_cluster)
print("n score")

# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_cluster)  # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_cluster, labels))

ncluster = 4
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_cluster)  # fit done on x,y pairs
labels = kmeans.labels_

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_cluster["CO2 Emission"],
            df_cluster["Population Growth"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("CO2 Emission")
plt.ylabel("Population Growth")
plt.show()

# data fitting
x=df_2020['CO2 Emission'].values
y=df_2020['Population Growth'].values
coefficients=np.polyfit(x,y,1)
print('slope:', coefficients[0])
print('Intercept:', coefficients[1])
