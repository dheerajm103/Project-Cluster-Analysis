# importing library

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
 
#  importing dataset

df = pd.read_csv("G:/project data/minute_weather.csv")

# EDA and data preprocessing
# random  sampling 10% of data

df1 = df.sample(n = 158725)

# dropping nominal columns and resetting index

df = df.drop(["rowID","hpwren_timestamp"] , axis = 1)
df1 = df1.drop(["rowID","hpwren_timestamp"] , axis = 1)
df1 = df1.reset_index(drop = True)

# checking original data with sample 

describe1 = df.describe().transpose()
describe2 = df1.describe().transpose()

# plotting histogram for comparison of sampe

def func(p,q):
    plt.subplot(1,2,1)
    plt.hist(p)
    plt.subplot(1,2,2)
    plt.hist(q)
   
func(df.iloc[:,[0]],df1.iloc[:,[0]])
func(df.iloc[:,[1]],df1.iloc[:,[1]])
func(df.iloc[:,[2]],df1.iloc[:,[2]])
func(df.iloc[:,[3]],df1.iloc[:,[3]])
func(df.iloc[:,[4]],df1.iloc[:,[4]])
func(df.iloc[:,[5]],df1.iloc[:,[5]])
func(df.iloc[:,[6]],df1.iloc[:,[6]])
func(df.iloc[:,[7]],df1.iloc[:,[7]])
func(df.iloc[:,[8]],df1.iloc[:,[8]])
func(df.iloc[:,[9]],df1.iloc[:,[9]])
func(df.iloc[:,[10]],df1.iloc[:,[10]])

# plotting pairplot for overall comparison

sns.pairplot(df)
sns.pairplot(df1)

# checking correlation , variance and skewness

corr = df1.corr()
var = df1.var()
skew = df1.skew()

# checking uniqueness in imbalanced columns

df1.nunique()
pd.value_counts(df1['rain_accumulation'])
pd.value_counts(df1['rain_duration'])

# dropping imbalanced columns

df2 = df1.drop(["rain_duration"] , axis = 1)

# plotting scatter plot for correlation

plt.scatter(df2.max_wind_speed,df2.avg_wind_speed)
plt.scatter(df2.min_wind_speed,df2.avg_wind_speed)
plt.scatter(df2.min_wind_speed,df2.max_wind_speed)
plt.scatter(df2.air_temp,df2.relative_humidity)

# checking for null values and dropping it

df2.isnull().sum()
df2 = df2.dropna(axis = 0)
df2.isnull().sum()

# checking for duplicate records

df2.duplicated().sum()
df2 = df2.drop_duplicates()
df2 = df2.reset_index(drop = True)

deleted_records = 158725 - 155479
deleted_records

# checking outliers

def func1(r):
    sns.boxplot(r)

func1(df2.air_pressure)
func1(df2.air_temp)
func1(df2.avg_wind_direction)
func1(df2.avg_wind_speed)
func1(df2.max_wind_direction)
func1(df2.max_wind_speed)
func1(df2.min_wind_direction)
func1(df2.min_wind_speed)
func1(df2.relative_humidity)

# removing outliers

df2.air_pressure = winsorize(df2.air_pressure, limits=[0.01, 0.011])
df2.air_temp = winsorize(df2.air_temp, limits=[0, 0.0001])
df2.avg_wind_speed = winsorize(df2.avg_wind_speed, limits=[0, 0.032])
df2.max_wind_speed = winsorize(df2.max_wind_speed, limits=[0, 0.038])
df2.min_wind_speed = winsorize(df2.min_wind_speed, limits=[0, 0.032])

plt.boxplot(df2)

# finding k value from scree plot

TWSS = []

k = list(range(2,15))

for i in  k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df2)
    TWSS.append(kmeans.inertia_)
TWSS    

# plotting scree plot

plt.plot(k,TWSS,"ro-")
plt.xlabel("clusters")
plt.ylabel("inertia")

# initialising and fitting kmeans

model = KMeans(n_clusters = 5).fit(df2)
model.labels_

cluster_label = pd.Series(model.labels_)
cl = cluster_label

# obtaning sihouette score

score = silhouette_score(df2, cl, metric='euclidean')
score

# plotting obtained clusters

df2["y"] = cl

label_0 = df2[df2.y == 0]
label_1 = df2[df2.y == 1]
label_2 = df2[df2.y == 2]
label_3 = df2[df2.y == 3]
label_4 = df2[df2.y == 4]

plt.scatter(label_0.iloc[:,[1]], label_0.iloc[:,[2]], color = 'red')
plt.scatter(label_1.iloc[:,[1]], label_1.iloc[:,[2]], color = 'black')
plt.scatter(label_2.iloc[:,[1]], label_2.iloc[:,[2]], color = 'blue')
plt.scatter(label_3.iloc[:,[1]], label_3.iloc[:,[2]], color = 'yellow')
plt.scatter(label_4.iloc[:,[1]], label_4.iloc[:,[2]], color = 'green')

# grouping data with clusters for prediction

p = label_0[label_0["relative_humidity"] < 50]
q = label_1[label_1["relative_humidity"] < 50]
r = label_2[label_2["relative_humidity"] < 50]
s = label_3[label_3["relative_humidity"] < 50]
t = label_4[label_4["relative_humidity"] < 50]
pred = df2.iloc[:, 0:12].groupby(df2.y).mean()              
pred







