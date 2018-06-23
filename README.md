# Data Preparation

## Overview

This section explores, cleans and scales the data to prepare them for a cluster analysis that identifies different weather patterns for a weather station in San Diego, CA using k-means. 

The dataset is described and imported in the [previous section](https://eagronin.github.io/weather-clustering-spark-acquire/).

The analysis is discussed in the [next section](https://eagronin.github.io/weather-clustering-spark-analyze/).

This project is based on assignments from Machine Learning With Big Data by University of California San Diego on Coursera

The analysis for this project was performed in Spark.

## Data Exploration, Cleaning and Scaling

The imported dataset includes over 1.5 million rows, as indicated by `df.count()`.  For the purpose of this analysis a smaller dataset was used that contains only one-tenth of the data.  The following code creates such a subset of data by keeping every 10th row in the subset and dropping all the other rows:

```python
filteredDF = df.filter((df.rowID % 10) == 0)
filteredDF.count()
```

The number of rows in the subset is 158,726.

Below are the summary statistics: 

```python
filteredDF.describe().toPandas().transpose()
```

```
summary			count	mean		stddev		min	max
rowID			158726	793625.0	458203.93	0	1587250
air_pressure		158726	916.83		3.05		905.0	929.5
air_temp		158726	61.85		11.83		31.64	99.5
avg_wind_direction	158680	162.15		95.27		0.0	359.0
avg_wind_speed		158680	2.77		2.05		0.0	31.9
max_wind_direction	158680	163.46		92.45		0.0	359.0
max_wind_speed		158680	3.40		2.41		0.1	36.0
rain_accumulation	158725	3.18E-4		0.01		0.0	3.12
rain_duration		158725	0.40		8.66		0.0	2960.0
relative_humidity	158726	47.60		26.21		0.9	93.0
```

The low average values for rain accumulation and duration in this dataset suggest that the data were collected during a dry period. The code below outputs the counts of days when the values of rain accumulation and duration are 0:

```python
filteredDF.filter(filteredDF.rain_accumulation == 0.0).count()
filteredDF.filter(filteredDF.rain_duration == 0.0).count()
```

For rain accumulation the count is 157,812 days, while for rain duration the count is 157,237 days, which are almost all the days in the sample.  Since the values for these features are almost all 0 (i.e., very limited variation in the data) and for the purpose of speeding up the analyses, these  features are dropped from the DataFrame. We also drop the hpwren_timestamp feature since it is not used in the analysis, as well as rowID since it is the row number:

```python
workingDF = filteredDF.drop('rain_accumulation').drop('rain_duration').drop('hpwren_timestamp').drop('rowID')
```

Next, we drop rows with missing values and count the number of rows that were dropped:

```python
before = workingDF.count()
workingDF = workingDF.na.drop()
after = workingDF.count()
before - after
```

The code above indicate that 46 rows in the `workingDF` dataframe had missing values in at least one feature, before these rows were dropped.  

Next, we combine the remaining features into a single vector column. Let's create an array of the columns we want to combine, and use VectorAssembler to create the vector column:

```python
 featuresUsed = ['air_pressure',
 'air_temp',
 'avg_wind_direction',
 'avg_wind_speed',
 'max_wind_direction',
 'max_wind_speed',
 'relative_humidity']
 
assembler = VectorAssembler(inputCols = featuresUsed, outputCol = 'features_unscaled')
assembled = assembler.transform(workingDF)
```

Finally, since the features have different scales (e.g., air temperature ranges from 31.6 to 99.5, while air pressure ranges from 905.0 to 929.5), they need to be scaled. We scale them using `StandardScaler()` so that each feature has the mean of 0 and the standard deviation of 1:

```
scaler = StandardScaler(inputCol = 'features_unscaled', outputCol = 'features', withStd = True, withMean = True)
scalerModel = scaler.fit(assembled)
scaledData = scalerModel.transform(assembled)
```

Next step: [Analysis](https://eagronin.github.io/weather-clustering-spark-analyze/)
