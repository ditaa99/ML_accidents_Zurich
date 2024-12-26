"""
Traffic Accident Analysis in Zurich
- Data Source: Swiss Open Data Portal
- Author: Dita Pelaj

Expanding a previous data analysis project to include machine learning techniques.
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import geopandas as gpd
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandasql as ps
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor



# Specify the file path or URL
file_path = 'RoadTrafficAccidentLocations.csv'

# Read the CSV data
accidentDf = pd.read_csv(file_path)

# Load the Zurich map for location visualization of the accidents
zurich_map = gpd.read_file("zurich.geojson")
zurich_map = zurich_map.to_crs("EPSG:2056") # map coordinates on swiss coordinates

# Print the DataFrame
# print(accidentDf)

'''check uniqueness and change datatypes for faster processing'''
# print(accidentDf['AccidentType'].unique())
# print(accidentDf['AccidentType_en'].unique())
accidentDf['AccidentType'] = accidentDf['AccidentType'].astype('category')
accidentDf['AccidentType_en'] = accidentDf['AccidentType_en'].astype('category')
# print(accidentDf.info())

# print(accidentDf['AccidentSeverityCategory'].unique())
# print(accidentDf['AccidentSeverityCategory_en'].unique())
accidentDf['AccidentSeverityCategory'] = accidentDf['AccidentSeverityCategory'].astype('category')
accidentDf['AccidentSeverityCategory_en'] = accidentDf['AccidentSeverityCategory_en'].astype('category')
# print(accidentDf.info())

# print(accidentDf['RoadType'].unique())
# print(accidentDf['RoadType_en'].unique())
accidentDf['RoadType'] = accidentDf['RoadType'].astype('category')
accidentDf['RoadType_en'] = accidentDf['RoadType_en'].astype('category')
# print(accidentDf.info())


# basic info about the dataset
print("Dataset Summary:")
print(accidentDf.info())
# print("\nFirst 5 Rows of the Dataset:")
# print(accidentDf.head())

# duplicates and remove them; i commented them after running them once
# print("\nNumber of Duplicates:", accidentDf.duplicated().sum())
# accidentDf.drop_duplicates(inplace=True)

# Check for null values and handle them; i commented them after running them once
# print("\nNull Values in the Dataset:")
# print(accidentDf.isnull().sum())

# Display summary statistics
print("\nSummary Statistics for Numeric Columns:")
print(accidentDf.describe())

# SQL-like operation: Select specific columns and filter data
query = """
SELECT AccidentYear, COUNT(*) as TotalAccidents
FROM accidentDf
WHERE AccidentYear >= 2013 AND AccidentYear <= 2017
GROUP BY AccidentYear
ORDER BY TotalAccidents DESC
"""
result = ps.sqldf(query, locals())
print("\nSQL Query Result:")
print(result)


'''Bar Plot for Accident Types with Labels'''
# Create a dictionary for accident types
accident_types = {
    "at0": "Skidding/self-accident",
    "at1": "Overtaking/changing lanes",
    "at2": "Rear-end collision",
    "at3": "Turning left or right",
    "at4": "Turning-into main road",
    "at5": "Crossing the lane",
    "at6": "Head-on collision",
    "at7": "Parking",
    "at8": "Pedestrians",
    "at9": "Animals",
    "at00": "Other"
}

# Replace the codes in the 'AccidentType' column with their descriptions
accidentDf['AccidentType_en'] = accidentDf['AccidentType'].map(accident_types)

# Get the frequency of different types of accidents
accident_type_freq = accidentDf['AccidentType_en'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(accident_type_freq.index, accident_type_freq.values)

# Add labels to the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 20, round(yval, 1), ha='center', va='bottom')
    
fig.autofmt_xdate() # Rotate the x-axis labels and make them fit into the figure
ax.set_title('Frequency of Different Types of Accidents')
ax.set_xlabel('Accident Type')
ax.set_ylabel('Frequency')
# plt.savefig("./plots/accident_types.png")
plt.show()

'''Plot the frequency of accidents for each day of the week.'''

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
accident_day_freq = accidentDf['AccidentWeekDay_en'].value_counts()
accident_day_freq = accident_day_freq.reindex(days_order)# Reindex the series to order the days of the week from Monday to Sunday

# Print the frequency of accidents for each day of the week
print("\nFrequency of Accidents for Each Day of the Week:")
print(accident_day_freq)


fig, ax = plt.subplots(figsize=(10, 6))
accident_day_freq.plot(kind='bar', ax=ax)
fig.autofmt_xdate()

ax.set_title('Frequency of Accidents for Each Day of the Week')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Frequency')
# plt.savefig("./plots/frequency_accidents.png")
plt.show()

# Group the data by 'AccidentWeekDay_en' and 'AccidentHour', and count the number of accidents in each group
accident_hour_day = accidentDf.groupby(['AccidentWeekDay_en', 'AccidentHour']).size()
accident_hour_day = accident_hour_day.reset_index() # Reset the index of the DataFrame
accident_hour_day.columns = ['AccidentWeekDay_en', 'AccidentHour', 'Count'] # Rename the columns

# Find the peak hour of accidents for each day of the week
peak_hour_day = accident_hour_day.loc[accident_hour_day.groupby('AccidentWeekDay_en')['Count'].idxmax()]

# Print the peak hour of accidents for each day of the week
print("\nPeak Hour of Accidents for Each Day of the Week:")
print(peak_hour_day)

'''Plot the number of accidents over time'''
# Convert the 'AccidentYear' column to datetime format
accidentDf['AccidentYear'] = pd.to_datetime(accidentDf['AccidentYear'], format='%Y')

# Group the data by 'AccidentYear' and count the number of accidents in each year
accidents_per_year = accidentDf.groupby(accidentDf['AccidentYear'].dt.year).size()


fig, ax = plt.subplots(figsize=(10, 6))

# line plot
ax.plot(accidents_per_year.index, accidents_per_year.values, marker='o', linestyle='-', color='b',
        label='Total Accidents')

# Highlight the peak year with a red star marker
peak_year = accidents_per_year.idxmax()
ax.plot(peak_year, accidents_per_year.max(), marker='*', markersize=10, color='r', label=f'Peak Year: {peak_year}')


ax.set_title('Number of Accidents Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Accidents')
ax.legend()
# plt.savefig("./plots/accidents_over_time.png")
plt.show()

'''I see there is a huge increase in 2014-2016 so i want to analyze it better (+-1year) and try to find out what could have caused it'''
# Filter the data for the time period from 2013 to 2017
filtered_df = accidentDf[(accidentDf['AccidentYear'].dt.year >= 2013) & (accidentDf['AccidentYear'].dt.year <= 2017)]

# Group the filtered data by 'AccidentYear' and count the number of accidents in each year
accidents_per_year = filtered_df.groupby(filtered_df['AccidentYear'].dt.year).size()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(accidents_per_year.index, accidents_per_year.values)

ax.set_title('Number of Accidents Over Time (2013-2017)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Accidents')
plt.show()

'''I notice that 2018 was the peak year of accidents, let's have a closer look into it'''
# Filter the data for the year 2018
accidents_2018 = accidentDf[accidentDf['AccidentYear'].dt.year == 2018]

# Group the data by 'AccidentMonth_en' and 'AccidentType_en' and count the number of accidents in each month for each type
accidents_per_month_type_2018 = accidents_2018.groupby(['AccidentMonth_en', 'AccidentType_en'],
                                                       observed=True).size().unstack().reindex(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
     'December'])

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(16, 8))

# Define a color map for different types of accidents
colors = [plt.colormaps['tab10'](i / len(accidents_per_month_type_2018.columns)) for i in range(len(accidents_per_month_type_2018.columns))]
colors[1] = '#FF1493'#two categories had the same color so i changed one of them

# Plot the frequency of accidents for each month with different colors for each type
accidents_per_month_type_2018.plot(
    kind='bar', stacked=True, ax=ax,
    color=colors
)

fig.autofmt_xdate()
ax.set_title('Monthly Distribution of Accidents in 2018 by Type')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Accidents')
ax.legend(title='Accident Type', bbox_to_anchor=(1, 1))
# plt.savefig("./plots/2018.png")
plt.show()


'''Identify patterns in accident types using categorical data analysis'''
# Create a pivot table to analyze accident types over the years
accident_types_over_time = accidentDf.pivot_table(
    index='AccidentYear', 
    columns='AccidentType_en', 
    aggfunc='size', 
    observed=False
)

fig, ax = plt.subplots(figsize=(12, 8))
accident_types_over_time.plot(kind='line', ax=ax, linewidth=2.5) # line plot, thicker lines

ax.set_title('Accident Types Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Frequency')
plt.legend(title='Accident Type', loc='upper left')
# plt.savefig("./plots/patterns.png")
plt.show()

'''Accident Types by Severity'''
# Create a pivot table to analyze accident types based on severity
accident_types_by_severity = accidentDf.pivot_table(
    index='AccidentType_en', 
    columns='AccidentSeverityCategory_en', 
    aggfunc='size', 
    observed=False 
)

fig, ax = plt.subplots(figsize=(12, 8))


accident_types_by_severity.plot(kind='bar', stacked=True, ax=ax) # stacked bar chart
fig.autofmt_xdate() 

ax.set_title('Accident Types by Severity')
ax.set_xlabel('Accident Type')
ax.set_ylabel('Frequency')
plt.legend(title='Severity', loc='upper right', bbox_to_anchor=(1.12, 1.1))
# plt.savefig("./plots/types_by_severity.png")
plt.show()

'''severity of accidents on each day of the week'''
# Create a cross-tabulation of 'AccidentSeverityCategory' and 'AccidentWeekDay_en'
crosstab = pd.crosstab(accidentDf['AccidentWeekDay_en'], accidentDf['AccidentSeverityCategory_en'])
crosstab = crosstab.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

fig, ax = plt.subplots(figsize=(10, 6))
crosstab.plot(kind='bar', stacked=True, ax=ax)
fig.autofmt_xdate()


ax.set_title('Severity of Accidents on Each Day of the Week')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Accidents')
ax.legend(title='Severity', loc='upper right', bbox_to_anchor=(1.12, 1.1))
# plt.savefig("./plots/severity.png")
plt.show()


'''Accident Types Heatmap'''
# Create a cross-tabulation for accident types
accident_types_cross_tab = pd.crosstab(accidentDf['AccidentWeekDay_en'], accidentDf['AccidentType_en'])

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
accident_types_cross_tab = accident_types_cross_tab.reindex(weekday_order)

total_accidents_by_type = accident_types_cross_tab.sum(axis=0) # Calculate the total number of accidents for each accident type

# Sort the accident types based on the total number of accidents in descending order
sorted_accident_types = total_accidents_by_type.sort_values(ascending=False).index 
accident_types_cross_tab = accident_types_cross_tab[sorted_accident_types]

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(accident_types_cross_tab, cmap='Blues', annot=True, fmt='d', linewidths=0.5, ax=ax) # heatmap for the correlation between accident types and weekdays
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax.set_title('Accident Types Heatmap')
ax.set_xlabel('Accident Type')
ax.set_ylabel('Day of the Week')
# plt.savefig("./plots/heatmap.png")
plt.show()

'''expanding the proj'''

'''Correlation Matrix'''
# Identify and exclude non-numeric columns
numeric_columns = accidentDf.select_dtypes(include=['number']).columns
correlation_matrix = accidentDf[numeric_columns].corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Numerical Variables')
plt.xticks(rotation=45, ha='right')
plt.show()


# check if time of the day has an effect on the severity of the accident
plt.figure(figsize=(10, 6))
sns.scatterplot(data=accidentDf, x='AccidentHour', y='AccidentSeverityCategory_en')
plt.title('Accident Hour vs. Severity Category')
plt.xlabel('Hour of the Day')
plt.ylabel('Severity Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# Analyze continuous variables: Distribution of accidents by hour
plt.figure(figsize=(10, 6))
sns.histplot(accidentDf['AccidentHour'], kde=True, bins=24, color='blue')
plt.title('Distribution of Accidents by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()

# Relationship between two variables: Severity by accident type
plt.figure(figsize=(12, 8))
sns.boxplot(data=accidentDf, x='AccidentType_en', y='AccidentSeverityCategory_en', hue='AccidentType_en', palette='Set2', legend=False)
plt.title('Accident Severity by Type')
plt.xlabel('Accident Type')
plt.ylabel('Severity Category')
plt.xticks(rotation=45, ha='right')
plt.show()


#Accident locations visualization
'''# Scatter plot to see raw accident locations
plt.figure(figsize=(10, 8))
plt.scatter(
    accidentDf["AccidentLocation_CHLV95_E"],
    accidentDf["AccidentLocation_CHLV95_N"],
    alpha=0.5,
    s=1,
    c='blue'
)
plt.title("Accident Locations in Zurich (Scatter Plot)")
plt.xlabel("East Coordinate (CHLV95_E)")
plt.ylabel("North Coordinate (CHLV95_N)")
plt.grid(alpha=0.3)
plt.show()'''

#Scatter plot with map overlay
fig, ax = plt.subplots(figsize=(12, 10))

# base map first 
zurich_map.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.7)

# scatter plot on top
plt.scatter(
    accidentDf["AccidentLocation_CHLV95_E"],
    accidentDf["AccidentLocation_CHLV95_N"],
    alpha=0.5,
    s=1,
    c='blue',
    label='Accidents'
)
# limits to map bounds
minx, miny, maxx, maxy = zurich_map.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

plt.title("Accident Locations in Zurich (Scatter Plot)")
plt.xlabel("East Coordinate (CHLV95_E)")
plt.ylabel("North Coordinate (CHLV95_N)")
ax.grid(alpha=0.3)
plt.legend()
plt.show()

'''# Hexbin plot for accident density
plt.figure(figsize=(10, 8))
hb = plt.hexbin(
    accidentDf["AccidentLocation_CHLV95_E"],
    accidentDf["AccidentLocation_CHLV95_N"],
    gridsize=100,  # Adjust grid size for more/less detail
    cmap="YlOrRd",
    mincnt=1  # Minimum count per bin to display
)
plt.title("Accident Density in Zurich (Hexbin Plot)")
plt.xlabel("East Coordinate (CHLV95_E)")
plt.ylabel("North Coordinate (CHLV95_N)")
cb = plt.colorbar(hb)
cb.set_label("Number of Accidents")
plt.show()'''

# Hexbin plot with map overlay
fig, ax = plt.subplots(figsize=(12, 10))

# base map first
zurich_map.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.5)

# hexbin plot on top
hb = plt.hexbin(
    accidentDf["AccidentLocation_CHLV95_E"],
    accidentDf["AccidentLocation_CHLV95_N"],
    gridsize=100,
    cmap="YlOrRd",
    mincnt=1,
    alpha=0.8 
)

# limits to map bounds
minx, miny, maxx, maxy = zurich_map.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

plt.title("Accident Density in Zurich (Hexbin Plot)")
plt.xlabel("East Coordinate (CHLV95_E)")
plt.ylabel("North Coordinate (CHLV95_N)")

cb = plt.colorbar(hb)
cb.set_label("Number of Accidents")

plt.show()



#takes too long to run
'''# KDE (Kernel Density Estimate) plot for accident density
plt.figure(figsize=(10, 8))
sns.kdeplot(
    x=accidentDf["AccidentLocation_CHLV95_E"],
    y=accidentDf["AccidentLocation_CHLV95_N"],
    cmap="YlOrRd",
    fill=True,
    bw_adjust=0.5,  # Bandwidth adjustment for smoothing
    levels=50  # Number of density levels
)
plt.title("Accident Density in Zurich (KDE Plot)")
plt.xlabel("East Coordinate (CHLV95_E)")
plt.ylabel("North Coordinate (CHLV95_N)")
cb = plt.colorbar(hb)
cb.set_label("Number of Accidents")
plt.grid(alpha=0.3)
plt.show()'''



'''ML: Accident Severity Prediction using Random Forest Classifier'''
# Prepare features and target
X = accidentDf[['AccidentType', 'RoadType', 'AccidentHour', 'AccidentWeekDay']]
y = accidentDf['AccidentSeverityCategory_en']

# Encode categorical variables
X = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n\nAccident Severity Prediction:")
# First approach: Using class weights
print("\nApproach 1: Class Weights")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))
rf_model_weights = RandomForestClassifier(n_estimators=100, 
                                        class_weight=weight_dict,
                                        random_state=42)
rf_model_weights.fit(X_train, y_train)
y_pred_weights = rf_model_weights.predict(X_test)
print(classification_report(y_test, y_pred_weights, zero_division=1))

# Second approach: Using SMOTE
print("\nApproach 2: SMOTE")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_model_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote, zero_division=1))

# Third approach: Using SMOTETomek
#takes too long to run, i commented it out after running it once
'''print("\nApproach 3: SMOTETomek")
smt = SMOTETomek(random_state=42)
X_train_tomek, y_train_tomek = smt.fit_resample(X_train, y_train)
rf_model_tomek = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_tomek.fit(X_train_tomek, y_train_tomek)
y_pred_tomek = rf_model_tomek.predict(X_test)
print(classification_report(y_test, y_pred_tomek, zero_division=1))'''

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model_weights.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(feature_importance.head(5))

# Plot the feature importance for top 5 with percentages
plt.figure(figsize=(10, 6))
bars = plt.bar(feature_importance['feature'][:5], feature_importance['importance'][:5])

# Add percentage labels on top of each bar
total = feature_importance['importance'][:5].sum()
for bar in bars:
    height = bar.get_height()
    percentage = (height/total) * 100
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{percentage:.1f}%',
             ha='center', va='bottom')

plt.title('Top 5 Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#go back to preplexity, ther was something clustering related that i wanted to try

# Accident Hotspot Prediction
'''
Use clustering algorithms (e.g., DBSCAN) to identify accident hotspots.
Features: geographical coordinates
'''

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Prepare the data
coords = accidentDf[['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N']]
coords_scaled = StandardScaler().fit_transform(coords)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=5)
accidentDf['Cluster'] = dbscan.fit_predict(coords_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 10))
plt.scatter(coords['AccidentLocation_CHLV95_E'], coords['AccidentLocation_CHLV95_N'], c=accidentDf['Cluster'], cmap='viridis')
plt.title('Accident Hotspots')
plt.xlabel('East Coordinate (CHLV95_E)')
plt.ylabel('North Coordinate (CHLV95_N)')
plt.colorbar(label='Cluster')
plt.show()



''' Time Series Analysis: Predicting the Number of Accidents Over Time'''
# Prepare time series data
accidents_by_month = accidentDf.groupby([accidentDf['AccidentYear'].dt.year, 
                                        accidentDf['AccidentMonth']])['AccidentType'].count().reset_index()
accidents_by_month['YearMonth'] = accidents_by_month['AccidentYear'].astype(str) + '-' + accidents_by_month['AccidentMonth'].astype(str)
accidents_by_month['TimeIndex'] = range(len(accidents_by_month))

# Prepare features and target
X_time = accidents_by_month[['TimeIndex']].values
y_time = accidents_by_month['AccidentType'].values

# Split the data
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train_time, y_train_time)

# Make predictions
y_pred_time = lr_model.predict(X_test_time)

# Evaluate the model
print("\nTime Series Regression Results:")
print(f"R² Score: {r2_score(y_test_time, y_pred_time):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_time, y_pred_time)):.3f}")


'''Spatial Regression: Predicting the Time of Accidents Based on Location'''
# Prepare spatial features
X_spatial = accidentDf[['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N']].values
y_spatial = accidentDf['AccidentHour'].values  # Predicting time of accidents based on location

# Create polynomial features for non-linear relationships
poly = PolynomialFeatures(degree=2)
X_spatial_poly = poly.fit_transform(X_spatial)

# Split the data
X_train_spatial, X_test_spatial, y_train_spatial, y_test_spatial = train_test_split(
    X_spatial_poly, y_spatial, test_size=0.2, random_state=42
)

# Train the model
spatial_model = LinearRegression()
spatial_model.fit(X_train_spatial, y_train_spatial)

# Make predictions
y_pred_spatial = spatial_model.predict(X_test_spatial)

# Evaluate the model
print("\nSpatial Regression Results:")
print(f"R² Score: {r2_score(y_test_spatial, y_pred_spatial):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_spatial, y_pred_spatial)):.3f}")


'''Accident Count Prediction using Gradient Boosting Regressor'''
def prepare_regression_features(df):
    features = pd.DataFrame()
    features['Hour'] = df['AccidentHour']
    features['Month'] = df['AccidentMonth']
    features['WeekDay'] = pd.Categorical(df['AccidentWeekDay']).codes
    features['AccidentLocation_CHLV95_E'] = df['AccidentLocation_CHLV95_E']
    features['AccidentLocation_CHLV95_N'] = df['AccidentLocation_CHLV95_N']
    
    # One-hot encode categorical variables
    features = pd.concat([
        features,
        pd.get_dummies(df['AccidentType'], prefix='Type'),
        pd.get_dummies(df['RoadType'], prefix='Road')
    ], axis=1)
    
    return features

# Prepare features
X_reg = prepare_regression_features(accidentDf)
# Prepare target: number of accidents per location
y_reg = accidentDf.groupby(['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N']).size().reset_index(name='count')
# Merge X_reg and y_reg on location coordinates
merged_data = pd.merge(X_reg, y_reg, on=['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N'], how='left')

# Separate features and target
X_reg = merged_data.drop(['count', 'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N'], axis=1)
y_reg = merged_data['count'].fillna(0)

# Verify shapes match
print(f"X shape: {X_reg.shape}")
print(f"y shape: {y_reg.shape}")

# Split the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = gb_regressor.predict(X_test_reg)

# Evaluate the model
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\nAccident Count Prediction Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")


# Feature importance
'''feature_importance = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': gb_regressor.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features for Accident Count Prediction:")
print(feature_importance.head(5))'''
