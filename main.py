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
import pandasql as ps
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

from scipy import stats
from scipy.stats import chi2_contingency

# file path
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
# print the peak hour
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
peak_year = accidents_per_year.idxmax() # Highlight the peak year with a red star marker
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



'''Correlation Matrix'''
numeric_columns = accidentDf.select_dtypes(include=['number']).columns# Identify and exclude non-numeric columns
correlation_matrix = accidentDf[numeric_columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Numerical Variables')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=30, ha='right')
plt.show()

# check if time of the day has an effect on the severity of the accident
plt.figure(figsize=(10, 6))
sns.scatterplot(data=accidentDf, x='AccidentHour', y='AccidentSeverityCategory_en')
plt.title('Accident Hour vs. Severity Category')
plt.xlabel('Hour of the Day')
plt.ylabel('Severity Category')
plt.yticks(rotation=45, ha='right')
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
plt.yticks(rotation=30, ha='right')
plt.show()


'''Statistical Analysis - Chi-square Test for Independence'''
print("\nStatistical Analysis:")
print("-" * 30)

# Create numeric mapping for severity categories
severity_mapping = {category: index for index, category in 
                   enumerate(accidentDf['AccidentSeverityCategory_en'].unique())}

# Create new numeric column
accidentDf['SeverityNumeric'] = accidentDf['AccidentSeverityCategory_en'].map(severity_mapping)

# Update t-test code to use numeric values
severity_by_type = {accident_type: group['SeverityNumeric'].values 
                   for accident_type, group in accidentDf.groupby('AccidentType_en', observed=True)}

# Perform t-tests between pairs of accident types
t_test_results = {}
for type1 in severity_by_type:
    for type2 in severity_by_type:
        if type1 < type2:
            t_stat, p_value = stats.ttest_ind(severity_by_type[type1], 
                                            severity_by_type[type2],
                                            nan_policy='omit')
            t_test_results[(type1, type2)] = (t_stat, p_value)

# Create contingency table
contingency_table = pd.crosstab(accidentDf['AccidentType_en'], 
                               accidentDf['AccidentSeverityCategory_en'])

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square statistic: {chi2}")
print(f"p-value: {p_value}")
print(f"Degrees of freedom: {dof}")


'''ML: Accident Severity Prediction using Random Forest Classifier'''
# Prepare features and target
X = accidentDf[['AccidentType', 'RoadType', 'AccidentHour', 'AccidentWeekDay']]
y = accidentDf['AccidentSeverityCategory_en']
# Encode categorical variables
X = pd.get_dummies(X)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n\nAccident Severity Prediction:")
print("-" * 30)

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

# Feature importance analysis for Class Weights model
feature_importance_weights = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model_weights.feature_importances_
})
feature_importance_weights = feature_importance_weights.sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features (Class Weights Model):")
print(feature_importance_weights.head(5))

# Pie chart for Class Weights model
plt.figure(figsize=(10, 8))
top_5_features = feature_importance_weights.head(5)
other_importance = feature_importance_weights.iloc[5:]['importance'].sum()
sizes = list(top_5_features['importance']) + [other_importance]
labels = list(top_5_features['feature']) + ['Other']

colors = [plt.colormaps['tab10'](i / len(sizes)) for i in range(len(sizes))]
colors[-1] = 'lightgrey'  # Set 'Other' to light grey

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.suptitle('Feature Importance (Class Weights Model)', fontweight='bold')
plt.title('Top 5 + other Features')
plt.axis('equal')
plt.show()


# Second approach: Using SMOTE
print("\nApproach 2: SMOTE")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_model_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote, zero_division=1))

# Feature importance analysis for SMOTE model
feature_importance_smote = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model_smote.feature_importances_
})
feature_importance_smote = feature_importance_smote.sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features (SMOTE Model):")
print(feature_importance_smote.head(5))

# Pie chart for SMOTE model
plt.figure(figsize=(10, 8))
top_5_features_smote = feature_importance_smote.head(5)
other_importance_smote = feature_importance_smote.iloc[5:]['importance'].sum()
sizes_smote = list(top_5_features_smote['importance']) + [other_importance_smote]
labels_smote = list(top_5_features_smote['feature']) + ['Other']

colors_smote = [plt.colormaps['tab10'](i / len(sizes_smote)) for i in range(len(sizes_smote))]
colors_smote[-1] = 'lightgrey'  # Set 'Other' to light grey

plt.pie(sizes_smote, labels=labels_smote, autopct='%1.1f%%', startangle=90, colors=colors_smote)
plt.suptitle('Feature Importance (SMOTE Model)', fontweight='bold')
plt.title('Top 5 + other Features')
plt.axis('equal')
plt.show()


# Third approach: Using SMOTETomek- takes too long to run
# e kam komentu mbasi qe ka run ni here, eshte i perfshire ne raport
'''print("\nApproach 3: SMOTETomek")
smt = SMOTETomek(random_state=42)
X_train_tomek, y_train_tomek = smt.fit_resample(X_train, y_train)
rf_model_tomek = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_tomek.fit(X_train_tomek, y_train_tomek)
y_pred_tomek = rf_model_tomek.predict(X_test)
print(classification_report(y_test, y_pred_tomek, zero_division=1))
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
plt.show()'''

'''Accident locations visualizations '''
# i kam komentu (edhe Scatter edhe Hexbin) meqe nen to jane verzionet me te mira me harte
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

# Prepare data for regression
accident_counts = accidentDf.groupby(['AccidentYear', 'AccidentMonth', 'AccidentHour']).size().reset_index(name='accident_count')

# Create time-based features
accident_counts['month_sin'] = np.sin(2 * np.pi * accident_counts['AccidentMonth']/12)
accident_counts['month_cos'] = np.cos(2 * np.pi * accident_counts['AccidentMonth']/12)
accident_counts['hour_sin'] = np.sin(2 * np.pi * accident_counts['AccidentHour']/24)
accident_counts['hour_cos'] = np.cos(2 * np.pi * accident_counts['AccidentHour']/24)

# Prepare features
X = accident_counts[['month_sin', 'month_cos', 'hour_sin', 'hour_cos']]
y = accident_counts['accident_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# split data for training and testing

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple regression models
models = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

# Train models and store predictions
predictions = {}
results = {}

print("\nRegression Models Results:")
print("-" * 30)

for name, model in models.items():
    #train
    model.fit(X_train_scaled, y_train)
    
    #predict
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'rmse': rmse, 'r2': r2}
    print(f"\n{name} Results:")
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"R-squared Score: {r2:.4f}")

# scatter plots for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Actual vs Predicted Accidents - Model Comparison', fontsize=14)

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]
    
    ax.scatter(y_test, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Labels and title
    ax.set_xlabel('Actual Number of Accidents')
    ax.set_ylabel('Predicted Number of Accidents')
    ax.set_title(f'{name}\nRMSE: {results[name]["rmse"]:.2f}, R²: {results[name]["r2"]:.4f}')
    
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Residual plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Residual Plots - Model Comparison', fontsize=14)

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]
    
    residuals = y_test - y_pred # Calculate residuals
    
    # Scatter plot of predictions vs residuals
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Labels and title
    ax.set_xlabel('Predicted Number of Accidents')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{name} - Residual Plot')
    
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3 plots: the test and training learning curve, the training samples vs fit times curve, the fit times vs score curve.
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=5,
                       n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                      train_sizes=train_sizes,
                      return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                        fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability analysis")

    # fit_times vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance analysis")

    return plt


# learning curves for each model
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Learning Curves for Different Models', fontsize=16, y=1.05)

for idx, (name, model) in enumerate(models.items()):
    plot_learning_curve(
        model, #same models as before
        title=f"Learning Curve - {name}",
        X=X_train_scaled, 
        y=y_train,
        axes=axes[idx],
        ylim=(0, 1.1),
        cv=5,
        n_jobs=-1
    )

plt.tight_layout()
plt.show()

# Print cross-validation scores for each model
print("\nCross-validation Scores:")
print("-" * 30)

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"\n{name}:")
    print(f"Mean R² Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    print(f"Individual fold scores: {scores}")
    
