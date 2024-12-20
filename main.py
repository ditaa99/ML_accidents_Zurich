"""
Traffic Accident Analysis in Zurich
- Data Source: Swiss Open Data Portal
- Objective: Analyze and visualize patterns in traffic accidents since 2011
- Author: Dita Pelaj
"""

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Specify the file path or URL
file_path = 'RoadTrafficAccidentLocations.csv'

# Read the CSV data
accidentDf = pd.read_csv(file_path)

# Print the DataFrame
# print(accidentDf)

"""check uniqueness and change datatypes for faster processing"""
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


"""checking for null values"""
# print(accidentDf.isnull().sum())


""" Bar Plot for Accident Types with Labels"""
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

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the frequency of different types of accidents with labels
bars = ax.bar(accident_type_freq.index, accident_type_freq.values)

# Add labels to the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 20, round(yval, 1), ha='center', va='bottom')

# Rotate the x-axis labels and make them fit into the figure
fig.autofmt_xdate()

# Set the title and labels
ax.set_title('Frequency of Different Types of Accidents')
ax.set_xlabel('Accident Type')
ax.set_ylabel('Frequency')

# plt.savefig("./plots/accident_types.png")
# Show the plot
plt.show()

"""Plot the frequency of accidents for each day of the week."""
# Define the order of the days of the week
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Get the frequency of accidents for each day of the week
accident_day_freq = accidentDf['AccidentWeekDay_en'].value_counts()

# Reindex the series to order the days of the week from Monday to Sunday
accident_day_freq = accident_day_freq.reindex(days_order)

# Print the frequency of accidents for each day of the week
print(accident_day_freq)

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the frequency of accidents for each day of the week
accident_day_freq.plot(kind='bar', ax=ax)

# Rotate the x-axis labels and make them fit into the figure
fig.autofmt_xdate()

# Set the title and labels
ax.set_title('Frequency of Accidents for Each Day of the Week')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Frequency')

# plt.savefig("./plots/frequency_accidents.png")
# Show the plot
plt.show()

# Group the data by 'AccidentWeekDay_en' and 'AccidentHour', and count the number of accidents in each group
accident_hour_day = accidentDf.groupby(['AccidentWeekDay_en', 'AccidentHour']).size()

# Reset the index of the DataFrame
accident_hour_day = accident_hour_day.reset_index()

# Rename the columns
accident_hour_day.columns = ['AccidentWeekDay_en', 'AccidentHour', 'Count']

# Find the peak hour of accidents for each day of the week
peak_hour_day = accident_hour_day.loc[accident_hour_day.groupby('AccidentWeekDay_en')['Count'].idxmax()]

# Print the peak hour of accidents for each day of the week
print(peak_hour_day)

"""Plot the number of accidents over time"""
# Convert the 'AccidentYear' column to datetime format
accidentDf['AccidentYear'] = pd.to_datetime(accidentDf['AccidentYear'], format='%Y')

# Group the data by 'AccidentYear' and count the number of accidents in each year
accidents_per_year = accidentDf.groupby(accidentDf['AccidentYear'].dt.year).size()

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the number of accidents over time using a line plot
ax.plot(accidents_per_year.index, accidents_per_year.values, marker='o', linestyle='-', color='b',
        label='Total Accidents')

# Highlight the peak year with a red star marker
peak_year = accidents_per_year.idxmax()
ax.plot(peak_year, accidents_per_year.max(), marker='*', markersize=10, color='r', label=f'Peak Year: {peak_year}')

# Set the title and labels
ax.set_title('Number of Accidents Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Accidents')

# Add legend to the plot
ax.legend()

# plt.savefig("./plots/accidents_over_time.png")
# Show the plot
plt.show()

"""I see there is a huge increase in 2014-2016 so i want to analyze it better (+-1year)"""
# Filter the data for the time period from 2014 to 2017
filtered_df = accidentDf[(accidentDf['AccidentYear'].dt.year >= 2013) & (accidentDf['AccidentYear'].dt.year <= 2017)]

# Group the filtered data by 'AccidentYear' and count the number of accidents in each year
accidents_per_year = filtered_df.groupby(filtered_df['AccidentYear'].dt.year).size()

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the number of accidents over time
ax.plot(accidents_per_year.index, accidents_per_year.values)

# Set the title and labels
ax.set_title('Number of Accidents Over Time (2013-2017)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Accidents')

# Show the plot
plt.show()

"""I notice that 2018 was the peak year of accidents, let's have a closer look into it"""
# Filter the data for the year 2018
accidents_2018 = accidentDf[accidentDf['AccidentYear'].dt.year == 2018]

# Group the data by 'AccidentMonth_en' and 'AccidentType_en' and count the number of accidents in each month for each type
# accidents_per_month_type_2018 = accidents_2018.groupby(['AccidentMonth_en', 'AccidentType_en']).size().unstack().reindex(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
accidents_per_month_type_2018 = accidents_2018.groupby(['AccidentMonth_en', 'AccidentType'],
                                                       observed=True).size().unstack().reindex(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
     'December'])

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(12, 8))

# Define a color map for different types of accidents
colors = plt.cm.get_cmap('tab10', len(accidents_per_month_type_2018.columns))
# colors = plt.cm.get_cmap('tab10', len(accidents_per_month_type_2018.columns))

# Plot the frequency of accidents for each month with different colors for each type
accidents_per_month_type_2018.plot(kind='bar', stacked=True, ax=ax,
                                   color=[colors(i) for i in range(len(accidents_per_month_type_2018.columns))])

# Rotate the x-axis labels and make them fit into the figure
fig.autofmt_xdate()

# Set the title and labels
ax.set_title('Monthly Distribution of Accidents in 2018 by Type')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Accidents')

# Add legend to the plot
ax.legend(title='Accident Type', bbox_to_anchor=(1, 1))

# plt.savefig("./plots/2018.png")
# Show the plot
plt.show()

# Identify and exclude non-numeric columns
numeric_columns = accidentDf.select_dtypes(include=['number']).columns
correlation_matrix = accidentDf[numeric_columns].corr()

# Print the correlation matrix
print(correlation_matrix)

"""Identify patterns in accident types using categorical data analysis"""
# Create a pivot table to analyze accident types over the years
accident_types_over_time = accidentDf.pivot_table(index='AccidentYear', columns='AccidentType_en', aggfunc='size',
                                                  fill_value=0)

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the trend of accident types over the years with thicker lines
accident_types_over_time.plot(kind='line', ax=ax, linewidth=2.5)

# Set the title and labels
ax.set_title('Accident Types Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Frequency')

# plt.savefig("./plots/patterns.png")
# Show the plot
plt.legend(title='Accident Type', loc='upper left')
plt.show()

"""Accident Types by Severity"""
# Create a pivot table to analyze accident types based on severity
accident_types_by_severity = accidentDf.pivot_table(index='AccidentType_en', columns='AccidentSeverityCategory_en',
                                                    aggfunc='size', fill_value=0)

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(12, 8))

# Plot a stacked bar chart for accident types by severity
accident_types_by_severity.plot(kind='bar', stacked=True, ax=ax)
# Rotate the x-axis labels and make them fit into the figure
fig.autofmt_xdate()

# Set the title and labels
ax.set_title('Accident Types by Severity')
ax.set_xlabel('Accident Type')
ax.set_ylabel('Frequency')

# plt.savefig("./plots/types_by_severity.png")

# Show the plot
plt.legend(title='Severity', loc='upper right', bbox_to_anchor=(1.12, 1.1))
plt.show()

"""severity of accidents on each day of the week"""
# Create a cross-tabulation of 'AccidentSeverityCategory' and 'AccidentWeekDay_en'
crosstab = pd.crosstab(accidentDf['AccidentWeekDay_en'], accidentDf['AccidentSeverityCategory_en'])

# Reorder the index to make the days of the week in order
crosstab = crosstab.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot a stacked bar chart for accident severity by day of the week
crosstab.plot(kind='bar', stacked=True, ax=ax)

# Rotate the x-axis labels and make them fit into the figure
fig.autofmt_xdate()

# Set the title and labels
ax.set_title('Severity of Accidents on Each Day of the Week')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Accidents')

# Move the legend outside of the chart and a few more pixels to the right
ax.legend(title='Severity', loc='upper right', bbox_to_anchor=(1.12, 1.1))

# plt.savefig("./plots/severity.png")
# Show the plot
plt.show()


"""Accident Types Heatmap"""
# Create a cross-tabulation for accident types
accident_types_cross_tab = pd.crosstab(accidentDf['AccidentWeekDay_en'], accidentDf['AccidentType_en'])

# Reorder the index to have weekdays in the correct order
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
accident_types_cross_tab = accident_types_cross_tab.reindex(weekday_order)

# Calculate the total number of accidents for each accident type
total_accidents_by_type = accident_types_cross_tab.sum(axis=0)

# Sort the accident types based on the total number of accidents in descending order
sorted_accident_types = total_accidents_by_type.sort_values(ascending=False).index
accident_types_cross_tab = accident_types_cross_tab[sorted_accident_types]

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(14, 10))

# Plot a heatmap for the correlation between accident types using seaborn
sns.heatmap(accident_types_cross_tab, cmap='Blues', annot=True, fmt='d', linewidths=0.5, ax=ax)

# Rotate the labels of accident types by 45 degrees
plt.xticks(rotation=45, ha='right')

# Rotate the labels of weekdays horizontally
plt.yticks(rotation=0)

# Set the title and labels
ax.set_title('Accident Types Heatmap')
ax.set_xlabel('Accident Type')
ax.set_ylabel('Day of the Week')

# plt.savefig("./plots/heatmap.png")
# Show the plot
plt.show()
