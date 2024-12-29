# Traffic Accident Analysis in Zurich

## Overview
This project explores traffic accidents in Zurich since 2011 using data from the Swiss Open Data Portal. Initially focused on visualizing accident trends, the analysis has expanded to include comprehensive data exploration, statistical analysis, and advanced machine learning techniques.

The project aims to:
1. Summarize and clean the data for effective analysis.
2. Perform exploratory data analysis (EDA) on accident trends and patterns.
3. Provide insights into relationships and correlations between key variables.
4. Implement and evaluate machine learning models for accident prediction.

## Data Source
[**Swiss Open Data Portal**](https://opendata.swiss/de/dataset/polizeilich-registrierte-verkehrsunfalle-auf-dem-stadtgebiet-zurich-seit-2011)

## Objectives
1. **Data Exploration**:
   - Load and summarize the dataset (columns, rows, data types).
   - Handle duplicates, missing values, and incorrect data types.
   - Compute summary statistics.
   - Integrate SQL queries within Python for advanced exploration.
2. **Data Analysis** (Non-relational):
   - Analyze categorical and continuous variables.
   - Generate statistical summaries.
   - Create visualizations for accident trends and patterns.
3. **Advanced Analysis**:
   - Investigate relationships between variables with graphical presentation.
   - Conduct correlation and regression analysis.
   - Perform statistical tests (e.g., Chi-square test for independence).
4. **Machine Learning**:
   - Implement regression models for accident prediction.
   - Evaluate and compare model performance.
   - Analyze learning curves and model scalability.

## Dependencies
- Python
- Matplotlib
- Pandas
- Seaborn
- Geopandas
- Pandasql
- NumPy
- Scikit-learn
- SciPy

## Project Structure
### Data Processing
- Load the dataset using Pandas.
- Change data types for efficient processing.
- Handle null values and duplicates.
- Explore and clean categorical and continuous variables.

### Visualizations
A variety of visualizations provide insights into traffic accident trends:
1. **Bar Plot of Accident Types**: Highlights the frequency of different accident types.
2. **Accidents by Day of the Week**: Displays accident frequency for each weekday.
3. **Accidents Over Time**: Tracks trends in accidents across years, highlighting peak years.
4. **Accident Types Over Time**: Examines the evolution of accident types over years.
5. **Severity Analysis**:
   - Stacked bar charts show accident severity by type and weekday.
   - Boxplots reveal severity distribution for different accident types.
7. **Heatmaps**: Illustrate correlations between accident types and days of the week.
8. **Distribution Analysis**: Histograms of accidents by hour.
9. **Geographical Visualization**: Scatter plots and hexbin plots of accident locations on Zurich map.

### Statistical Analysis
- Comprehensive statistical summaries.
- Correlation analysis using heatmaps.
- Chi-square test for independence between accident types and severity.

### Machine Learning Models
- Implementation of multiple regression models:
  - Linear Regression
  - Polynomial Regression
  - Gradient Boosting Regressor
- Model evaluation using metrics such as RMSE and R-squared.
- Visualization of model performance:
  - Actual vs Predicted scatter plots
  - Residual plots
  - Learning curves

## Instructions for Use
1. Clone this repository.
2. Ensure the required dependencies are installed:
   ```bash
   pip install matplotlib pandas seaborn geopandas pandasql numpy scikit-learn scipy
   ```
3. Execute the Python script to generate visualizations, insights, and machine learning models.

## Results
- Detailed insights into accident patterns and trends in Zurich.
- Identification of key years, times, and conditions associated with higher accident rates.
- Comparative analysis of different machine learning models for accident prediction.
- Evaluation of model performance and scalability.

## License

This project is licensed under the MIT License.