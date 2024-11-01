import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Define the path to the data folder
data_folder = 'data'

# Load COVID-19 data
covid_data_path = os.path.join(data_folder, 'covid19_Confirmed_dataset.csv')
covid_data = pd.read_csv(covid_data_path)
# print(covid_data.head())
# print(covid_data.shape)
# print(covid_data.columns)

# Drop unnecessary columns
covid_data.drop(columns=["Lat", "Long"], axis=1, inplace=True)
# print(covid_data.columns)

# Group data by country
grouped_data_by_country = covid_data.groupby("Country/Region").sum()
# print(grouped_data_by_country.head())
# print(grouped_data_by_country.shape)

# Plot data for Hungary and Austria
grouped_data_by_country.loc["Hungary"].plot()
grouped_data_by_country.loc["Austria"].plot()
plt.legend()
plt.show()

# print(grouped_data_by_country.loc["Hungary"].diff().max())  # 210
# print(grouped_data_by_country.loc["Austria"].diff().max())  # 1321

# Create a list of countries
country_list = list(grouped_data_by_country.index)
# Select numeric data
numeric_data = grouped_data_by_country.select_dtypes(include=[np.number])
# Get maximum infection rates
max_infection_rates = [numeric_data.loc[country].diff().max() for country in numeric_data.index]
grouped_data_by_country["max_infection_rate"] = max_infection_rates
# print(grouped_data_by_country.head())

# Create a DataFrame with maximum infection rates
max_infection_rate_df = pd.DataFrame(grouped_data_by_country["max_infection_rate"])
# print(max_infection_rate_df.head())

# Load happiness report
happiness_report_path = os.path.join(data_folder, 'worldwide_happiness_report.csv')
happiness_report = pd.read_csv(happiness_report_path)
# print(happiness_report.head())
# print(happiness_report.shape)

# Drop unnecessary columns from happiness report
unnecessary_columns_happiness = ["Overall rank", "Score", "Generosity", "Perceptions of corruption"]
happiness_report.drop(columns=unnecessary_columns_happiness, inplace=True)
# print(happiness_report.head())

# Set index for happiness report
happiness_report.set_index("Country or region", inplace=True)
# print(happiness_report.head())

# Load population data
population_data_path = os.path.join(data_folder, 'population.csv')
population_data = pd.read_csv(population_data_path)
population_2020 = population_data[population_data['Year'] == 2020].copy()
population_2020.reset_index(drop=True, inplace=True)

# Drop unnecessary columns from population data
unnecessary_columns_population = ["Country Code", "Year"]
population_2020.drop(columns=unnecessary_columns_population, inplace=True)
population_2020.set_index("Country Name", inplace=True)
# print(population_2020.head(20))

# Combine datasets
combined_data = max_infection_rate_df.join(happiness_report, how="inner")
combined_data = combined_data.join(population_2020, how="inner")

# Calculate maximum infection rate per capita
max_infection_rate_per_capita = combined_data["Value"] / combined_data["max_infection_rate"]
combined_data["max_infection_rate_per_capita"] = max_infection_rate_per_capita
# print(combined_data.head())
# print(combined_data.describe())

# Print correlation matrix
print(combined_data.corr(method='pearson'))

# Plot correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_data.corr(method='pearson'), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Create scatter and regression plots
y = combined_data["max_infection_rate_per_capita"]
x = combined_data["GDP per capita"]
sns.scatterplot(x=x, y=np.log(y))
plt.show()
sns.regplot(x=x, y=np.log(y))
plt.show()
