# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
######################################################################################
#Read data from file to df
df_ai = pd.read_csv('../data/AI usage in enterprises by year.csv')

#Select rows, containing 'Total' cell
df_annual_ai = df_ai[df_ai['Industry'] == 'Total']

#Select wanted columns
df_annual_ai = df_annual_ai[['Year','The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises']]

# Change column name
df_annual_ai = df_annual_ai.rename(columns={'The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises': 'Enterprises_%'})


# Replace empty cells with Nan
df_annual_ai['Enterprises_%'] = pd.to_numeric(df_annual_ai['Enterprises_%'].replace('.', np.nan), errors='coerce')

# Impute value in empty cell(s)
df_annual_ai['Enterprises_%'] = df_annual_ai['Enterprises_%'].interpolate(method='linear')



#%%
##############################################################

#Read data from file to df
df_unemployment = pd.read_csv('../data/General unemployment by month.csv', usecols=['Month','Unemployment rate, %'])

#Rename columns
df_unemployment = df_unemployment.rename(columns={'Unemployment rate, %': 'Unem_rate_%'})

# Remove missing values
df_unemployment = df_unemployment.dropna()

# 3. Takes Year from Month values (example "2009M01" -> 2009)
df_unemployment['Year'] = df_unemployment['Month'].str[:4].astype(int)

# 4. Drop Year 2026 out (this year), counts annual unemployment rate by mean of grouped months 
df_annual_unemployment = df_unemployment[df_unemployment['Year'] != 2026].groupby('Year')['Unem_rate_%'].mean().reset_index()

# 5. Rounding results
df_annual_unemployment['Unem_rate_%'] = df_annual_unemployment['Unem_rate_%'].round(1)

#%%
####################################
# Connecting dataframes

# Connects AI data (df_annual_ai) to Unemployment data (df_annual_unemployment) dataframe 
df_start_point = pd.merge(df_annual_unemployment, df_annual_ai, on='Year', how='left')

# Ensures right order of Years
df_start_point = df_start_point.sort_values('Year').reset_index(drop=True)

# Removes data before 2020 OUT
df_start_point = df_start_point[df_start_point['Year'] >= 2020].reset_index(drop=True)


#%%






#######################################################
# Dataplotting
##Only example

df_start_point.plot(x='Year', y=['Unem_rate_%', 'Enterprises_%'])

# Visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plotting two things to one
fig, ax1 = plt.subplots(figsize=(10, 5))


# Left Y-axel: Unemployment rate
color1 = '#1f77b4'
ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
ax1.set_ylabel('Unemployment rate (%)', color=color1, fontsize=11, fontweight='bold')
line1 = ax1.plot(df_start_point['Year'], df_start_point['Unem_rate_%'], color=color1, marker='o', linewidth=2.5, label='Työttömyysaste (%)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

# Left Y-axel betweem 0-10
ax1.set_ylim(bottom=0, top=10)

# Right Y-axel AI usage
ax2 = ax1.twinx()
color2 = '#ff7f0e'
ax2.set_ylabel('Enterprises AI usage (%)', color=color2, fontsize=11, fontweight='bold')
line2 = ax2.plot(df_start_point['Year'], df_start_point['Enterprises_%'], color=color2, marker='s', linestyle='--', linewidth=2.5, label='AI käyttö (%)')
ax2.tick_params(axis='y', labelcolor=color2)

# Right Y-axel starts from 0
ax2.set_ylim(bottom=0)

# Connects legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Title
plt.title('Työttömyysaste vs. Yritysten Tekoälykäyttö', fontsize=12, fontweight='bold', pad=12)

# Visual layout
plt.tight_layout()
plt.show()