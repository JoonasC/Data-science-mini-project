#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#data reading

import pandas as pd

## Read data
# Luetaan latin1-muodossa ja tallennetaan uutena UTF-8-tiedostona
with open('../data/Unemployment by occupation by month.csv', 'r', encoding='latin1') as f_in:
    content = f_in.read()

with open('../data/Unemployment by occupation by month_utf8.csv', 'w', encoding='utf-8') as f_out:
    f_out.write(content)

df_unemp_raw = pd.read_csv('../data/Unemployment by occupation by month_utf8.csv')


print("Alkuperäinen muoto (Wide Format):")
print(df_unemp_raw.shape)  # Esim. (esim. 120 riviä kuukausille, 400 saraketta ammateille)

# To long form
# 1. Tunnistetaan ensimmäiset tunnististesarakkeet (esim. 'Information' ja 'Month')
id_vars = ['Information', 'Month']

# Kaikki muut sarakkeet ovat ammattien otsikoita
value_vars = [col for col in df_unemp_raw.columns if col not in id_vars]

# 2. Muunnetaan data pitkäksi (Long Format)
df_unemp_long = pd.melt(
    df_unemp_raw,
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='Occupation_Raw',
    value_name='Unemployed_Count'
)

# 3. Puhdistetaan arvot (muutetaan luvut numeerisiksi, korvataan '...' puuttuvaksi arviksi)
df_unemp_long['Unemployed_Count'] = pd.to_numeric(df_unemp_long['Unemployed_Count'], errors='coerce')

# Poistetaan tyhjät rivit
df_unemp_long = df_unemp_long.dropna(subset=['Unemployed_Count'])

# 4. Luodaan ISCO-koodi (poimitaan 1. numero ryhmittelyä varten)
df_unemp_long['ISCO_1digit'] = df_unemp_long['Occupation_Raw'].str[0]
df_unemp_long.loc[df_unemp_long['Occupation_Raw'].str.startswith('X'), 'ISCO_1digit'] = 'X'

#make years from months
df_unemp_long['Year'] = df_unemp_long['Month'].str[:4].astype(int)

#Select wanted columns
df_unemp_long = df_unemp_long[['Year','ISCO_1digit','Unemployed_Count']]



print("\nMuunnettu data (Long Format):")
print(df_unemp_long.head())


###############################################################

#%%
#data handling


# 1. Luodaan sanakirja (Mapping), joka yhdistää ISCO-pääryhmän (1-digit) vastaavaan Toimialaan (NACE/TOL)
isco_to_nace_map = {
    '1': 'M_Professional_and_Management',
    '2': 'J_Information_and_Communication',
    '3': 'J_Information_and_Communication',
    '4': 'N_Administrative_and_Support',
    '5': 'I_Accommodation_and_Food',
    '6': 'A_Agriculture_Forestry',
    '7': 'F_Construction',
    '8': 'C_Manufacturing_and_Transport',
    '9': 'N_Administrative_and_Support',
    '0': 'Other_Special',
    'X': 'Other_Special'
}


# 3. Yhdistetään toimialaluokkaan ISCO-pääryhmän perusteella
df_unemp_long['Mapped_NACE_Industry'] = df_unemp_long['ISCO_1digit'].map(isco_to_nace_map)

# 4. Ryhmitellään (Aggregate) työttömyysluvut uuden toimialalinkityksen ja kuukauden mukaan
df_aggregated = df_unemp_long.groupby(['Year', 'Mapped_NACE_Industry', 'ISCO_1digit'])['Unemployed_Count'].mean().reset_index()
df_aggregated['Unemployed_Count'] = df_aggregated['Unemployed_Count'].round(1)
df_aggregated= df_aggregated.drop(columns=['Mapped_NACE_Industry'])


print("--- Ryhmitelty data, joka voidaan yhdistää AI-dataan ---")
print(df_aggregated.head(10))


# %%
###AI data

# Luetaan tekoälyn käyttödata toisesta tiedostosta

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ==========================================
# 1. AI-DATAN KÄSITTELY JA LUKU
# ==========================================
# Luetaan tekoälydata (korvaa 'ai_data.csv' omalla tiedostollasi, jossa erottimena pilkku tai puolipiste)
# pandas tulkitsee pisteet '.' puuttuviksi arvoiksi (NaN)
df_ai = pd.read_csv('../data/AI usage in enterprises by year.csv', na_values=['.'])

# Valitaan pääindikaattori: "The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises"
ai_col = [c for c in df_ai.columns if "uses Artificial Intelligence" in c][0]

# Suodatetaan pois tyhjät vuodet (kuten 2022)
df_ai_clean = df_ai.dropna(subset=[ai_col]).copy()
df_ai_clean[ai_col] = pd.to_numeric(df_ai_clean[ai_col])

# Lasketaan keskimääräinen AI-käyttöaste vuosittain toimialatasolla
print("--- TEKOÄLYN KÄYTTÖASTE VUOSITTAIN (%) ---")
print(df_ai_clean.groupby(['Year', 'Industry'])[ai_col].mean().unstack())

# 1. MÄÄRITETÄÄN SANAKIRJA (MAPPING): TOIMIALA (NACE) -> ISCO 1-DIGIT
# Kohdistetaan df_ai:n Industry-sarakkeen tekstiselitteet ISCO-pääryhmiin:

isco_mapping = {
    'J Information and communication (58-63)': {
        'ISCO_1digit': '2',
        'ISCO_Group': '2 - Erityisasiantuntijat (esim. Softakehittäjät, ICT-asiantuntijat)'
    },
    'M Professional, scientific and technical activities (69-75)': {
        'ISCO_1digit': '1',
        'ISCO_Group': '1 - Johtajat & Ylemmät asiantuntijat'
    },
    'L, N, S951 Real estate activities, administrative and support service activities and repair of computers and communication equipment': {
        'ISCO_1digit': '4',
        'ISCO_Group': '4 - Toimisto- ja asiakaspalvelutyöntekijät'
    },
    '41-43 Construction': {
        'ISCO_1digit': '7',
        'ISCO_Group': '7 - Rakennus-, korjaus- ja valmistustyöntekijät'
    },
    'H Transportation and storage (49-53)': {
        'ISCO_1digit': '8',
        'ISCO_Group': '8 - Prosessi- ja kuljetustyöntekijät'
    },
    'C-E Manufacturing; electricity, gas, steam and air conditioning and water supply; sewerage and waste management (10-39)': {
        'ISCO_1digit': '8',
        'ISCO_Group': '8 - Prosessi-, teollisuus- ja koneenkäyttäjät'
    },
    '55, 56 Accommodation and food service activities': {
        'ISCO_1digit': '5',
        'ISCO_Group': '5 - Palvelu- ja myyntityöntekijät'
    },
    '45-46 Wholesale and sale of motor vehicles': {
        'ISCO_1digit': '5',
        'ISCO_Group': '5 - Palvelu- ja myyntityöntekijät (Tukkukauppa)'
    },
    '47 Retail trade, except of motor vehicles and motorcycles': {
        'ISCO_1digit': '5',
        'ISCO_Group': '5 - Palvelu- ja myyntityöntekijät (Vähittäiskauppa)'
    },
    'Total': {
        'ISCO_1digit': 'TOTAL',
        'ISCO_Group': 'Kaikki ammattiryhmät yhteensä'
    }
}

# 2. LUODAAN MAPPING-DATAFRAME JA YHDISTETÄÄN (MERGE)
df_isco_info = pd.DataFrame.from_dict(isco_mapping, orient='index')

# Yhdistetään ISCO-tiedot df_ai-dataframeen 'Industry'-sarakkeen perusteella
df_ai_isco = df_ai.merge(df_isco_info, left_on='Industry', right_index=True, how='left')

df_ai_isco = df_ai_isco[['Year', 'ISCO_1digit','The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises' ]]
# Poistetaan rivit, joissa ISCO_1digit on 'TOTAL'
df_ai_isco = df_ai_isco[df_ai_isco['ISCO_1digit'] != 'TOTAL']


#%%
#ennen anovaa
df_final = pd.merge(
    df_aggregated, 
    df_ai_isco, 
    on=['Year', 'ISCO_1digit'], 
    how='inner'  # Pitää vain rivit, joille löytyy pari molemmista datoista
)

df_final = df_final.dropna()

#### Testausta
# 1. Tarkistetaan puuttuvat arvot (NaN / None)
missing_values = df_final.isnull().sum()
print("\n1. Puuttuvat arvot per sarake:")
print(missing_values)

# 2. Tarkistetaan tietotyypit (Data types)
print("\n2. Sarakkeiden tietotyypit:")
print(df_final.dtypes)

# 3. Tarkistetaan duplikaatit
print(f"Duplikaattirivejä: {df_final.duplicated().sum()}")

#%%
#Two-way-Anova

import statsmodels.api as sm
from statsmodels.formula.api import ols

# 1. Varmistetaan, että ryhmittelysarakkeet käsitellään luokkamuuttujina (kategorisina)
df_final['ISCO_1digit'] = df_final['ISCO_1digit'].astype(str)
df_final['Year'] = df_final['Year'].astype(str)

# 2. Määritellään additiivinen malli: Työttömyys ~ Ammattiryhmä + Vuosi
# C()-syntaksi varmistaa, että statsmodels ymmärtää muuttujat kategorisiksi
model = ols('Unemployed_Count ~ C(ISCO_1digit) + C(Year)', data=df_final).fit()

# 3. Lasketaan ANOVA-taulukko (Type II SS on yleisin standardi)
anova_table = sm.stats.anova_lm(model, typ=2)

print("=== TWO-WAY ANOVA TULOKSET ===")
print(anova_table)

# Nimetään tekoälysarake selkeyden vuoksi, jos siinä on pitkä nimi
df_final = df_final.rename(columns={df_final.columns[3]: 'AI_Rate'})

# Regressio: Selittääkö tekoälyn käyttöaste työttömyyttä, kun ammattiryhmä vakioidaan?
reg_model = ols('Unemployed_Count ~ AI_Rate + C(ISCO_1digit)', data=df_final).fit()

print("\n=== TEKOÄLYN VAIKUTUS (REGRESSIO / ANCOVA) ===")
print(reg_model.summary())

print(reg_model.summary().tables[1])


from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Suoritetaan Tukeyn post-hoc -testi ammattiryhmille
tukey = pairwise_tukeyhsd(
    endog=df_final['Unemployed_Count'], 
    groups=df_final['ISCO_1digit'], 
    alpha=0.05
)

print("=== TUKEY HSD POST-HOC VERTAILUT ===")
print(tukey)

###Two-way-anova ei ollukka paras, koitetaan toista

#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Two-Way Fixed Effects OLS / ANCOVA
# Control-muuttujina sekä ISCO-ryhmä että Vuosi
fe_model = ols('Unemployed_Count ~ AI_Rate + C(ISCO_1digit) + C(Year)', data=df_final).fit()

# Tulostetaan yhteenveto (erityisesti AI_Raten kerroin ja p-arvo)
print("=== TWO-WAY FIXED EFFECTS / ANCOVA MUDEL ===")
print(fe_model.summary().tables[1]) # Tulostaa kertoimet ja p-arvot

#%%
#####
#plotting

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Tallennetaan mallin ennusteet dataframeen
df_final['Predicted_Unemployment'] = fe_model.predict(df_final)

# 2. Luodaan kuvaaja
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Piirretään toteutuneet vs. ennustetut arvot ammattiryhmittäin
sns.scatterplot(
    data=df_final, 
    x='Unemployed_Count', 
    y='Predicted_Unemployment', 
    hue='ISCO_1digit', 
    style='ISCO_1digit',
    s=120, 
    palette='tab10'
)

# Lisätään 45-asteen täydellisen sovitteen viiva (y = x)
min_val = min(df_final['Unemployed_Count'].min(), df_final['Predicted_Unemployment'].min())
max_val = max(df_final['Unemployed_Count'].max(), df_final['Predicted_Unemployment'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5, label='Täydellinen sovite (y = x)')

# Otsikot ja akselit
plt.title('Two-Way Fixed Effects / ANCOVA: Toteutunut vs. Ennustettu työttömyys', fontsize=14, pad=15)
plt.xlabel('Toteutunut työttömyys (Unemployed_Count)', fontsize=12)
plt.ylabel('Mallin ennustama työttömyys (Predicted_Unemployment)', fontsize=12)
plt.legend(title='ISCO-ryhmä', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

##miten kuvaajaa luetaan
# ==============================================================================
# KUVAAN TULKINTA JA LUKUOHJE:
#
# 1. Pisteiden osuminen punaiselle katkoviivalle:
#    Mitä lähempänä pisteet ovat punaisesta viivasta (y = x), sitä paremmin
#    malli selittää aineiston työttömyysvaihtelun.
#
# 2. Ammattiryhmien erottuminen (Ryppäät / Clusterit):
#    Kuvaajasta näkyy selvästi, miten ammattiryhmät ryhmittyvät eri kohtiin
#    akselia (esim. ISCO 5 näkyy kaukana oikeassa yläkulmassa korkean
#    työttömyyden ryhmänä ja ISCO 1 vasemmassa alakulmassa).
#    Tämä havainnollistaa lukijalle R^2-arvon korkeaa tasoa visuaalisesti.
# ==============================================================================