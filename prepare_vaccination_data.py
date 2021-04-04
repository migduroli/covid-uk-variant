from configparser import ConfigParser

config = ConfigParser()
config.read('configs/uk.ini')

import pandas as pd

cases = pd.read_csv('data/uk_data.csv')
data = pd.read_csv('data/raw_vaccination_data_uk.csv')

total_population_uk = config.getfloat('default', 'population')

df = (data
      .rename(columns={'newPeopleVaccinatedFirstDoseByPublishDate': 'n'})
      .groupby('date')['n']
      .sum() / total_population_uk) \
    .reset_index()

df.date = pd.to_datetime(df.date, format='%Y-%m-%d')

dates = pd.DataFrame({'date': pd.date_range(start=cases.date.min(), end=cases.date.max())})

df = dates.merge(df, how='left', on='date').fillna(0)

df.to_csv('data/vaccination_data_uk.csv', index=False)
