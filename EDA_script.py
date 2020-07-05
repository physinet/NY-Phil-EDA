import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


''' Load dataset '''
url = 'https://raw.githubusercontent.com/nyphilarchive/PerformanceHistory/master/Programs/json/complete.json'

r = requests.get(url)
programs = r.json()['programs']

meta_cols = ['id', 'programID', 'orchestra', 'season']
df_concerts = pd.json_normalize(data=programs, record_path='concerts',
    meta=meta_cols)
df_works = pd.json_normalize(data=programs, record_path='works',
    meta=meta_cols)
df_soloists = pd.json_normalize(data=programs, record_path=['works', 'soloists'],
    meta=meta_cols)

df = pd.merge(df_works, df_concerts, 'outer', on=meta_cols)

df = df.drop(columns='soloists')
orch_types = ['New York Philharmonic', 'New York Symphony', 
            'Stadium-NY Philharmonic', 'Members of NY Philharmonic',
            'New/National Symphony Orchestra', 'Strike Orchestra (Philharmonic)']
df = df[df['orchestra'].isin(orch_types)]

event_types = df['eventType'].unique()
event_types = [event for event in event_types if 'Chamber' not in event]
other_small_events = ['Contact!', 'Holiday Brass', 'Insight Series', 
                      'Leinsdorf Lecture', 'Nightcap', 'Off the Grid',
                      'Pre-Concert Recital', 'Sound ON', 
                      'Tour - Very Young People\'s Concert',
                      'Very Young People\'s Concert']
event_types = [event for event in event_types if event not in other_small_events]
df = df[df['eventType'].isin(event_types)]

df = df[df.interval.isna()].drop(columns='interval')

cols = ['movement._', 'movement.em', 'workTitle._', 'workTitle.em']
df[df[cols].notna().any(axis=1)].sample(5, random_state=3)

for col in ['movement', 'workTitle']:
    rows = df[col].isna()
    df[col][rows] = df[col+'._'][rows] + ' ' +  df[col+'.em'][rows]
    df.drop(columns=[col+'._', col+'.em'], inplace=True)

df = df[df['conductorName'].notna()]


df['DateTime'] = pd.to_datetime(df['Date'].str.split('T').str[0] \
                                 + ' ' + df['Time'].str.replace('None', ''))
df.drop(columns=['Date', 'Time'], inplace=True)

df['work_id'] = df['ID'].str.split('*').str[0]
df['mvt_id'] = df['ID'].str.split('*').str[1]

grouped = df.groupby(['DateTime', 'work_id'])['mvt_id'].nunique()

df = df.drop(columns=['movement', 'mvt_id'])\
        .drop_duplicates(['DateTime', 'work_id'])

df = df.join(grouped, on=['DateTime', 'work_id'])

df = df.rename(columns={'id': 'program_guid', 'programID': 'program_id',
                       'mvt_id': 'num_movements'})

cols = ['program_guid', 'program_id', 'work_id', 'composerName', 'workTitle', 
       'num_movements', 'orchestra', 'conductorName', 'season', 'eventType', 'Location', 
        'Venue', 'DateTime']

df = df[cols]

# convert string ids to numeric
cols = ['program_id', 'work_id']
df[cols] = df[cols].apply(pd.to_numeric)

composer_counts = df['composerName'].value_counts()
df.insert(4, 'composer_popularity', np.log(composer_counts[df['composerName']]).values)

grouped = df.groupby(['program_id', 'DateTime'])['composer_popularity']
df.insert(5, 'safety', grouped.transform('mean'))

boroughs = ['Manhattan, NY', 'Brooklyn, NY', 'Queens, NY', 'Bronx, NY', 'Staten Island, NY']
df_nyc = df[df['Location'].isin(boroughs)]
df_tour = df.drop(df_nyc.index)

grouped_nyc = df_nyc.groupby(['program_id', 'DateTime'])
grouped_tour = df_tour.groupby(['program_id', 'DateTime'])

safety_nyc = grouped_nyc['composer_popularity'].mean()
safety_tour = grouped_tour['composer_popularity'].mean()

fig, ax = plt.subplots()
ax.hist(safety_nyc, bins=30, alpha=0.7, density=True, label='NYC')
ax.hist(safety_tour, bins=30, alpha=0.7, density=True, label='On tour')
ax.set_xlabel('Non-adventurousness of repertoire')
ax.set_ylabel('Proportion of concerts');
ax.legend(loc='best')
fig.savefig('Adventurousness_on_tour.pdf', bbox_inches='tight')

def n_composers(x, n):
    return x.nunique() == n

num_composers = np.arange(1, 5)

prop_with_num_nyc = np.zeros(5)
prop_with_num_tour = np.zeros(5)

for i, n in enumerate(num_composers):
    prop_with_num_nyc[i] = grouped_nyc['composerName'].transform(n_composers, n).sum() / len(df_nyc)
    prop_with_num_tour[i] = grouped_tour['composerName'].transform(n_composers, n).sum() / len(df_tour)
    
# "5+" 
prop_with_num_nyc[4] = 1 - prop_with_num_nyc.sum()
prop_with_num_tour[4] = 1 - prop_with_num_tour.sum()

df_one_composer = pd.DataFrame({'NYC': prop_with_num_nyc, 'On tour': prop_with_num_tour})
ax = df_one_composer.plot.bar(rot=0)
ax.set_xticklabels([1, 2, 3, 4, '5+'])
ax.set_xlabel('Number of composers per program')
ax.set_ylabel('Proportion of single-composer concerts')
ax.figure.savefig('Number_of_composers.pdf', bbox_inches='tight')