
# Getting started with Statsbomb Data

## 1) Get the Data

Go to https://github.com/statsbomb/open-data, click on the green 'Clone or download' button and choose download ZIP to your harddrive.
Unpack the zip.

### 2) Take the time to read their terms and conditions.
#### From their github page:
"Whilst we are keen to share data and facilitate research, we also urge you to be responsible with the data. Please register your details on https://www.statsbomb.com/resource-centre and read our User Agreement carefully."

### https://towardsdatascience.com/advanced-sports-visualization-with-pandas-matplotlib-and-seaborn-9c16df80a81b

is a great article to get started using StatsBomb Data and showcases Panda's json_normalize function.<br>
It provides us with a super easy way to take the nested json structure and put it into a nice and orderly DataFrame.

## 3) Process the data and combine it into DataFrames
This will create a DataFrame for all events, one for all Freeze Frames, and one with the information for all Matches

So far I relied on R (FC_rStats made a great series of posts here: https://github.com/FCrSTATS/StatsBomb_WomensData) <br> to process the data, now I can get all the information Statsbomb Data provides with Python, including all the Freeze Frame Information.


```python
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from os import listdir
from os.path import isfile, join

'''
Set mypath to your open-data-master/data/ path
'''
mypath = 


files = [f for f in listdir(mypath+'events/') if isfile(join(mypath+'events/', f))]
try: #if you're on MacOS like I am this file might mess with you, so try removing it
    files.remove('.DS_Store')
except:
    pass


dfs = {}
ffs = {}

for file in files:
    with open(mypath+'events/'+file) as data_file:
        #print (mypath+'events/'+file)
        data = json.load(data_file)
        #get the nested structure into a dataframe 
        df = json_normalize(data, sep = "_").assign(match_id = file[:-5])
        #store the dataframe in a dictionary with the match id as key (remove '.json' from string)
        dfs[file[:-5]] = df.set_index('id')    
        shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
        
        #get the freeze frame information for every shot in the df
        for id_, row in shots.iterrows():
            try:
                ff = json_normalize(row.shot_freeze_frame, sep = "_")
                ff = ff.assign(x = ff.apply(lambda x: x.location[0], axis = 1)).\
                        assign(y = ff.apply(lambda x: x.location[1], axis = 1)).\
                        drop('location', axis = 1).\
                        assign(id = id_)
                ffs[id_] = ff
            except:
                pass

#concatenate all the dictionaries
#this creates a multi-index with the dictionary key as first level
df = pd.concat(dfs, axis = 0)

#split locations into x and y components
df[['location_x', 'location_y']] = df['location'].apply(pd.Series)
df[['pass_end_location_x', 'pass_end_location_y']] = df['pass_end_location'].apply(pd.Series)

#split the shot_end_locations into x,y and z components (some don't include the z-part)
df['shot_end_location_x'], df['shot_end_location_y'], df['shot_end_location_z'] = np.nan, np.nan, np.nan
end_locations = np.vstack(df.loc[df.type_name == 'Shot'].shot_end_location.apply(lambda x: x if len(x) == 3
                                       else x + [np.nan]).values)
df.loc[df.type_name == 'Shot', 'shot_end_location_x'] = end_locations[:, 0]
df.loc[df.type_name == 'Shot', 'shot_end_location_y'] = end_locations[:, 1]
df.loc[df.type_name == 'Shot', 'shot_end_location_z'] = end_locations[:, 2]
df = df.drop(['location', 'pass_end_location', 'shot_end_location'], axis = 1)

#concatenate all the Freeze Frame dataframes
ff_df = pd.concat(ffs, axis = 0)

files = [f for f in listdir(mypath+'matches/') if isfile(join(mypath+'matches/', f))]
try:
    files.remove('.DS_Store')
except:
    pass

matches_dfs = {}
for file in files:
    with open(mypath+'matches/'+file) as data_file:
        #print (mypath+'lineups/'+file)
        data = json.load(data_file)
        #get the nested structure into a dataframe 
        df_ = json_normalize(data, sep = "_")
        #store the dataframe in a dictionary with the competition id as key
        matches_dfs[file[:-5]] = df_

matches_df = pd.concat(matches_dfs)
```

### Save the data
HDF Files provide an easy and fast way to store and read larger files.


```python
df.to_hdf(mypath+'Statsbomb_Data_df.hdf', key = 'df')
ff_df.to_hdf(mypath+'Statsbomb_Data_ff_df.hdf', key = 'ff_df')
matches_df.to_hdf(mypath+'Statsbomb_Data_matches_df.hdf', key = 'matches_df')
```

### Read the data


```python
df = pd.read_hdf(mypath+'Statsbomb_Data_df.hdf')
ff_df = pd.read_hdf(mypath+'Statsbomb_Data_ff_df.hdf')
matches_df = pd.read_hdf(mypath+'Statsbomb_Data_matches_df.hdf')
```


```python
df.head()
```


```python
ff_df.head()
```


```python
matches_df.head()
```
