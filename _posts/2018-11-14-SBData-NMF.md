
## Finding Patterns in Statsbomb Data: Non-Negative Matrix Factorization Applications


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline
from FootiePy.drawpitch_ import drawpitch

import warnings
warnings.filterwarnings('ignore')
```

Let's first read in the data.

If you have not read my article on getting and processing the data https://github.com/znstrider/znstrider.github.io/blob/master/_posts/2018-11-11-Getting_Started_with_StatsBomb_Data.md, you can do so now.

Since first posting that article, I have updated it to include a lineups DataFrame that includes minutes played for every player and every match.<br>
We can use that later on to adjust actions for time played.


```python
# set mypath to your /open-data-master/data/ directory
mypath = 
df = pd.read_hdf(mypath+'Statsbomb_Data_df.hdf')
ff_df = pd.read_hdf(mypath+'Statsbomb_Data_ff_df.hdf')
matches_df = pd.read_hdf(mypath+'Statsbomb_Data_matches_df.hdf')
lineups_df = pd.read_hdf(mypath+'Statsbomb_Data_lineups_df.hdf')
```


```python
playtime = lineups_df.groupby(['player_name'])['minutes_played'].sum()
```

Let's filter the data for World Cup Matches


```python
wc_match_ids = matches_df.loc['43', 'match_id'].astype('str').values
wc_df = df.loc[df.match_id.isin(wc_match_ids), :]
```

Let's first familiarize ourselves with the dataset a little:

What's the shape of the dataset?<br>
What Columns do we have?<br>
What kind of event types are there?


```python
wc_df.shape
```




    (177350, 121)




```python
wc_df.columns[:100], wc_df.columns[100:]
```




    (Index(['50_50_outcome_id', '50_50_outcome_name', 'bad_behaviour_card_id',
            'bad_behaviour_card_name', 'ball_receipt_outcome_id',
            'ball_receipt_outcome_name', 'ball_recovery_offensive',
            'ball_recovery_recovery_failure', 'block_deflection', 'block_offensive',
            'block_save_block', 'clearance_aerial_won', 'dribble_nutmeg',
            'dribble_outcome_id', 'dribble_outcome_name', 'dribble_overrun',
            'duel_outcome_id', 'duel_outcome_name', 'duel_type_id',
            'duel_type_name', 'duration', 'foul_committed_advantage',
            'foul_committed_card_id', 'foul_committed_card_name',
            'foul_committed_offensive', 'foul_committed_penalty',
            'foul_committed_type_id', 'foul_committed_type_name',
            'foul_won_advantage', 'foul_won_defensive', 'foul_won_penalty',
            'goalkeeper_body_part_id', 'goalkeeper_body_part_name',
            'goalkeeper_outcome_id', 'goalkeeper_outcome_name',
            'goalkeeper_position_id', 'goalkeeper_position_name',
            'goalkeeper_technique_id', 'goalkeeper_technique_name',
            'goalkeeper_type_id', 'goalkeeper_type_name', 'index',
            'injury_stoppage_in_chain', 'interception_outcome_id',
            'interception_outcome_name', 'match_id', 'minute',
            'miscontrol_aerial_won', 'off_camera', 'pass_aerial_won', 'pass_angle',
            'pass_assisted_shot_id', 'pass_backheel', 'pass_body_part_id',
            'pass_body_part_name', 'pass_cross', 'pass_cut_back', 'pass_deflected',
            'pass_goal_assist', 'pass_height_id', 'pass_height_name', 'pass_length',
            'pass_miscommunication', 'pass_outcome_id', 'pass_outcome_name',
            'pass_recipient_id', 'pass_recipient_name', 'pass_shot_assist',
            'pass_switch', 'pass_through_ball', 'pass_type_id', 'pass_type_name',
            'period', 'play_pattern_id', 'play_pattern_name', 'player_id',
            'player_name', 'position_id', 'position_name', 'possession',
            'possession_team_id', 'possession_team_name', 'related_events',
            'second', 'shot_aerial_won', 'shot_body_part_id', 'shot_body_part_name',
            'shot_deflected', 'shot_first_time', 'shot_follows_dribble',
            'shot_freeze_frame', 'shot_key_pass_id', 'shot_one_on_one',
            'shot_open_goal', 'shot_outcome_id', 'shot_outcome_name',
            'shot_redirect', 'shot_statsbomb_xg', 'shot_technique_id',
            'shot_technique_name'],
           dtype='object'),
     Index(['shot_type_id', 'shot_type_name', 'substitution_outcome_id',
            'substitution_outcome_name', 'substitution_replacement_id',
            'substitution_replacement_name', 'tactics_formation', 'tactics_lineup',
            'team_id', 'team_name', 'timestamp', 'type_id', 'type_name',
            'under_pressure', 'location_x', 'location_y', 'pass_end_location_x',
            'pass_end_location_y', 'shot_end_location_x', 'shot_end_location_y',
            'shot_end_location_z'],
           dtype='object'))




```python
wc_df.type_name.value_counts()
```




    Pass                 62866
    Ball Receipt*        58964
    Pressure             23463
    Ball Recovery         5676
    Duel                  3141
    Block                 2162
    Dribble               2109
    Clearance             2074
    Goal Keeper           1975
    Foul Committed        1876
    Foul Won              1789
    Camera On             1756
    Shot                  1706
    Miscontrol            1535
    Dribbled Past         1441
    Interception          1276
    Dispossessed          1189
    Camera off             393
    Substitution           382
    Injury Stoppage        291
    Half End               284
    Half Start             284
    Starting XI            128
    Tactical Shift         116
    50/50                  108
    Player On               63
    Player Off              63
    Referee Ball-Drop       57
    Shield                  56
    Bad Behaviour           40
    Error                   37
    Offside                 26
    Own Goal Against        12
    Own Goal For            12
    Name: type_name, dtype: int64




```python
df.clearance_aerial_won.value_counts()
```




    True    236
    Name: clearance_aerial_won, dtype: int64



Now we can filter te dataset for what we need.

First I want to look at defensive actions.<br>
That means Pressure events, Ball Recoveries, Blocks, Fouls, Tackles, Interceptions and Aerials.<br>
For Aerials I am only looking at clearances, because it seems 'miscontrol_aerial_won' and 'pass_aerial_won' can both be defensive or offensive actions.


```python
# We can create the Tackle events from the duel_type_name
# I am assuming that like with Opta Data, all Tackles are successful events - correct me if I'm wrong.
wc_df.loc[:, 'Tackle'] = wc_df.duel_type_name.apply(lambda x: True if x == 'Tackle' else np.nan)
```

## Dimensionality Reduction: Non-Negative Matrix Factorization (NMF)

The problem I want to tackle is as follows:<br>
How does one make sense of actions across the pitch? How can we gether patterns hidden in the data?<br>
That's what <i>Dimensionality Reduction Techniques</i>, specifically <i>Non Negative Matrix Factorization</i> allows us to do.

The first thing we can do is bin the data across both the x- and y-axes.
I chose to divide the pitch into 5x5 meter bins. Other dimensions might work better, you can try this out yourself if you'd like.

To keep the example simpler, let's assume I were to divide the pitch into 10 bins across both axes. That would give us a 10x10 grid, n = 100 bins in total.<br>

For every player in the dataset, we can count the number of event occurences within each bin, and put those 100 values into a row vector that keeps the score for a given player.

Stacking all m, say 500, player row-vectors gives us a 500 x 100 Matrix.

To this matrix we can apply the NMF Algorithm to obtain:

1) <b>The Model Components</b>: N vectors within the 100 Dimensional Space (N = the number of model components we wish to obtain)<br>
We have reduced the 100 dimensions across the columns to only N columns.

2) <b>The weights</b> for each of those vectors, for each of the 500 players.<br>
For every player we get a value for each of the N model components, that signals how important a component is for a specific player.

The benefit of this analysis is that this gives us patterns that are inherit in the data instead of just looking at arbitrarily defined bins. As we will see below, this captures some positional patterns really well.<br>
Looking at the model weights for players allows us to categorize and rank players across those learned dimensions.

This is similar to other dimensionality reduction techniques like SVD or PCA, but as the name suggests, we impose the constraint, that our resulting Model Components and Weights need to be non-negative.<br>
This is necessary, as negative values do not make any sense in light of the count data we are working with.

I deliberately tried to keep this explanation simple. Justin Jacobs (@squared2020) has a really good blog post up at https://squared2020.com/2018/10/04/understanding-trends-in-the-nba-how-nnmf-works/ that goes into a little more detail.<br>
As I am not nearly as statistically savvy, I'll leave it to the reader to further dig into the approach of Dimensionality Reduction and NMF.

Now let's dig into the actual analysis:


```python
adjusted_for_time_played = False #set to true if you want to adjust for time played

defensive_actions = wc_df.loc[df.type_name.isin(
        (['Pressure', 'Ball Recovery', 'Block', 'Foul Committed', 'Interception']))|
                              (pd.notnull(wc_df.loc[:, 'clearance_aerial_won']))|
                              (pd.notnull(wc_df.loc[:, 'Tackle'])),
        ['player_name', 'location_x', 'location_y', 'type_name']].\
        reset_index().rename(columns = {'level_0': 'match_id'}).\
        set_index(['player_name','match_id', 'id']).sort_index()

if not adjusted_for_time_played:
    binned_values = []

    indices = defensive_actions.index.levels[0]
    for player in indices:
        #bin 2-dimensionally
        binned, bins_x, bins_y = np.histogram2d(defensive_actions.loc[player, 'location_x'],
                                     defensive_actions.loc[player, 'location_y'],
                                     bins = ([np.arange(0,125,5),
                                             np.arange(0,85,5)]))
        binned_values.append(binned.ravel())

else :
    mask = playtime[playtime >= 150].index
    defensive_actions = defensive_actions.reset_index().\
    loc[defensive_actions.reset_index()['player_name'].isin(mask), :].\
    set_index('player_name')
    
    binned_values = []

    #indices = defensive_actions.index.levels[0]
    indices = defensive_actions.index.unique()
    for player in indices:
        binned, bins_x, bins_y = np.histogram2d(defensive_actions.loc[player, 'location_x'],
                                     defensive_actions.loc[player, 'location_y'],
                                     bins = ([np.arange(0,125,5),
                                             np.arange(0,85,5)]))
        binned_values.append(binned.ravel()
                             / (playtime.loc[player] / 90) #adjusting for time played
                               )
```

I also tried adjusting for minutes played, but unfortunately the small sample size for most of the players made the results come out rather badly.

With filtering out players with < 150 min that worked much better, but as a lot of unknowns show up there, I chose to stick with the unfiltered dataset.<br>
Feel free to download the notebook to uncomment and you can also change the N of model components below.

To get the most out of this kind of analysis, I assume possession adjusting the counts as well as normalizing based on minutes played is the way to go. With the small dataset at hand I'd rather show some sensible patterns and players though.

As you will see, the model components do capture some patterns really nicely as is, but I assume the numbers to not be very meaningful as players with most games will register the most actions and thus will show the biggest influence.

Keep that in mind.


```python
from sklearn.decomposition import NMF
N = 12
import string
cols = [s for s in string.ascii_uppercase[:N]]

bin_freqs = np.vstack(binned_values)
model = NMF(n_components=N, init='random', random_state=0)
W = model.fit_transform(bin_freqs)
H = model.components_

model_components = pd.DataFrame(W,
             index = indices,
             columns = cols)
```


```python
print('Shape of the Weights Matrix W: ', W.shape,
      '\nShape of the Model Components Matrix H: )', H.shape)
```

    Shape of the Weights Matrix W:  (600, 12) 
    Shape of the Model Components Matrix H: ) (12, 384)



```python
fig, ax = plt.subplots(int(H.shape[0]/4), 4, figsize = (16,16))
ax = ax.flatten()

top_rated = model_components.idxmax().values

X, Y = np.meshgrid(bins_x, bins_y)
for i in range(H.shape[0]):
    drawpitch(ax = ax[i], measure = 'SB', orientation = 'vertical')
    plt.sca(ax[i])
    plt.pcolor(Y.T,
               X.T,
               H[i, :].reshape(-1, len(bins_y)-1))
    plt.title('Model Component '+cols[i] + '\nTop Rated: \n' + top_rated[i], fontsize = 14)  
```


![png](/images/2018-11-14-SBData-NMF_files/output_20.png)


As you can see, this does capture some positional patterns really well:<br>
Defensive Midfielder<br>
Right Midfielder<br>
Right Fullback<br>
Left Defensive Midfielder<br>
Right Defensive Midfielder<br>
Left Back<br>
Left Midfielder<br>
Right Back<br>
Center Forward<br>
(another) Right Back<br>
Goalkeeper<br>
Right Center Midfield

It also shows a problem that might arise:<br>
The approach might not always be able to differentiate between very similar positional components.

#### As comparison, below you will see the defensive activity heatmaps for the top rated players of each component.


```python
fig, ax = plt.subplots(int(H.shape[0]/4), 4, figsize = (16,16))
ax = ax.ravel()

top_rated = model_components.idxmax().values

for i, player in enumerate(top_rated):
    plt.sca(ax[i])
    binned, bins_x, bins_y = np.histogram2d(defensive_actions.loc[player, 'location_x'],
                                 defensive_actions.loc[player, 'location_y'],
                                 bins = ([np.arange(0,125,5),
                                         np.arange(0,85,5)]))
    X, Y = np.meshgrid(bins_x, bins_y)
    drawpitch(ax = ax[i], measure = 'SB', orientation = 'vertical')
    plt.pcolor(Y.T,
               X.T,
               binned)
    plt.title('Heatmap for ' + top_rated[i], fontsize = 14)  
```


![png](/images/2018-11-14-SBData-NMF_files/output_23.png)


Let's gather some more information on how values are distributed within the components:


```python
model_components.describe().round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>J</th>
      <th>K</th>
      <th>L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
      <td>600.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.206</td>
      <td>0.239</td>
      <td>0.133</td>
      <td>0.149</td>
      <td>0.137</td>
      <td>0.111</td>
      <td>0.142</td>
      <td>0.072</td>
      <td>0.170</td>
      <td>0.084</td>
      <td>0.088</td>
      <td>0.144</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.717</td>
      <td>0.588</td>
      <td>0.341</td>
      <td>0.270</td>
      <td>0.279</td>
      <td>0.268</td>
      <td>0.279</td>
      <td>0.170</td>
      <td>0.330</td>
      <td>0.189</td>
      <td>0.171</td>
      <td>0.258</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.028</td>
      <td>0.005</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>0.016</td>
      <td>0.000</td>
      <td>0.007</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.198</td>
      <td>0.214</td>
      <td>0.122</td>
      <td>0.170</td>
      <td>0.156</td>
      <td>0.090</td>
      <td>0.149</td>
      <td>0.057</td>
      <td>0.196</td>
      <td>0.077</td>
      <td>0.102</td>
      <td>0.200</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.025</td>
      <td>5.755</td>
      <td>5.110</td>
      <td>2.159</td>
      <td>2.627</td>
      <td>3.170</td>
      <td>2.605</td>
      <td>1.687</td>
      <td>2.722</td>
      <td>1.819</td>
      <td>1.445</td>
      <td>2.287</td>
    </tr>
  </tbody>
</table>
</div>



Values at the 50th percentile range between 0 and 0.03,
at the 75th between 0.05 to 0.22,
while maximum values range from about 1.5 to over 14.

Just something to keep in mind when looking at the values.

Now, what this allows us to do is, look at which players show the highest values for a model component we are interested in.<br>
Who for example is most similar in defensive activity in the component lead by N'Golo Kanté?


```python
model_components.A.sort_values(ascending = False)[:20]
```




    player_name
    N'Golo Kanté                     14.025341
    Vahid Amiri                       5.078418
    Roman Zobnin                      4.254000
    Denis Cheryshev                   2.758362
    Paul Labile Pogba                 2.544006
    Rodrigo Bentancur Colmán          2.135550
    Ola Toivonen                      2.084988
    Onyinye Ndidi                     2.045143
    Yassine Meriah                    1.923451
    Antoine Griezmann                 1.857233
    Jordan Henderson                  1.785737
    Ellyes Skhiri                     1.778376
    Blaise Matuidi                    1.728032
    Jacek Góralski                    1.700957
    James David Rodríguez Rubio       1.592296
    Aron Einar Gunnarsson             1.283240
    Aníbal Cesis Godoy                1.101542
    Emil Forsberg                     1.098896
    Mousa Dembélé                     1.095833
    Diego Sebastián Laxalt Suárez     1.017982
    Name: A, dtype: float64



Who are the strikers that put in defensive activity all across the opposing side of the pitch as well as into their own half like Mario Mandžukić?


```python
model_components.I.sort_values(ascending = False)[:20]
```




    player_name
    Mario Mandžukić                   2.722239
    Harry Kane                        2.109760
    Falcao                            2.075953
    Ivan Rakitić                      1.940288
    Gabriel Jesus                     1.920490
    Antoine Griezmann                 1.827962
    Marcus Berg                       1.783312
    Ola Toivonen                      1.690852
    Artem Dzyuba                      1.628784
    Kevin De Bruyne                   1.561835
    Yūya Ōsako                        1.258692
    Luis Alberto Suárez Díaz          1.201976
    Wahbi Khazri                      1.188456
    Javier Alejandro Mascherano       1.118328
    Diego da Silva Costa              1.079861
    Olivier Giroud                    1.066128
    Alfreð Finnbogason                1.046388
    Lionel Andrés Messi Cuccittini    1.009059
    Aleksandar Mitrovic               0.940295
    Edinson Cavani                    0.925394
    Name: I, dtype: float64



And we can look at specific players we are interested in.


```python
model_components.loc[top_rated[0], :]
```




    A    14.025341
    B     0.298928
    C     0.518126
    D     0.000000
    E     0.797850
    F     0.089665
    G     0.000000
    H     0.124631
    I     0.000000
    J     0.000000
    K     0.032220
    L     0.203027
    Name: N'Golo Kanté, dtype: float64



In addition to killing it in component A, N'Golo Kanté has decent values in components B,C,E, L and H - all towards the right side, mostly in one's own half.

We can also look at who scores best across all components. This might be useful. Who knows ¯|_(ツ)_/¯


```python
model_components.sum(axis=1).sort_values(ascending=False)[:20]
```




    player_name
    N'Golo Kanté                    16.089788
    Viktor Claesson                  9.281645
    Roman Zobnin                     8.431647
    Kieran Trippier                  7.342967
    Mário Figueira Fernandes         7.176085
    Paul Labile Pogba                6.876051
    Vahid Amiri                      6.824149
    Thomas Meunier                   6.435447
    Aleksandr Samedov                6.298756
    Nahitan Michel Nández Acosta     6.150166
    Jesse Lingard                    5.794497
    Jordan Henderson                 5.695838
    Kevin De Bruyne                  5.493141
    Antoine Griezmann                5.452880
    Rodrigo Bentancur Colmán         5.443671
    Bernardo Silva                   5.420548
    Blaise Matuidi                   5.369748
    Kylian Mbappé                    5.320653
    Onyinye Ndidi                    5.318237
    Fágner Conserva Lemos            5.015290
    dtype: float64



## Does this also help us to better categorize positioning on offensive actions?


```python
offensive_actions = wc_df.loc[df.type_name.isin(
        ['Shot', 'Pass', 'Ball Receipt', 'Dribble', 'Foul Won']),
        ['player_name', 'location_x', 'location_y', 'type_name']].\
        reset_index().rename(columns = {'level_0':'match_id'}).\
        set_index(['player_name', 'match_id', 'id']).sort_index()

binned_values = []
for player in offensive_actions.index.levels[0]:

    binned, bins_x, bins_y = np.histogram2d(offensive_actions.loc[player, 'location_x'],
                                 offensive_actions.loc[player, 'location_y'],
                                 bins = ([np.arange(0,125,5),
                                         np.arange(0,85,5)]))
    binned_values.append(binned.ravel() 
                        # / (playtime.loc[player] / 90)
                        )
    
from sklearn.decomposition import NMF
N = 20
import string
cols = [s for s in string.ascii_uppercase[:N]]

bin_freqs = np.vstack(binned_values)
model = NMF(n_components=N, init='random', random_state=0)
W = model.fit_transform(bin_freqs)
H = model.components_

model_components = pd.DataFrame(W,
             index = offensive_actions.index.levels[0],
             columns = cols)

fig, ax = plt.subplots(int(H.shape[0]/4), 4, figsize = (18,40))
ax = ax.flatten()

top_rated = model_components.idxmax().values

X, Y = np.meshgrid(bins_x, bins_y)
for i in range(H.shape[0]):
    drawpitch(ax = ax[i], measure = 'SB', orientation = 'vertical')
    plt.sca(ax[i])
    plt.pcolor(Y.T,
               X.T,
               H[i, :].reshape(-1, len(bins_y)-1))
    plt.title('Model Component '+cols[i] + '\nTop Rated: ' + top_rated[i], fontsize = 14)  
```


![png](/images/2018-11-14-SBData-NMF_files/output_37.png)


I really like how the model is able to capture a lot of positional patterns here. <br>
Especially the corner and set piece takers surprised me a bit.

## Shots


```python
offensive_actions = wc_df.loc[df.type_name.isin(
        ['Shot']),
        ['player_name', 'location_x', 'location_y', 'type_name']].\
        reset_index().rename(columns = {'level_0': 'match_id'}).\
        set_index(['player_name', 'match_id', 'id']).sort_index()

binned_values = []
for player in offensive_actions.index.levels[0]:

    binned, bins_x, bins_y = np.histogram2d(offensive_actions.loc[player, 'location_x'],
                                 offensive_actions.loc[player, 'location_y'],
                                 bins = ([np.arange(0,125,5),
                                         np.arange(0,85,5)]))
    binned_values.append(binned.ravel() 
                        # / (playtime.loc[player] / 90)
                        )
    
from sklearn.decomposition import NMF
N = 5
import string
cols = [s for s in string.ascii_uppercase[:N]]

bin_freqs = np.vstack(binned_values)
model = NMF(n_components=N, init='random', random_state=0)
W = model.fit_transform(bin_freqs)
H = model.components_

model_components = pd.DataFrame(W,
             index = offensive_actions.index.levels[0],
             columns = cols)

fig, ax = plt.subplots(int(H.shape[0]/N), N, figsize = (18,12))
ax = ax.flatten()

top_rated = model_components.idxmax().values

X, Y = np.meshgrid(bins_x, bins_y)
for i in range(H.shape[0]):
    drawpitch(ax = ax[i], measure = 'SB', orientation = 'vertical')
    plt.sca(ax[i])
    plt.pcolor(Y.T,
               X.T,
               H[i, :].reshape(-1, len(bins_y)-1))
    plt.title('Model Component '+cols[i] + '\nTop Rated: ' + top_rated[i], fontsize = 14) 
```


![png](/images/2018-11-14-SBData-NMF_files/output_40.png)


## Key Passes


```python
offensive_actions = wc_df.reset_index().loc[wc_df.reset_index().id.isin(wc_df.shot_key_pass_id.dropna().values),
        ['player_name', 'location_x', 'location_y', 'type_name', 'level_0', 'id']].\
        reset_index().rename(columns = {'level_0': 'match_id'}).\
        set_index(['player_name', 'match_id', 'id']).sort_index()

binned_values = []
for player in offensive_actions.index.levels[0]:

    binned, bins_x, bins_y = np.histogram2d(offensive_actions.loc[player, 'location_x'],
                                 offensive_actions.loc[player, 'location_y'],
                                 bins = ([np.arange(0,125,5),
                                         np.arange(0,85,5)]))
    binned_values.append(binned.ravel() 
                        # / (playtime.loc[player] / 90)
                        )
    
from sklearn.decomposition import NMF
N = 10
import string
cols = [s for s in string.ascii_uppercase[:N]]

bin_freqs = np.vstack(binned_values)
model = NMF(n_components=N, init='random', random_state=0)
W = model.fit_transform(bin_freqs)
H = model.components_

model_components = pd.DataFrame(W,
             index = offensive_actions.index.levels[0],
             columns = cols)

fig, ax = plt.subplots(int(H.shape[0]/5), 5, figsize = (18,9))
ax = ax.flatten()

top_rated = model_components.idxmax().values

X, Y = np.meshgrid(bins_x, bins_y)
for i in range(H.shape[0]):
    drawpitch(ax = ax[i], measure = 'SB', orientation = 'vertical')
    plt.sca(ax[i])
    plt.pcolor(Y.T,
               X.T,
               H[i, :].reshape(-1, len(bins_y)-1))
    plt.title('Model Component '+cols[i] + '\nTop Rated: ' + top_rated[i], fontsize = 14) 
```


![png](/images/2018-11-14-SBData-NMF_files/output_42.png)


In general more data is better, but overall I am quite pleased of the structures all of the analyses bring to light.

What would you be interested to look at?
