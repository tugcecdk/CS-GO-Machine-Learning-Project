

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=8)

plt.close("all")

# read datasets 
economy = pd.read_csv("economy.csv", sep=",")
#economy = economy.head(1000)
columns=["best_of", "t1_start", "t2_start"]

# drop round winners
for i in range(1, 31):
    columns.append(str(i)+"_winner")
    
# drop economy cols
for i in range(1, 31):
    columns.append(str(i)+"_t1")
for i in range(1, 31):
    columns.append(str(i)+"_t2")

economy.drop(columns=columns, inplace=True, axis=1)

results = pd.read_csv("results.csv", sep=",")

results.drop(columns=["date",
                      "team_1",
                      "team_2",
                      "result_1",
                      "result_2",
                      "event_id",
                      "map_wins_1",
                      "map_wins_2",
                      "match_winner",
                      "t_1",
                      "t_2",
                      "ct_1",
                      "ct_2"], inplace=True)

# create unique id for each game and map
id_col = []
for match_id, _map in zip(economy["match_id"], economy["_map"]):
    id_col.append(str(match_id) + _map)
economy["id"] = id_col

id_col = []
for match_id, _map in zip(results["match_id"], results["_map"]):
    id_col.append(str(match_id) + _map)
results["id"] = id_col

# drop the columns below to avoid duplications
results.drop("_map", inplace=True, axis=1)

economy.drop("match_id", inplace=True, axis=1)
results.drop("match_id", inplace=True, axis=1)

# merge economy and results datasets
data = economy.merge(results, on="id")

# remove default maps which is undefined (there are approximately 10-20 of them)
data = data[data['_map'] != "Default"]

# convert winning team from 1 and 2 to 0 and 1
data["map_winner"] = data["map_winner"] - 1
data["starting_ct"] = data["starting_ct"] - 1

# fill na values with 0
data.fillna(value=.0, inplace=True)

# move label column as the last column
y = data["map_winner"]
data.drop("map_winner", inplace=True, axis=1)
data = pd.concat([data, y], axis=1)

data.drop("id", inplace=True, axis=1)

# include player powers
players = pd.read_csv("players.csv", sep=",")

# player-team data
player_team = players[["player_name",
                       "team",
                       "date"]].copy()

# drop dublicate rows to make player stats unique
player_team = player_team.drop_duplicates(subset =["player_name", "team", "date"], keep = "first").copy()


# select necessary columns
players = players[["date",
                   "player_name",
                  "assists",
                  "hs",
                  "kast",
                  "kddiff",
                  "adr",
                  "kddiff_ct",
                  "adr_ct",
                  "kast_ct",
                  "kddiff_t",
                  "adr_t",
                  "kast_t"]]


# fill na values with
players = players.fillna(players.min())

#players_avg_stats = players.groupby(['player_name']).mean()


# rescale columns
"""
assist  min 0   max 27   *100/max
hs 	   min 0   max 51   *100/max
kast   min 16  max 100  -min * (100/max)
kddiff min -37 max 27   +min * (100/max)
adr    min 2   max 130  -min * (100/max)
"""

# scale values between 0 and 100
players["assists"] = players["assists"] / players["assists"].max() * 100
players["hs"] = players["hs"] / players["hs"].max() * 100

players["kast"] = players["kast"] - players["kast"].min()
players["kast"] = players["kast"] / players["kast"].max() * 100
players["kast_ct"] = players["kast_ct"] - players["kast_ct"].min()
players["kast_ct"] = players["kast_ct"] / players["kast_ct"].max() * 100
players["kast_t"] = players["kast_t"] - players["kast_t"].min()
players["kast_t"] = players["kast_t"] / players["kast_t"].max() * 100

players["kddiff"] = players["kddiff"] - players["kddiff"].min()
players["kddiff"] = players["kddiff"] / players["kddiff"].max() * 100
players["kddiff_ct"] = players["kddiff_ct"] - players["kddiff_ct"].min()
players["kddiff_ct"] = players["kddiff_ct"] / players["kddiff_ct"].max() * 100
players["kddiff_t"] = players["kddiff_t"] - players["kddiff_t"].min()
players["kddiff_t"] = players["kddiff_t"] / players["kddiff_t"].max() * 100

players["adr"] = players["adr"] - players["adr"].min()
players["adr"] = players["adr"] / players["adr"].max()* 100 
players["adr_ct"] = players["adr_ct"] - players["adr_ct"].min()
players["adr_ct"] = players["adr_ct"] / players["adr_ct"].max() * 100 
players["adr_t"] = players["adr_t"] - players["adr_t"].min()
players["adr_t"] = players["adr_t"] / players["adr_t"].max() * 100

# create player power dataframe and add pleyer names
#player_powers = pd.DataFrame(players_avg_stats["player_name"])
player_powers = pd.DataFrame(players.iloc[:, 2:13].mean(axis=1))

players = pd.concat([pd.DataFrame(players.iloc[:, 0:2]), player_powers], axis=1).copy()
players.columns = ["date", "player_name", "power"]

# create unique id for each game and map
id_col = []
for date, name, power in zip(players["date"], players["player_name"], players["power"]):
    id_col.append(str(match_id) + str(name) + str(power))
players["id"] = id_col

player_power_rows = {}

start = time.time()
def extract_player_games(names, players):
    for player_name in names:
        player_power_rows[player_name] = players[(players["player_name"] == player_name)]

player_names = players["player_name"].unique()

futures_list = []


for names in np.array_split(player_names, 1024):
    futures_list.append(executor.submit(extract_player_games, names, players.copy()))
    
c = 0
for feature in futures_list:
    feature.result(timeout=60*60)
    print(c+1,"/",len(futures_list))
    c += 1

print("extract_player_games finished", time.time() - start)
    

def getTeamPowers(row_indexes, data, player_team, player_power_rows):
    for index in row_indexes:
    
        team1 = data.iloc[index]["team_1"]
        team2 = data.iloc[index]["team_2"]
        date = data.iloc[index]["date"]
        
        team_1_players = player_team[(player_team["team"] == team1) & (player_team["date"] == date)]
        team_2_players = player_team[(player_team["team"] == team2) & (player_team["date"] == date)]
        
        team_player_power_sum = 0
        num_players = 0
    
        for player_name in team_1_players["player_name"]:
            if player_name not in player_power_rows:
                continue
            player_rows = player_power_rows[player_name][(player_power_rows[player_name]["date"] < date)]["power"]
            player_power_sum = 0
            found = 0
            i = 0
            
            while found < 10 and i < len(player_rows):
                player_power_sum += player_rows.iloc[i]
                found += 1    
                i += 1
                
            if found != 0:
                player_avg_power = player_power_sum / found
                team_player_power_sum += player_avg_power
                num_players += 1
                
        if num_players != 0:
            team1_power[index] = (player_power_sum/num_players)
        else:
            team1_power[index] = None
            
        team_player_power_sum = 0
        num_players = 0
        
        for player_name in team_2_players["player_name"]:
            if player_name not in player_power_rows:
                continue
            player_rows = player_power_rows[player_name][(player_power_rows[player_name]["date"] < date)]["power"]
            player_power_sum = 0
            found = 0
            i = 0
            
            while found < 10 and i < len(player_rows):
                player_power_sum += player_rows.iloc[i]
                found += 1               
                i += 1
    
            if found != 0:
                player_avg_power = player_power_sum / found
                team_player_power_sum += player_avg_power
                num_players += 1
    
        if num_players != 0:
            team2_power[index] = (player_power_sum/num_players)
        else:
            team2_power[index] = None
        
        

team1_power = np.zeros((len(data), ))
team2_power = np.zeros((len(data), ))

futures_list = []

start = time.time()
for row_indexes in np.array_split(range(len(data)), 1024):
    futures_list.append(executor.submit(getTeamPowers, row_indexes, data.copy(), player_team.copy(), player_power_rows.copy()))

c = 0
for feature in futures_list:
    feature.result(timeout=60*60)   
    print(c+1,"/",len(futures_list))
    c += 1
    
        
print("getTeamPowers finished", time.time()-start)
    
data["team_1_player_power"] = pd.Series(team1_power)
data["team_2_player_power"] = pd.Series(team2_power)

# fill na values in team powers
data = data.fillna(data.min())

# rescale team powers between 0 and 100
data["team_1_player_power"] = data["team_1_player_power"] - data["team_1_player_power"].min()
data["team_1_player_power"] = data["team_1_player_power"] / data["team_1_player_power"].max() * 100

data["team_2_player_power"] = data["team_2_player_power"] - data["team_2_player_power"].min()
data["team_2_player_power"] = data["team_2_player_power"] / data["team_2_player_power"].max() * 100


def label_encoder(x, columns):
    for column in columns:
        le = preprocessing.LabelEncoder()
        x.iloc[:,column] = le.fit_transform(x.iloc[:,column])
    return x

# encode maps
encoded_data = label_encoder(data.copy(), [2, 3, 4])
map_list = ["cache ", "cobblestone", "dust2", "inferno", "mirage", "nuke", "overpass", "train", "vertigo"]

encoded_data.drop("date", inplace=True, axis=1)

def getTeamWinRate(team1, team2, index, encoded_data):
    team1_game_count = 0
    team1_win_count = 0
    
    team2_game_count = 0
    team2_win_count = 0
    
    i = index + 1
    while i < len(encoded_data):
        if team1_game_count < 10:
            if encoded_data.iloc[i, 1] == team1:
                if encoded_data.iloc[i, 7] == 0:
                    team1_win_count += 1
                team1_game_count += 1
                
            elif encoded_data.iloc[i, 2] == team1:
                if encoded_data.iloc[i, 7] == 1:
                    team1_win_count += 1
                team1_game_count += 1
        

        elif team2_game_count < 10:
            if encoded_data.iloc[i, 1] == team2:
                if encoded_data.iloc[i, 7] == 0:
                    team2_win_count += 1
                team2_game_count += 1
            
            elif encoded_data.iloc[i, 2] == team2:
                if encoded_data.iloc[i, 7] == 1:
                    team2_win_count += 1
                team2_game_count += 1
        else:
            break
        
        i += 1
            
    return team1_win_count, team2_win_count



futures_list = []

team_1_last_10_game_wins = np.zeros((len(encoded_data), ), dtype=np.int32)
team_2_last_10_game_wins = np.zeros((len(encoded_data), ), dtype=np.int32)

start = time.time()

for index, row in encoded_data.iterrows():    
    team1 = row["team_1"]
    team2 = row["team_2"]
    
    futures_list.append(executor.submit(getTeamWinRate, team1, team2, index, encoded_data.copy()))
    
c = 0
for feature in futures_list:
    team1_win_count, team2_win_count = feature.result(timeout=60*60)
    team_1_last_10_game_wins[c] = team1_win_count
    team_2_last_10_game_wins[c] = team2_win_count 
    print(c+1,"/",len(futures_list))
    c += 1
    
print("getTeamWinRate finished", time.time()-start)

encoded_data["team_1_last_10_game_wins"] = pd.Series(team_1_last_10_game_wins)
encoded_data["team_2_last_10_game_wins"] = pd.Series(team_2_last_10_game_wins)

# set label as last column
y = encoded_data["map_winner"]
encoded_data.drop("map_winner", inplace=True, axis=1)
encoded_data = pd.concat([encoded_data, y], axis=1)

encoded_data.to_csv("encoded_data.csv", index=False)









