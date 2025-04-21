import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams
from substitution_overlap_graph import build_overlap_graph

SEASON = '2024-25'
THRESHOLD_SECONDS = 600
OUTPUT_DIR = 'data'
LIMIT_GAMES = 10  # Limit number of games for now

boxscore_fields = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK', 'PLUS_MINUS']

X_seq = []
G_seq = []
player_id2name = {}
player_id2team = {}
player_id2position = {}

# Get all games from the season
nba_teams = teams.get_teams()
gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=SEASON, league_id_nullable='00')
games = gamefinder.get_data_frames()[0]
games.drop_duplicates('GAME_ID', keep='first', inplace=True)

print(f"Total games fetched: {len(games)}")

game_ids = games['GAME_ID'].tolist()[:LIMIT_GAMES]  # Only use the first N games

# Loop through games and collect data
for game_id in game_ids:
    try:
        print(f"Processing game {game_id}")
        G = build_overlap_graph(game_id, threshold_seconds=THRESHOLD_SECONDS)
        if G.number_of_edges() == 0:
            print(f"Skipping game {game_id} — no valid player overlaps")
            continue
        G_seq.append(G)

        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = boxscore.get_data_frames()[0]

        player_vectors = []
        player_ids = []

        for _, row in player_stats.iterrows():
            pid = row['PLAYER_ID']
            features = [row.get(field, 0) for field in boxscore_fields]
            player_vectors.append(features)
            player_ids.append(pid)

            player_id2name[pid] = row['PLAYER_NAME']
            player_id2team[pid] = row['TEAM_ABBREVIATION']
            pos = row['START_POSITION']
            player_id2position[pid] = [int(p in pos) for p in ['F', 'G', 'C']]

        df = pd.DataFrame(player_vectors, index=player_ids, columns=boxscore_fields)
        df = df.reindex(sorted(player_id2name.keys()), fill_value=0)
        X_seq.append(df.values)

        print(f"Processed game {game_id} with {len(df)} players")
        time.sleep(0.6)  # rate limit

    except Exception as e:
        print(f"Failed to process game {game_id}: {e}")
        continue

max_players = max(x.shape[0] for x in X_seq)
feature_dim = X_seq[0].shape[1]
X_seq_padded = np.zeros((len(X_seq), max_players, feature_dim))
for i, x in enumerate(X_seq):
    X_seq_padded[i, :x.shape[0], :] = x

# Save data 
os.makedirs(OUTPUT_DIR, exist_ok=True)
pickle.dump(X_seq_padded, open(f"{OUTPUT_DIR}/X_seq.pkl", "wb"))
pickle.dump(G_seq, open(f"{OUTPUT_DIR}/G_seq.pkl", "wb"))
pickle.dump(player_id2name, open(f"{OUTPUT_DIR}/player_id2name.pkl", "wb"))
pickle.dump(player_id2team, open(f"{OUTPUT_DIR}/player_id2team.pkl", "wb"))
pickle.dump(player_id2position, open(f"{OUTPUT_DIR}/player_id2position.pkl", "wb"))