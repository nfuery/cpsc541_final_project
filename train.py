import copy
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from IPython import display
from torch_geometric.nn import GATConv
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from gatv2tcn import ASTGCN, GATv2TCN, GATv2Conv
from sklearn import preprocessing
from tqdm import tqdm


SEQ_LENGTH = 10
OFFSET = 1
player_boxscore_fields = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK', 'PLUS_MINUS']
player_boxscore_tracking_fields = ['TCHS', 'PASS', 'DIST']
player_boxscore_advanced_fields = ['PACE', 'USG_PCT', 'TS_PCT']
player_prediction_metrics = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK']
player_prediction_metrics_index = [
    (player_boxscore_fields + player_boxscore_tracking_fields + player_boxscore_advanced_fields).index(metric) for
    metric in player_prediction_metrics]


def fill_zeros_with_last(seq):
    seq_ffill = np.zeros_like(seq)
    for i in range(seq.shape[1]):
        arr = seq[:, i]
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        seq_ffill[:, i] = arr[prev]

    return seq_ffill

def construct_input_sequences_and_output(z, seq_length=10, offset=1):
    # Check if z is already a numpy array
    if isinstance(z, (np.ndarray, np.generic)):
        # For NumPy arrays, use sliding_window_view
        if offset == 0:
            x = sliding_window_view(z, seq_length, axis=0)
        else:
            x = sliding_window_view(z[:-offset], seq_length, axis=0)
        y = z[seq_length+offset-1:]
    else:
        # For lists of tensors or other objects, manually create sequences
        x = []
        for i in range(len(z) - seq_length - offset + 1):
            x.append(z[i:i+seq_length])
        y = z[seq_length+offset-1:]
    
    return x, y    

def create_dataset():
    # X_seq breakdown
    # 3D NumPy array of shape (num_games, num_players, num_features)
        # num_games -> each entry in the first dimension corresponds to a snapshot of a day's worth of games (single game day)
        # num_players -> all players appeared across any game (aligned by player id)
        # num_features -> stats (pts, ast, reb, to, stl, blk, +-)
    # X_seq[i] is a snapshot of all tracked players' features
        # E.g. X_seq[0] -> stats from 10/25 games, X_seq[1] -> stats from 10/27 games
            # Then X_seq[0][12] and X_seq[1][12] are the same player on diff game days
                # If they didnt play, row will be all 0s or missing (I THINK)
    X_seq = pd.read_pickle('data/X_seq.pkl')
    G_seq = pd.read_pickle('data/G_seq.pkl')
    player_id_to_team = pd.read_pickle('data/player_id2team.pkl')
    player_id_to_position = pd.read_pickle('data/player_id2position.pkl')

    le = preprocessing.LabelEncoder()
    df_id2team = pd.DataFrame.from_dict(player_id_to_team, orient='index').apply(le.fit_transform)
    enc = preprocessing.OneHotEncoder()
    enc.fit(df_id2team)
    onehotlabels = enc.transform(df_id2team).toarray()
    team_tensor = Variable(torch.FloatTensor(onehotlabels))
    position_tensor = Variable(torch.FloatTensor(np.stack(list(player_id_to_position.values()), axis=0)))

    Xs = np.zeros_like(X_seq)
    for i in range(X_seq.shape[1]):
        Xs[:, i, :] = fill_zeros_with_last(X_seq[:, i, :])

    Gs = []
    c = 0
    for g in G_seq:
        c += 1
        print(c)
        node_dict = {node: i for i, node in enumerate(g.nodes())}
        edges = np.array([edge.split(' ') for edge in nx.generate_edgelist(g)])[:, :2].astype(int).T
        edges = np.vectorize(node_dict.__getitem__)(edges)
        Gs.append(torch.LongTensor(np.hstack((edges, edges[[1, 0]]))))

    X_in, X_out = construct_input_sequences_and_output(Xs, seq_length=SEQ_LENGTH, offset=OFFSET)
    G_in, G_out = construct_input_sequences_and_output(Gs, seq_length=SEQ_LENGTH, offset=OFFSET)

    X_in = Variable(torch.FloatTensor(X_in))
    X_out = Variable(torch.FloatTensor(X_out))

    X_train, X_val, X_test = X_in[:31], X_in[41:41+16], X_in[41+26:]
    y_train, y_val, y_test = X_out[:31], X_out[41:41+16], X_out[41+26:]

    g_train = G_in[:31]
    g_val = G_in[41:41+16]
    g_test = G_in[41+26:]
    h_train = G_out[:31]
    h_val = G_out[41:41+16]
    h_test = G_out[41+26:]

    print(X_train.shape, X_val.shape, X_test.shape)
    print(f"g_train length: {len(g_train)}, g_val length: {len(g_val)}, g_test length: {len(g_test)}")
    print(f"h_train length: {len(h_train)}, y_train length: {len(y_train)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test, h_train, h_val, h_test, team_tensor, position_tensor


X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test, h_train, h_val, h_test, team_tensor, position_tensor = create_dataset()
team_embedding_in = team_tensor.shape[-1]
team_embedding_out = 8
team_embedding = nn.Linear(team_embedding_in, team_embedding_out)

position_embedding_in = position_tensor.shape[-1]
position_embedding_out = 8
position_embedding = nn.Linear(position_embedding_in, position_embedding_out)

model_in = y_train.shape[-1] + team_embedding_out + position_embedding_out

model = GATv2TCN(in_channels=model_in,
        out_channels=6,
        len_input=10,
        len_output=1,
        temporal_filter=64,
        out_gatv2conv=32,
        dropout_tcn=0.25,
        dropout_gatv2conv=0.5,
        head_gatv2conv=4)

model_name = 'gatv2tcn-team-position-embedding'

if not os.path.exists(f"model/{model_name}"):
    os.mkdir(f"model/{model_name}")

parameters = list(model.parameters()) + list(team_embedding.parameters()) + list(position_embedding.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=0.001)

min_val_loss = np.inf
min_val_iter = -1

PT_INDEX = 0
EPOCHS = 300
train_loss_history = np.zeros(EPOCHS)
val_loss_history = np.zeros(EPOCHS)
for epoch in tqdm(range(EPOCHS)):
    epoch_train_loss_sum = 0.0
    model.train()

    # Ensuring embedding layers are in train mode
    team_embedding.train()
    position_embedding.train()

    team_embedding_vector = team_embedding(team_tensor)
    position_embedding_vector = position_embedding(position_tensor)

    optimizer.zero_grad()

    for i in range(X_train.shape[0]):
        y_train_mask = h_train[i].unique()
        X_list = []
        G_list = []
        for j in range(SEQ_LENGTH):
            X_list.append(torch.cat([X_train[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
            G_list.append(g_train[i][j])
        x = torch.stack(X_list, dim=-1)
        x = x[None, :, :, :] # Add batch dim
        
        # Forward pass
        x_astgat = model(x, G_list)[0, ...] # Remove batch dim from output

        train_loss = F.l1_loss(x_astgat[y_train_mask], y_train[i][y_train_mask][:, player_prediction_metrics_index])
        train_loss.backward(retain_graph=True) # Accumulate gradients
        epoch_train_loss_sum += train_loss.item() # Store itemized loss

    # Update weights after processing all samples in the epoch
    optimizer.step()

    epoch_val_loss_sum = 0.0
    model.eval()
    team_embedding.eval()
    position_embedding.eval()

    with torch.no_grad(): # Ensure no gradients are computed during validation
        for i in range(X_val.shape[0]):
            y_val_mask = h_val[i].unique()
            X_list = []
            G_list = []
            for j in range(SEQ_LENGTH):
                X_list.append(torch.cat([X_val[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
                G_list.append(g_val[i][j])
            x = torch.stack(X_list, dim=-1)
            x = x[None, :, :, :]
            x_astgat = model(x, G_list)[0, :, :]

            val_loss = F.l1_loss(x_astgat[y_val_mask], y_val[i][y_val_mask][:, player_prediction_metrics_index])
            epoch_val_loss_sum += val_loss.item()

    # Store epoch losses
    epoch_avg_train_loss = epoch_train_loss_sum / X_train.shape[0]
    epoch_avg_val_loss = epoch_val_loss_sum / X_val.shape[0]

    train_loss_history[epoch] = epoch_avg_train_loss
    val_loss_history[epoch] = epoch_avg_val_loss

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_avg_train_loss:.5f}, Val Loss: {epoch_avg_val_loss:.5f}, LR: {current_lr:.6f}")

    if min_val_loss > epoch_avg_val_loss:
        print(f"Validation Loss Decreased({min_val_loss:.5f}--->{epoch_avg_val_loss:.5f}) \t Saving The Model")
        min_val_loss = epoch_avg_val_loss
        min_val_iter = epoch
        # Saving State Dict
        torch.save(model.state_dict(), f"model/{model_name}/saved_astgcn.pth")
        torch.save(team_embedding.state_dict(), f"model/{model_name}/team_embedding.pth")
        torch.save(position_embedding.state_dict(), f"model/{model_name}/position_embedding.pth")

    # Early stopping
    if epoch - min_val_iter > 20:
        print(f"Early stopping at epoch {epoch} (no improvement since {min_val_iter})")
        break

print(min_val_loss, min_val_iter)

# Displaying train and validation loss plots
plt.plot(train_loss_history, label='Train')
plt.plot(val_loss_history, label='Val')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

#GATv2TCN
astgcn_test = copy.deepcopy(model)
astgcn_test.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))
astgcn_test.eval()

team_embedding_test = copy.deepcopy(team_embedding)
team_embedding_test.load_state_dict(torch.load(f"model/{model_name}/team_embedding.pth"))
team_embedding_test.eval()

position_embedding_test = copy.deepcopy(position_embedding)
position_embedding_test.load_state_dict(torch.load(f"model/{model_name}/position_embedding.pth"))
position_embedding_test.eval()

team_embedding_vector = team_embedding_test(team_tensor)
position_embedding_vector = position_embedding_test(position_tensor)

test_loss_l1 = 0.0
test_loss_rmse = 0.0
test_corr = 0.0
test_loss_mape = 0.0

for i in range(X_test.shape[0]):
    y_test_mask = h_test[i].unique()
    X_list = []
    G_list = []
    for j in range(SEQ_LENGTH):
        X_list.append(torch.cat([X_test[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
        G_list.append(g_test[i][j])
    x = torch.stack(X_list, dim=-1)
    x = x[None, :, :, :]
    x_astgcn = astgcn_test(x, G_list)[0, :, :]
    test_loss_rmse += root_mean_squared_error(x_astgcn[y_test_mask].detach().numpy(), y_test[i][y_test_mask][:, player_prediction_metrics_index].detach().numpy()) # torch.sqrt(F.mse_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index]))
    test_loss_l1 += F.l1_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index])
    test_loss_mape += mean_absolute_percentage_error(x_astgcn[y_test_mask].detach().numpy(), y_test[i][y_test_mask][:, player_prediction_metrics_index].detach().numpy()) # torch.sqrt(F.mse_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index]))
    test_corr += torch.tanh(torch.mean(torch.stack([torch.arctanh(torch.corrcoef(torch.stack([x_astgcn[y_test_mask][:, metric_idx],
                             y_test[i][y_test_mask][:, player_prediction_metrics_index][:, metric_idx]], dim=0))[0, 1])
                            for metric_idx in range(len(player_prediction_metrics))])))
print(f"RMSE: {test_loss_rmse/X_test.shape[0]}, MAPE: {test_loss_mape/X_test.shape[0]}, CORR: {test_corr/X_test.shape[0]}, MAE: {test_loss_l1/X_test.shape[0]}")


player_id_to_team = pd.read_pickle('data/player_id2team.pkl')
from nba_api.stats.static import teams, players
nba_teams = teams.get_teams()
team_vec = team_embedding_vector.detach().numpy()
from pandas.plotting._matplotlib.style import get_standard_colors
from matplotlib.lines import Line2D
colors = get_standard_colors(num_colors=len(nba_teams))
markers = list(Line2D.markers.keys())[:len(nba_teams)+1]

fig, ax = plt.subplots()
for i, team in enumerate(nba_teams):
    player_in_team = [idx for idx, team_name in enumerate(player_id_to_team.values()) if team_name == team['nickname']]
    ax.plot(team_vec[player_in_team, 0], team_vec[player_in_team, 1], color=colors[i], marker=markers[i+1], label=team['nickname'])
    plt.text(team_vec[player_in_team, 0].mean(), team_vec[player_in_team, 1].mean(), team['nickname'])

player_id_to_position = pd.read_pickle('data/player_id2position.pkl')
position_vec = position_embedding_vector.detach().numpy()

fig, ax = plt.subplots()
position_dict = {(0, 0, 0): 'No position',
                 (0, 0, 1): 'C',
                 (0, 1, 0): 'G',
                 (1, 0, 0): 'F',
                 (1, 0, 1): 'F/C',
                 (1, 1, 0): 'F/G'}
for i, position in enumerate(np.unique(np.array(list(player_id_to_position.values())), axis=0)):
    player_at_position = [idx for idx, player_position in enumerate(player_id_to_position.values()) if (player_position==position).all()]
    label = position_dict[tuple(position)]
    ax.plot(position_vec[player_at_position, 0], position_vec[player_at_position, 1], color=colors[i], marker=markers[i+1], label=label)
ax.legend()
pass
