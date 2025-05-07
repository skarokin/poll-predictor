import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        # x shape: (batch_size, num_teams_in_week, input_dim)
        # We want to process each team independently to get a score
        
        # If batch_size is 1,
        # x will be (1, num_teams_in_week, input_dim)
        # We can squeeze the batch dimension if it's 1, or iterate if it's larger
        # For simplicity with current setup, assuming batch_size=1 from DataLoader
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0) # Shape becomes (num_teams_in_week, input_dim)

        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x) # Shape: (num_teams_in_week, 1)
        return x.squeeze(-1) # Shape: (num_teams_in_week)

class FootballDataset(Dataset):
    def __init__(self, df, ap_df, input_cols, min_teams=25, encoder_path='team_encoder.pkl'):  
        self.data = df.merge(ap_df, on=["seasonYear", "weekNumber", "teamName"])
        
        with open(encoder_path, 'rb') as f:
            self.le_team = pickle.load(f)

        week_counts = self.data.groupby(["seasonYear", "weekNumber"]).size()
        valid_weeks = week_counts[week_counts >= min_teams].index
        self.data = self.data.set_index(["seasonYear", "weekNumber"]).loc[valid_weeks].reset_index()
        
        self.groups = self.data.groupby(["seasonYear", "weekNumber"]).groups
        self.input_cols = input_cols
        
        self.features = []
        self.labels = []
        self.team_names = []
        
        for (season, week), indices in self.groups.items():
            week_data = self.data.loc[indices]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(week_data[input_cols])
            self.features.append(torch.tensor(scaled_features, dtype=torch.float32))
            self.labels.append(torch.tensor(week_data["apRank"].values, dtype=torch.float32))
            self.team_names.append(week_data["teamName"].values)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.team_names[idx]  
    
    def get_team_name(self, encoded_value):
        return self.le_team.inverse_transform([encoded_value])[0]

def listMLELoss(y_pred, y_true):
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)
    
    sorted_idx = torch.argsort(y_true, dim=1, descending=True)  
    sorted_preds = torch.gather(y_pred, 1, sorted_idx)
    loss = torch.logcumsumexp(sorted_preds, dim=1) - sorted_preds
    return loss.mean()

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    
    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        
        if x.shape[1] < 2:
            continue
            
        optimizer.zero_grad()
        preds = model(x)
        loss = listMLELoss(preds, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else float('nan')

def evaluate(model, dataloader, device, top_k=25, verbose=False):
    model.eval()
    all_rankings = []
    weekly_metrics = {
        'spearman': [],
        'kendall': [],
        'top_k_acc': [],
        'weeks': []
    }
    
    with torch.no_grad():
        for batch_idx, (x, y_true, team_ids) in enumerate(dataloader):
            x = x.to(device)
            y_true = y_true.to(device)
            
            if x.shape[1] < top_k:
                continue
                
            preds = model(x)
            current_key = list(dataloader.dataset.groups.keys())[batch_idx]
            season, week = current_key
            weekly_metrics['weeks'].append(f"{season}-{week}")
            
            pred_ranks = torch.argsort(torch.argsort(preds.squeeze(), descending=True)).cpu().numpy() + 1
            true_ranks = torch.argsort(torch.argsort(y_true.squeeze(), descending=True)).cpu().numpy() + 1
            
            rho, _ = spearmanr(pred_ranks, true_ranks)
            tau, _ = kendalltau(pred_ranks, true_ranks)
            top_k_match = np.mean(np.isin(pred_ranks[:top_k], true_ranks[:top_k]))
            
            weekly_metrics['spearman'].append(rho)
            weekly_metrics['kendall'].append(tau)
            weekly_metrics['top_k_acc'].append(top_k_match)
            
            team_names = [dataloader.dataset.get_team_name(tid.item()) for tid in team_ids[0]]
            
            sorted_indices = torch.argsort(preds.squeeze(), descending=True)
            ranked_teams = [team_names[i] for i in sorted_indices.cpu().numpy()]
            ranked_scores = preds.squeeze()[sorted_indices].cpu().numpy()
            
            if verbose:
                print(f"\nSeason {season}, Week {week} Top {top_k}:")
                for rank, (team, score) in enumerate(zip(ranked_teams[:top_k], ranked_scores[:top_k]), 1):
                    print(f"{rank}. {team} (Score: {score:.4f})")
            
            all_rankings.append({
                'season': season,
                'week': week,
                'ranking': ranked_teams[:top_k],
                'scores': ranked_scores[:top_k].tolist(),
                'true_ranks': true_ranks,
                'pred_ranks': pred_ranks,
                'team_names': team_names
            })
    
    if verbose:
        print("\nWeekly Metrics Summary:")
        print(f"{'Week':<15}{'Spearman (ρ)':<15}{'Kendall (τ)':<15}{'Top-25 Acc':<15}")
        for week, rho, tau, acc in zip(weekly_metrics['weeks'], weekly_metrics['spearman'],  weekly_metrics['kendall'], weekly_metrics['top_k_acc']):
            print(f"{week:<15}{rho:<15.4f}{tau:<15.4f}{acc:<15.4f}")
        
        avg_rho = np.mean(weekly_metrics['spearman'])
        avg_tau = np.mean(weekly_metrics['kendall'])
        avg_acc = np.mean(weekly_metrics['top_k_acc'])
        print("\nAverage Metrics:")
        print(f"Spearman Correlation (ρ): {avg_rho:.4f}")
        print(f"Kendall's Tau (τ): {avg_tau:.4f}")
        print(f"Top-25 Accuracy: {avg_acc:.4f}")
        
        plot_metrics(weekly_metrics)
    
    return all_rankings, weekly_metrics

def plot_metrics(weekly_metrics):
    plt.figure(figsize=(18, 5))  
    
    plt.subplot(131)
    plt.plot(weekly_metrics['spearman'], marker='o')
    plt.title("Spearman Correlation")
    plt.xlabel("Week")
    plt.ylabel("ρ")
    plt.xticks(range(len(weekly_metrics['weeks'])), weekly_metrics['weeks'], rotation=45)
    plt.gca().set_xmargin(0.05) 
    
    plt.subplot(132)
    plt.plot(weekly_metrics['kendall'], marker='o')
    plt.title("Kendall's Tau")
    plt.xlabel("Week")
    plt.ylabel("τ")
    plt.xticks(range(len(weekly_metrics['weeks'])), weekly_metrics['weeks'], rotation=45)
    plt.gca().set_xmargin(0.05)
    
    plt.subplot(133)
    plt.plot(weekly_metrics['top_k_acc'], marker='o')
    plt.title(f"Top-25 Accuracy")
    plt.xlabel("Week")
    plt.ylabel("Accuracy")
    plt.xticks(range(len(weekly_metrics['weeks'])), weekly_metrics['weeks'], rotation=45)
    plt.gca().set_xmargin(0.05)
    
    plt.tight_layout(w_pad=2.0)  
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_df = pd.read_csv("college_football.csv")
    ap_rank_df = pd.read_csv("ground_truth.csv")

    input_cols = [col for col in features_df.columns 
                 if col not in ["teamName", "weekNumber", "seasonYear", "nextWeekRank"]]
    
    dataset = FootballDataset(features_df, ap_rank_df, input_cols, min_teams=25)
    # since we expect to train one week at a time, we must be using batch size of 1
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True) 

    model = MLPModel(
        input_dim=len(input_cols),
        hidden_dim1=256,
        hidden_dim2=128,
        dropout_rate=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)  

    num_epochs = 50 # Or your desired number of epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        if epoch == num_epochs - 1:
            print("\nFinal Evaluation Results:")
            weekly_rankings, weekly_metrics = evaluate(model, train_loader, device, verbose=True)
        else:
            _, _ = evaluate(model, train_loader, device, verbose=False)