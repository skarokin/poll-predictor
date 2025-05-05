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

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    def forward(self, x):
        seq_len = x.size(1)
        freqs = torch.outer(torch.arange(seq_len, device=x.device), self.inv_freq)
        sin, cos = torch.sin(freqs), torch.cos(freqs)
        sin, cos = sin[:, None, :], cos[:, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.depth = d_model // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEncoding(self.depth)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.depth).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q), self.rope(k)
        scores = (q @ k.transpose(-2, -1)) / (self.depth ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.fc(out)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.Sequential(*[EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x).squeeze(-1)

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
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = TransformerEncoder(
        input_dim=len(input_cols),
        d_model=256,
        num_heads=4,
        d_ff=256,
        num_layers=8
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  

    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        if epoch == num_epochs - 1:
            print("\nFinal Evaluation Results:")
            weekly_rankings, weekly_metrics = evaluate(model, train_loader, device, verbose=True)
        else:
            _, _ = evaluate(model, train_loader, device, verbose=False)
    
    
