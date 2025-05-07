import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm
import pickle
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import time

class FootballRankingLGBM:
    def __init__(self, features_path="college_football.csv", 
                 labels_path="ground_truth.csv",
                 encoder_path="team_encoder.pkl",
                 min_teams=25):
        self.features_df = pd.read_csv(features_path)
        self.labels_df = pd.read_csv(labels_path)
        self.min_teams = min_teams
        
        with open(encoder_path, 'rb') as f:
            self.le_team = pickle.load(f)
        
        self.exclude_cols = ["teamName", "weekNumber", "seasonYear", "nextWeekRank"]
        self.input_cols = [col for col in self.features_df.columns if col not in self.exclude_cols]

        self.prepare_data()

        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 5,
            'max_depth': 6,
            'n_estimators': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        
        self.model = None
    
    def prepare_data(self):
        data = self.features_df.merge(self.labels_df, on=["seasonYear", "weekNumber", "teamName"])
        
        week_counts = data.groupby(["seasonYear", "weekNumber"]).size()
        valid_weeks = week_counts[week_counts >= self.min_teams].index
        data = data.set_index(["seasonYear", "weekNumber"]).loc[valid_weeks].reset_index()
        
        weeks = data.groupby(["seasonYear", "weekNumber"])
        
        X_list = []
        y_list = []
        qid_list = []
        team_names_list = []
        week_keys = []
        
        for i, ((season, week), week_data) in enumerate(weeks):
            scaler = StandardScaler()
            week_features = week_data[self.input_cols]
            X = scaler.fit_transform(week_features)
            
            # AP Rank - convert to relevance score (lower rank = higher relevance)
            # convert ranks to relevance scores (26 - rank) so rank 1 = relevance 25
            # unranked teams (rank 26) will get relevance score 0
            y = week_data["apRank"].values
            relevance = np.where(y <= 25, 26 - y, 0)
            
            qid = np.full(len(week_data), i)
            
            team_names = week_data["teamName"].values
            
            X_list.append(X)
            y_list.append(relevance)
            qid_list.append(qid)
            team_names_list.append(team_names)
            week_keys.append((season, week))
        
        self.X = np.vstack(X_list)
        self.y = np.concatenate(y_list)
        self.qids = np.concatenate(qid_list)
        self.team_names = team_names_list
        self.week_keys = week_keys
        
        self.group_sizes = [len(group) for group in y_list]
    
    def train(self):
        print("Training LightGBM ranking model...")
        start_time = time.time()

        train_data = lgbm.Dataset(
            self.X, 
            label=self.y,
            group=self.group_sizes
        )
        
        self.model = lgbm.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
        )
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds")

        importance = self.model.feature_importance(importance_type='split')
        feature_names = self.input_cols
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
    
    def evaluate(self, top_k=25, verbose=True):
        if self.model is None:
            print("Model not trained yet. Please call train() first.")
            return None
        
        all_rankings = []
        weekly_metrics = {
            'spearman': [],
            'kendall': [],
            'top_k_acc': [],
            'weeks': []
        }
        
        # Get predictions for all instances
        all_preds = self.model.predict(self.X)
        
        # Split predictions by week
        start_idx = 0
        for i, group_size in enumerate(self.group_sizes):
            season, week = self.week_keys[i]
            team_ids = self.team_names[i]
            
            # Get predictions and true ranks for this week
            week_preds = all_preds[start_idx:start_idx+group_size]
            week_relevance = self.y[start_idx:start_idx+group_size]
            
            # Convert relevance back to ranks (for comparison with predictions)
            true_ranks = np.where(week_relevance > 0, 26 - week_relevance, 26)
            
            # Get predicted ranks (smaller score = worse rank)
            # Higher prediction value = higher relevance = better rank (lower number)
            sorted_indices = np.argsort(-week_preds)  
            pred_ranks = np.empty_like(sorted_indices)
            pred_ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
            
            try:
                rho, _ = spearmanr(pred_ranks, true_ranks)
                if np.isnan(rho):
                    rho = 0.0  # Default value when correlation is undefined
            except:
                rho = 0.0
                
            try:
                tau, _ = kendalltau(pred_ranks, true_ranks)
                if np.isnan(tau):
                    tau = 0.0
            except:
                tau = 0.0
            
            true_top_k = set(np.where(true_ranks <= top_k)[0])
            pred_top_k = set(sorted_indices[:top_k])
            
            denominator = min(len(true_top_k), top_k)
            if denominator > 0:
                top_k_match = len(true_top_k & pred_top_k) / denominator
            else:
                # there are no teams in true top-k, we can't calculate accuracy
                top_k_match = 0.0  
            
            weekly_metrics['weeks'].append(f"{season}-{week}")
            weekly_metrics['spearman'].append(rho)
            weekly_metrics['kendall'].append(tau)
            weekly_metrics['top_k_acc'].append(top_k_match)
            
            # Get team names for this week's ranking
            team_names = [self.le_team.inverse_transform([tid])[0] for tid in team_ids]
            ranked_teams = [team_names[i] for i in sorted_indices]
            ranked_scores = week_preds[sorted_indices]
            
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
            
            start_idx += group_size
        
        if verbose:
            print("\nWeekly Metrics Summary:")
            print(f"{'Week':<15}{'Spearman (ρ)':<15}{'Kendall (τ)':<15}{'Top-25 Acc':<15}")
            for week, rho, tau, acc in zip(weekly_metrics['weeks'], weekly_metrics['spearman'], 
                                          weekly_metrics['kendall'], weekly_metrics['top_k_acc']):
                print(f"{week:<15}{rho:<15.4f}{tau:<15.4f}{acc:<15.4f}")
            
            avg_rho = np.mean(weekly_metrics['spearman'])
            avg_tau = np.mean(weekly_metrics['kendall'])
            avg_acc = np.mean(weekly_metrics['top_k_acc'])
            print("\nAverage Metrics:")
            print(f"Spearman Correlation (ρ): {avg_rho:.4f}")
            print(f"Kendall's Tau (τ): {avg_tau:.4f}")
            print(f"Top-25 Accuracy: {avg_acc:.4f}")
            
            self.plot_metrics(weekly_metrics)
        
        return all_rankings, weekly_metrics
    
    def plot_metrics(self, weekly_metrics):
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
    
    def save_model(self, path="football_ranking_lgbm.txt"):
        if self.model:
            self.model.save_model(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save. Please train the model first.")
    
    def load_model(self, path="football_ranking_lgbm.txt"):
        self.model = lgbm.Booster(model_file=path)
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    lgbm_ranker = FootballRankingLGBM()
    lgbm_ranker.train()
    
    rankings, metrics = lgbm_ranker.evaluate()
    
    lgbm_ranker.save_model()