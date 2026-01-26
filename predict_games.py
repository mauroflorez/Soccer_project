import pandas as pd
import numpy as np
import pickle
import os
import argparse
from tabulate import tabulate

# Configuration
DATA_FILE = "data/matches.csv"
MODEL_FILE = "models/models.pkl"
ROLLING_WINDOW = 5

def load_models():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file {MODEL_FILE} not found. Run train_model.py first.")
    with open(MODEL_FILE, "rb") as f:
        models = pickle.load(f)
    return models

def get_team_latest_stats(df, team_name):
    """
    Calculates the latest rolling stats for a specific team based on their last games.
    """
    # Create a team-centric view (similar to feature_engineering.py but we want the LATEST state)
    # We filter for games involving this team
    team_games = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    
    if len(team_games) == 0:
        print(f"Warning: No games found for {team_name}")
        return None
        
    team_games['Date'] = pd.to_datetime(team_games['Date'], dayfirst=True)
    team_games = team_games.sort_values('Date')
    
    # Calculate stats for each game *from the perspective of the target team*
    stats = []
    for _, row in team_games.iterrows():
        if row['HomeTeam'] == team_name:
            stats.append({
                'GoalsScored': row['FTHG'],
                'GoalsConceded': row['FTAG'],
                'Points': 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
            })
        else: # AwayTeam == team_name
            stats.append({
                'GoalsScored': row['FTAG'],
                'GoalsConceded': row['FTHG'],
                'Points': 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
            })
            
    stats_df = pd.DataFrame(stats)
    
    # Calculate rolling averages for the LAST 5 games
    # We take the tail (most recent)
    last_5 = stats_df.tail(ROLLING_WINDOW)
    
    if len(last_5) < 1:
        return None
        
    return {
        'AvgGoalsScored': last_5['GoalsScored'].mean(),
        'AvgGoalsConceded': last_5['GoalsConceded'].mean(),
        'AvgPoints': last_5['Points'].mean()
    }

def predict_match(home_team, away_team, models, df):
    # Get latest stats
    home_stats = get_team_latest_stats(df, home_team)
    away_stats = get_team_latest_stats(df, away_team)
    
    if not home_stats:
        return f"Could not calculate stats for {home_team}"
    if not away_stats:
        return f"Could not calculate stats for {away_team}"
        
    # Prepare features vector (single row dataframe)
    # Must match the column names used in training (Statsmodels formula uses these names)
    features = pd.DataFrame([{
        'Home_AvgGoalsScored': home_stats['AvgGoalsScored'],
        'Home_AvgGoalsConceded': home_stats['AvgGoalsConceded'],
        'Home_AvgPoints': home_stats['AvgPoints'],
        'Away_AvgGoalsScored': away_stats['AvgGoalsScored'],
        'Away_AvgGoalsConceded': away_stats['AvgGoalsConceded'],
        'Away_AvgPoints': away_stats['AvgPoints']
    }])
    
    # Predict using Poisson models
    poisson_home = models.get('poisson_home')
    poisson_away = models.get('poisson_away')
    
    if not poisson_home or not poisson_away:
        return "Poisson models not found."
        
    home_goals_exp = poisson_home.predict(features).values[0]
    away_goals_exp = poisson_away.predict(features).values[0]
    
    return {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'HomeGoals_Exp': home_goals_exp,
        'AwayGoals_Exp': away_goals_exp,
        'PredictedScore': f"{int(round(home_goals_exp))} - {int(round(away_goals_exp))}"
    }

def main():
    parser = argparse.ArgumentParser(description="Predict EPL Match Scores")
    parser.add_argument("--home", type=str, help="Home Team Name", required=False)
    parser.add_argument("--away", type=str, help="Away Team Name", required=False)
    args = parser.parse_args()

    print("Loading models and data...")
    try:
        models = load_models()
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    teams = sorted(df['HomeTeam'].unique())
    
    if args.home and args.away:
        # Predict single match from arguments
        if args.home not in teams or args.away not in teams:
            print(f"Error: Invalid team name. Available teams: {teams[:5]}...")
            return
        result = predict_match(args.home, args.away, models, df)
        print("\nPrediction:")
        print(f"{result['HomeTeam']} vs {result['AwayTeam']}")
        print(f"Expected Goals: {result['HomeGoals_Exp']:.2f} - {result['AwayGoals_Exp']:.2f}")
        print(f"SCORE PREDICTION: {result['PredictedScore']}")
        
    else:
        # Interactive Mode or Demo
        print("\n--- Interactive Prediction Mode ---")
        print("Enter team names exactly as they appear in data (e.g., 'Man City', 'Liverpool', 'Arsenal')")
        
        while True:
            print(f"\nAvailable Teams (Top 5 alpha): {teams[:5]}...")
            h = input("Enter Home Team (or 'q' to quit): ").strip()
            if h.lower() == 'q': break
            a = input("Enter Away Team: ").strip()
            
            if h not in teams:
                print(f"Error: '{h}' not found.")
                continue
            if a not in teams:
                print(f"Error: '{a}' not found.")
                continue
                
            result = predict_match(h, a, models, df)
            
            table = [
                ["Match", f"{result['HomeTeam']} vs {result['AwayTeam']}"],
                ["Expected Goals", f"{result['HomeGoals_Exp']:.2f} - {result['AwayGoals_Exp']:.2f}"],
                ["PREDICTION", result['PredictedScore']]
            ]
            print(tabulate(table, tablefmt="grid"))

if __name__ == "__main__":
    main()
