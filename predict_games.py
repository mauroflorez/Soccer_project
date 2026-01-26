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

import json
from datetime import datetime

def get_next_fixtures(df):
    """
    Returns the upcoming fixtures based on user input (Jan 26 - Feb 2).
    """
    return [
        ("Everton", "Leeds"),            # Jan 26
        ("Leeds", "Arsenal"),           # Jan 31
        ("Wolves", "Bournemouth"),      # Jan 31
        ("Brighton", "Everton"),        # Jan 31
        ("Chelsea", "West Ham"),        # Jan 31
        ("Liverpool", "Newcastle"),     # Jan 31
        ("Aston Villa", "Brentford"),   # Feb 1
        ("Man United", "Fulham"),       # Feb 1
        ("Nott'm Forest", "Crystal Palace"), # Feb 1 (Check team name spelling in CSV)
        ("Tottenham", "Man City"),      # Feb 1
        ("Sunderland", "Burnley")       # Feb 2
    ]

def evaluate_past_predictions(df, models):
    """
    Evaluates predictions against the specific results provided by the user (Jan 24-25).
    Since the CSV might lag, we hardcoded the actuals here to ensure the website shows them correctly.
    """
    # Specific results from Jan 24-25
    recent_results = [
        {"Home": "West Ham", "Away": "Sunderland", "Actual": "3 - 1"},
        {"Home": "Fulham", "Away": "Brighton", "Actual": "2 - 1"},
        {"Home": "Burnley", "Away": "Tottenham", "Actual": "2 - 2"},
        {"Home": "Man City", "Away": "Wolves", "Actual": "2 - 0"},
        {"Home": "Bournemouth", "Away": "Liverpool", "Actual": "3 - 2"},
        {"Home": "Crystal Palace", "Away": "Chelsea", "Actual": "1 - 3"},
        {"Home": "Newcastle", "Away": "Aston Villa", "Actual": "0 - 2"},
        {"Home": "Brentford", "Away": "Nott'm Forest", "Actual": "0 - 2"}, # Check spelling
        {"Home": "Arsenal", "Away": "Man United", "Actual": "2 - 3"}
    ]
    
    # Handle team name mapping if needed (e.g. Nott'm Forest vs Nottm Forest)
    # in matches.csv it's usually "Nott'm Forest"
    
    results_data = []
    teams_in_db = set(df['HomeTeam'].unique())

    for match in recent_results:
        home = match["Home"]
        away = match["Away"]
        
        # Simple name correction if needed
        if home == "Nottm Forest" and "Nott'm Forest" in teams_in_db: home = "Nott'm Forest"
        if away == "Nottm Forest" and "Nott'm Forest" in teams_in_db: away = "Nott'm Forest"

        if home not in teams_in_db or away not in teams_in_db:
            print(f"Skipping {home} vs {away} - Not in dataset")
            continue

        # Retroactive prediction
        pred = predict_match(home, away, models, df)
        
        if isinstance(pred, str): continue 
        
        results_data.append({
            'HomeTeam': home,
            'AwayTeam': away,
            'PredictedScore': pred['PredictedScore'],
            'ActualScore': match["Actual"],
            'ExpectedGoals': f"{pred['HomeGoals_Exp']:.2f} - {pred['AwayGoals_Exp']:.2f}"
        })
        
    return results_data

def main():
    parser = argparse.ArgumentParser(description="Predict EPL Match Scores")
    parser.add_argument("--home", type=str, help="Home Team Name", required=False)
    parser.add_argument("--away", type=str, help="Away Team Name", required=False)
    parser.add_argument("--json", type=str, help="Output JSON file path for UPCOMING predictions", required=False)
    parser.add_argument("--past-results-json", type=str, help="Output JSON file for PAST results/eval", required=False)
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
        # Predict single match
        if args.home not in teams or args.away not in teams:
            print(f"Error: Invalid team name.")
            return
        result = predict_match(args.home, args.away, models, df)
        
        print("\nPrediction:")
        print(f"{result['HomeTeam']} vs {result['AwayTeam']}")
        print(f"Expected Goals: {result['HomeGoals_Exp']:.2f} - {result['AwayGoals_Exp']:.2f}")
        print(f"SCORE PREDICTION: {result['PredictedScore']}")
        
    elif args.json:
        # 1. Generate Upcoming Predictions
        print("Generating upcoming predictions...")
        next_matchups = get_next_fixtures(df)
        predictions = []
        for h, a in next_matchups:
            # Check availability
            if h in teams and a in teams:
                result = predict_match(h, a, models, df)
                predictions.append(result)
        
        with open(args.json, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"Upcoming predictions saved to {args.json}")
        
        # 2. Generate Past Results (if requested)
        if args.past_results_json:
            print("Generating past results evaluation...")
            past_results = evaluate_past_predictions(df, models)
            with open(args.past_results_json, 'w') as f:
                json.dump(past_results, f, indent=4)
            print(f"Past results saved to {args.past_results_json}")

    else:
        # Interactive Mode
        print("\n--- Interactive Prediction Mode ---")
        while True:
            print(f"\nAvailable Teams (Top 5 alpha): {teams[:5]}...")
            h = input("Enter Home Team (or 'q' to quit): ").strip()
            if h.lower() == 'q': break
            a = input("Enter Away Team: ").strip()
            
            if h not in teams or a not in teams:
                print("Error: Team not found.")
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
