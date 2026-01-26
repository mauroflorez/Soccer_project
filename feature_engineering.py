import pandas as pd
import numpy as np
import os

# Configuration
DATA_FILE = "data/matches.csv"
OUTPUT_FILE = "data/features.csv"
ROLLING_WINDOW = 5

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please run data_loader.py first.")
    
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows from {DATA_FILE}")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Sort by Date
    df = df.sort_values('Date')
    
    return df

def calculate_team_stats(df):
    """
    Calculates rolling statistics for each team.
    We need to restructure the data to be team-centric first (one row per team per match)
    to easily calculate rolling means, then merge back.
    """
    # Create a team-centric dataframe
    home_df = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    home_df.rename(columns={'HomeTeam': 'Team', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded'}, inplace=True)
    home_df['IsHome'] = 1
    home_df['Points'] = home_df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    
    away_df = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR']].copy()
    away_df.rename(columns={'AwayTeam': 'Team', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded'}, inplace=True)
    away_df['IsHome'] = 0
    away_df['Points'] = away_df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    
    team_df = pd.concat([home_df, away_df]).sort_values(['Team', 'Date'])
    
    # Calculate rolling stats
    # Shift by 1 to ensure we only use PAST data for the current prediction
    team_df['AvgGoalsScored_Last5'] = team_df.groupby('Team')['GoalsScored'].transform(lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean())
    team_df['AvgGoalsConceded_Last5'] = team_df.groupby('Team')['GoalsConceded'].transform(lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean())
    team_df['AvgPoints_Last5'] = team_df.groupby('Team')['Points'].transform(lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean())
    
    return team_df

def merge_features(original_df, team_stats_df):
    """
    Merges calculated team stats back into the original match dataframe.
    """
    # Merge Home Team Stats
    home_stats = team_stats_df[team_stats_df['IsHome'] == 1][['Date', 'Team', 'AvgGoalsScored_Last5', 'AvgGoalsConceded_Last5', 'AvgPoints_Last5']]
    merged_df = pd.merge(original_df, home_stats, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    merged_df.rename(columns={
        'AvgGoalsScored_Last5': 'Home_AvgGoalsScored', 
        'AvgGoalsConceded_Last5': 'Home_AvgGoalsConceded',
        'AvgPoints_Last5': 'Home_AvgPoints'
    }, inplace=True)
    merged_df.drop(columns=['Team'], inplace=True)
    
    # Merge Away Team Stats
    away_stats = team_stats_df[team_stats_df['IsHome'] == 0][['Date', 'Team', 'AvgGoalsScored_Last5', 'AvgGoalsConceded_Last5', 'AvgPoints_Last5']]
    merged_df = pd.merge(merged_df, away_stats, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    merged_df.rename(columns={
        'AvgGoalsScored_Last5': 'Away_AvgGoalsScored', 
        'AvgGoalsConceded_Last5': 'Away_AvgGoalsConceded',
        'AvgPoints_Last5': 'Away_AvgPoints'
    }, inplace=True)
    merged_df.drop(columns=['Team'], inplace=True)
    
    return merged_df

def main():
    print("Starting feature engineering...")
    try:
        df = load_data()
        
        # We only really need a subset of columns for the model + the ones needed for feature calc
        # Basic columns
        needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if pd.Series(needed_cols).isin(df.columns).all():
             df_subset = df[needed_cols].copy()
        else:
             print("Warning: Missing some expected columns. Using available columns.")
             df_subset = df.copy()

        team_stats = calculate_team_stats(df_subset)
        features_df = merge_features(df_subset, team_stats)
        
        # Drop rows with NaN (first few games of the dataset where rolling stats are not available)
        # Or fill with 0 if we assume average performance
        # For better accuracy, dropping is safer to avoid misleading the model
        print(f"Dropping {features_df.isna().any(axis=1).sum()} rows with missing rolling data (start of dataset).")
        features_df.dropna(inplace=True)
        
        print(f"Saving {len(features_df)} rows to {OUTPUT_FILE}...")
        features_df.to_csv(OUTPUT_FILE, index=False)
        print("Feature engineering complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
