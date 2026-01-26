import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Configuration
FEATURES_FILE = "data/features.csv"
MODEL_DIR = "models"
TRAIN_SPLIT_DATE = "2023-08-01" # Train on seasons up to 22/23, Test on 23/24 onwards

def load_features():
    df = pd.read_csv(FEATURES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def train_xgboost(X_train, y_train_home, y_train_away, X_test, y_test_home, y_test_away):
    print("\nTRAINING XGBOOST MODELS...")
    
    # Train Home Goals Model
    xgb_home = xgb.XGBRegressor(objective='count:poisson', n_estimators=1000, learning_rate=0.01, max_depth=5)
    xgb_home.fit(X_train, y_train_home)
    preds_home = xgb_home.predict(X_test)
    mae_home = mean_absolute_error(y_test_home, preds_home)
    print(f"XGBoost Home Goals MAE: {mae_home:.4f}")
    
    # Train Away Goals Model
    xgb_away = xgb.XGBRegressor(objective='count:poisson', n_estimators=1000, learning_rate=0.01, max_depth=5)
    xgb_away.fit(X_train, y_train_away)
    preds_away = xgb_away.predict(X_test)
    mae_away = mean_absolute_error(y_test_away, preds_away)
    print(f"XGBoost Away Goals MAE: {mae_away:.4f}")
    
    return xgb_home, xgb_away

def train_poisson_glm(train_df, test_df):
    print("\nTRAINING POISSON GLM MODELS (Statsmodels)...")
    
    # Formula for Poisson Regression
    # Ensuring features match DataFrame columns. Need to handle categorical 'HomeTeam'/'AwayTeam' if used.
    # In features.csv we likely have rolling stats. We can treat Teams as random effects or fixed effects, 
    # but simplest is using the numeric features we built.
    
    formula_features = "Home_AvgGoalsScored + Home_AvgGoalsConceded + Home_AvgPoints + " \
                       "Away_AvgGoalsScored + Away_AvgGoalsConceded + Away_AvgPoints"
    
    # Train Home Goals
    try:
        poisson_home = smf.glm(formula=f"FTHG ~ {formula_features}", data=train_df, family=sm.families.Poisson()).fit()
        print(poisson_home.summary())
        preds_home = poisson_home.predict(test_df)
        mae_home = mean_absolute_error(test_df['FTHG'], preds_home)
        print(f"Poisson GLM Home Goals MAE: {mae_home:.4f}")
    except Exception as e:
        print(f"Error training Poisson Home: {e}")
        poisson_home = None

    # Train Away Goals
    try:
        poisson_away = smf.glm(formula=f"FTAG ~ {formula_features}", data=train_df, family=sm.families.Poisson()).fit()
        preds_away = poisson_away.predict(test_df)
        mae_away = mean_absolute_error(test_df['FTAG'], preds_away)
        print(f"Poisson GLM Away Goals MAE: {mae_away:.4f}")
    except Exception as e:
        print(f"Error training Poisson Away: {e}")
        poisson_away = None
        
    return poisson_home, poisson_away

def save_models(models):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    with open(os.path.join(MODEL_DIR, "models.pkl"), "wb") as f:
        pickle.dump(models, f)
    print(f"\nModels saved to {MODEL_DIR}/models.pkl")

def main():
    df = load_features()
    
    # Define features and targets
    feature_cols = [
        'Home_AvgGoalsScored', 'Home_AvgGoalsConceded', 'Home_AvgPoints',
        'Away_AvgGoalsScored', 'Away_AvgGoalsConceded', 'Away_AvgPoints'
    ]
    target_home = 'FTHG'
    target_away = 'FTAG'
    
    # Time-series Split
    train_df = df[df['Date'] < TRAIN_SPLIT_DATE].copy()
    test_df = df[df['Date'] >= TRAIN_SPLIT_DATE].copy()
    
    print(f"Train Set: {len(train_df)} games (Before {TRAIN_SPLIT_DATE})")
    print(f"Test Set: {len(test_df)} games (From {TRAIN_SPLIT_DATE} onwards)")
    
    X_train = train_df[feature_cols]
    y_train_home = train_df[target_home]
    y_train_away = train_df[target_away]
    
    X_test = test_df[feature_cols]
    y_test_home = test_df[target_home]
    y_test_away = test_df[target_away]
    
    # Train Models
    xgb_h, xgb_a = train_xgboost(X_train, y_train_home, y_train_away, X_test, y_test_home, y_test_away)
    glm_h, glm_a = train_poisson_glm(train_df, test_df)
    
    # Save all models
    models_dict = {
        'xgboost_home': xgb_h,
        'xgboost_away': xgb_a,
        'poisson_home': glm_h,
        'poisson_away': glm_a
    }
    save_models(models_dict)

if __name__ == "__main__":
    main()
