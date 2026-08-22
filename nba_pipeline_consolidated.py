"""
================================================================================
NBA GAME PREDICTION PIPELINE -- CONSOLIDATED (Phases 1-4)
================================================================================
Run this top-to-bottom in a fresh kernel. Everything below has already been
validated step-by-step in our session -- this file exists so a kernel
restart doesn't force you to hunt through scattered messages again.

SECTIONS:
  Phase 1  -- Data ingestion (LeagueGameLog -> game-level wide dataframe)
  Phase 2  -- Feature engineering (rolling form, ratings, rest, travel,
              streaks, home/away splits) -- all leakage-safe, season-aware
  Phase 3  -- Modeling (walk-forward CV, Moneyline/Spread/Totals, tuning)
  Phase 4  -- Live inference (fetch a date's schedule, predict)
================================================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
from itertools import product

from nba_api.stats.endpoints import leaguegamelog, scoreboardv2
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams as nba_teams_static

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, log_loss,
                              brier_score_loss, mean_absolute_error, mean_squared_error)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

# ==============================================================================
# PHASE 1: DATA INGESTION
# ==============================================================================

SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
SEASON_TYPE = 'Regular Season'
REQUEST_DELAY = 2.0
MAX_RETRIES = 5
API_TIMEOUT = 120

NBAStatsHTTP.headers.update({
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Accept': 'application/json, text/plain, */*',
})

TEAM_ID_TO_ABBR = {t['id']: t['abbreviation'] for t in nba_teams_static.get_teams()}
TEAM_ABBR_TO_ID = {t['abbreviation']: t['id'] for t in nba_teams_static.get_teams()}


def fetch_season_game_log(season: str, season_type: str = SEASON_TYPE) -> pd.DataFrame:
    """Fetches ALL team game logs for a season in one API call."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=season_type,
                player_or_team_abbreviation='T', timeout=API_TIMEOUT
            )
            df = log.get_data_frames()[0]
            df['SEASON'] = season
            time.sleep(REQUEST_DELAY)
            return df
        except Exception as e:
            wait = REQUEST_DELAY * (2 ** (attempt - 1))
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] {season} failed: {type(e).__name__}: {e} -> retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch season {season} after {MAX_RETRIES} attempts.")


def fetch_multi_season_logs(seasons: list) -> pd.DataFrame:
    all_logs = []
    for season in seasons:
        print(f"Fetching {season}...")
        season_df = fetch_season_game_log(season)
        print(f"  -> {len(season_df)} rows")
        all_logs.append(season_df)
    return pd.concat(all_logs, ignore_index=True)


def parse_matchup_column(df: pd.DataFrame) -> pd.DataFrame:
    """'BOS @ NYK' -> BOS away; 'BOS vs. NYK' -> BOS home."""
    df = df.copy()
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.', regex=False)
    df['OPPONENT_ABBREVIATION'] = np.where(
        df['IS_HOME'], df['MATCHUP'].str.split('vs. ').str[1], df['MATCHUP'].str.split('@ ').str[1]
    )
    return df


def build_game_level_df(long_df: pd.DataFrame) -> pd.DataFrame:
    """Reshapes team-level long data into one row per GAME_ID (HOME/AWAY paired)."""
    df = parse_matchup_column(long_df)
    home_df = df[df['IS_HOME']].copy()
    away_df = df[~df['IS_HOME']].copy()

    home_df = home_df.add_suffix('_HOME')
    away_df = away_df.add_suffix('_AWAY')

    home_df = home_df.rename(columns={'GAME_ID_HOME': 'GAME_ID', 'GAME_DATE_HOME': 'GAME_DATE', 'SEASON_HOME': 'SEASON'})
    away_df = away_df.rename(columns={'GAME_ID_AWAY': 'GAME_ID'})
    away_df = away_df.drop(columns=[c for c in away_df.columns if c.replace('_AWAY', '') in ['GAME_DATE', 'SEASON']])

    game_df = pd.merge(home_df, away_df, on='GAME_ID', how='inner')

    game_df['HOME_WIN'] = (game_df['WL_HOME'] == 'W').astype(int)
    game_df['POINT_DIFF'] = game_df['PTS_HOME'] - game_df['PTS_AWAY']
    game_df['TOTAL_PTS'] = game_df['PTS_HOME'] + game_df['PTS_AWAY']

    game_df['GAME_DATE'] = pd.to_datetime(game_df['GAME_DATE'])
    game_df = game_df.sort_values('GAME_DATE').reset_index(drop=True)
    return game_df


# ==============================================================================
# PHASE 2: FEATURE ENGINEERING
# ==============================================================================

def build_team_games_long(game_df: pd.DataFrame) -> pd.DataFrame:
    """One row per TEAM per GAME (both home and away perspectives stacked)."""
    base_cols = ['GAME_ID', 'GAME_DATE', 'SEASON']
    home_cols = [c for c in game_df.columns if c.endswith('_HOME')]
    stat_names = [c.replace('_HOME', '') for c in home_cols]

    home_rows = game_df[base_cols + home_cols].copy()
    home_rows.columns = base_cols + stat_names
    home_rows['IS_HOME'] = 1
    home_rows['OPPONENT'] = game_df['TEAM_ABBREVIATION_AWAY'].values
    home_rows['OPPONENT_PTS'] = game_df['PTS_AWAY'].values

    away_cols = [c for c in game_df.columns if c.endswith('_AWAY')]
    away_rows = game_df[base_cols + away_cols].copy()
    away_rows.columns = base_cols + [c.replace('_AWAY', '') for c in away_cols]
    away_rows['IS_HOME'] = 0
    away_rows['OPPONENT'] = game_df['TEAM_ABBREVIATION_HOME'].values
    away_rows['OPPONENT_PTS'] = game_df['PTS_HOME'].values

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games['WON'] = (team_games['WL'] == 'W').astype(int)
    team_games = team_games.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)
    return team_games


def add_rolling_features(team_games: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """Leakage-safe rolling averages, RESET PER SEASON (shift(1) before rolling)."""
    team_games = team_games.copy()
    roll_metrics = ['PTS', 'OPPONENT_PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                     'REB', 'AST', 'TOV', 'STL', 'BLK', 'WON']
    roll_metrics = [m for m in roll_metrics if m in team_games.columns]
    grouped = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'], group_keys=False)

    for window in windows:
        for metric in roll_metrics:
            team_games[f'{metric}_ROLL{window}'] = grouped[metric].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    return team_games


def add_rest_days(team_games: pd.DataFrame) -> pd.DataFrame:
    """Rest days since previous game, RESET PER SEASON. First game of season -> 3 (default)."""
    team_games = team_games.copy()
    team_games['PREV_GAME_DATE'] = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'])['GAME_DATE'].shift(1)
    team_games['REST_DAYS'] = (team_games['GAME_DATE'] - team_games['PREV_GAME_DATE']).dt.days
    team_games['REST_DAYS'] = team_games['REST_DAYS'].fillna(3)
    team_games['IS_BACK_TO_BACK'] = (team_games['REST_DAYS'] <= 1).astype(int)
    return team_games.drop(columns=['PREV_GAME_DATE'])


def add_possessions_and_ratings(team_games: pd.DataFrame) -> pd.DataFrame:
    """POSSESSIONS = FGA - OREB + TOV + 0.4*FTA; OFF/DEF_RATING per 100 possessions."""
    team_games = team_games.copy()
    required = ['FGA', 'OREB', 'TOV', 'FTA', 'PTS', 'OPPONENT_PTS']
    missing = [c for c in required if c not in team_games.columns]
    if missing:
        raise ValueError(f"Missing required columns for possession estimate: {missing}")

    team_games['POSSESSIONS'] = team_games['FGA'] - team_games['OREB'] + team_games['TOV'] + (0.4 * team_games['FTA'])
    safe_poss = team_games['POSSESSIONS'].replace(0, np.nan)
    team_games['OFF_RATING'] = 100 * team_games['PTS'] / safe_poss
    team_games['DEF_RATING'] = 100 * team_games['OPPONENT_PTS'] / safe_poss
    return team_games


def add_rating_rolling_features(team_games: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """Rolling OFF/DEF_RATING/POSSESSIONS, leakage-safe, reset per season."""
    team_games = team_games.copy()
    grouped = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'], group_keys=False)
    for window in windows:
        for metric in ['OFF_RATING', 'DEF_RATING', 'POSSESSIONS']:
            team_games[f'{metric}_ROLL{window}'] = grouped[metric].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    return team_games


ARENA_LOCATIONS = {
    'ATL': (33.7573, -84.3963), 'BOS': (42.3662, -71.0621), 'BKN': (40.6826, -73.9754),
    'CHA': (35.2251, -80.8392), 'CHI': (41.8807, -87.6742), 'CLE': (41.4965, -81.6882),
    'DAL': (32.7905, -96.8103), 'DEN': (39.7487, -105.0077), 'DET': (42.3410, -83.0550),
    'GSW': (37.7680, -122.3877), 'HOU': (29.7508, -95.3621), 'IND': (39.7640, -86.1555),
    'LAC': (34.0430, -118.2673), 'LAL': (34.0430, -118.2673), 'MEM': (35.1382, -90.0505),
    'MIA': (25.7814, -80.1870), 'MIL': (43.0451, -87.9172), 'MIN': (44.9795, -93.2760),
    'NOP': (29.9490, -90.0821), 'NYK': (40.7505, -73.9934), 'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3839), 'PHI': (39.9012, -75.1720), 'PHX': (33.4457, -112.0712),
    'POR': (45.5316, -122.6668), 'SAC': (38.5802, -121.4997), 'SAS': (29.4269, -98.4375),
    'TOR': (43.6435, -79.3791), 'UTA': (40.7683, -111.9011), 'WAS': (38.8981, -77.0209),
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles. Vectorized."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_travel_distance(team_games: pd.DataFrame) -> pd.DataFrame:
    """Distance traveled since previous game, leakage-safe, reset per season."""
    team_games = team_games.copy()

    def get_game_location(row):
        team = row['TEAM_ABBREVIATION'] if row['IS_HOME'] == 1 else row['OPPONENT']
        return ARENA_LOCATIONS.get(team, (np.nan, np.nan))

    locations = team_games.apply(get_game_location, axis=1)
    team_games['GAME_LAT'] = [loc[0] for loc in locations]
    team_games['GAME_LON'] = [loc[1] for loc in locations]

    grouped = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'])
    team_games['PREV_LAT'] = grouped['GAME_LAT'].shift(1)
    team_games['PREV_LON'] = grouped['GAME_LON'].shift(1)

    team_games['TRAVEL_DISTANCE'] = haversine_distance(
        team_games['PREV_LAT'], team_games['PREV_LON'], team_games['GAME_LAT'], team_games['GAME_LON']
    )
    team_games['TRAVEL_DISTANCE'] = team_games['TRAVEL_DISTANCE'].fillna(0)
    return team_games.drop(columns=['GAME_LAT', 'GAME_LON', 'PREV_LAT', 'PREV_LON'])


def add_streak_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Signed win/loss streak length going INTO each game (excludes that game's own result)."""
    team_games = team_games.copy()
    team_games = team_games.sort_values(['TEAM_ABBREVIATION', 'SEASON', 'GAME_DATE']).reset_index(drop=True)

    def compute_streak(group):
        prior_won = group['WON'].shift(1)
        change = (prior_won != prior_won.shift(1)).cumsum()
        streak_len = prior_won.groupby(change).cumcount() + 1
        signed = np.where(prior_won == 1, streak_len, np.where(prior_won == 0, -streak_len, np.nan))
        return pd.Series(signed, index=group.index)

    team_games['STREAK'] = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON'], group_keys=False).apply(compute_streak)
    team_games['STREAK'] = team_games['STREAK'].fillna(0)
    return team_games


def add_home_away_split_form(team_games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling form computed separately for home games vs away games."""
    team_games = team_games.copy()
    split_metrics = ['PTS', 'OPPONENT_PTS', 'WON']
    grouped = team_games.groupby(['TEAM_ABBREVIATION', 'SEASON', 'IS_HOME'], group_keys=False)
    for metric in split_metrics:
        team_games[f'{metric}_ROLL{window}_SPLIT'] = grouped[metric].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    return team_games


def merge_features_to_game_level(game_df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Joins engineered team_games features back onto game_df as HOME_*/AWAY_* columns."""
    feature_cols_ = [c for c in team_games.columns if
                      'ROLL' in c or 'STREAK' in c or 'TRAVEL' in c or c in [
                          'REST_DAYS', 'IS_BACK_TO_BACK', 'OFF_RATING', 'DEF_RATING', 'POSSESSIONS']]

    merge_keys = ['GAME_ID', 'TEAM_ABBREVIATION']
    feature_slice = team_games[merge_keys + feature_cols_].copy()

    home_features = feature_slice.add_suffix('_HOME')
    home_features = home_features.rename(columns={'GAME_ID_HOME': 'GAME_ID', 'TEAM_ABBREVIATION_HOME': 'TEAM_ABBREVIATION_HOME'})
    merged = pd.merge(game_df, home_features, on=['GAME_ID', 'TEAM_ABBREVIATION_HOME'], how='left')

    away_features = feature_slice.add_suffix('_AWAY')
    away_features = away_features.rename(columns={'GAME_ID_AWAY': 'GAME_ID', 'TEAM_ABBREVIATION_AWAY': 'TEAM_ABBREVIATION_AWAY'})
    merged = pd.merge(merged, away_features, on=['GAME_ID', 'TEAM_ABBREVIATION_AWAY'], how='left')

    return merged


def run_full_feature_pipeline(game_level_df: pd.DataFrame) -> tuple:
    """Runs the entire Phase 2 sequence in the correct order. Returns (team_games, model_ready_df)."""
    tg = build_team_games_long(game_level_df)
    tg = add_rolling_features(tg, windows=(5, 10))
    tg = add_rest_days(tg)
    tg = add_possessions_and_ratings(tg)
    tg = add_rating_rolling_features(tg, windows=(5, 10))
    tg = add_travel_distance(tg)
    tg = add_streak_features(tg)
    tg = add_home_away_split_form(tg, window=10)

    model_ready = merge_features_to_game_level(game_level_df, tg)
    return tg, model_ready


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with incomplete rolling features (early-season games)."""
    df = df.copy()
    critical_cols = ['OFF_RATING_ROLL5_HOME', 'DEF_RATING_ROLL5_HOME',
                      'OFF_RATING_ROLL5_AWAY', 'DEF_RATING_ROLL5_AWAY']
    before = len(df)
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with incomplete rolling features ({before} -> {len(df)})")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Whitelist: only leakage-safe, pre-game-known feature columns."""
    allowed_patterns = ['_ROLL5', '_ROLL10', 'REST_DAYS', 'IS_BACK_TO_BACK',
                         'STREAK', 'TRAVEL_DISTANCE', '_SPLIT', 'MISSING_PLAYERS_COUNT',
                         'MISSING_MINUTES_SHARE', 'MISSING_PTS_IMPACT']
    feature_cols_ = [c for c in df.columns if any(p in c for p in allowed_patterns)]
    print(f"Selected {len(feature_cols_)} leakage-safe feature columns.")
    return feature_cols_


def walk_forward_season_splits(df: pd.DataFrame, season_col: str = 'SEASON'):
    """Yields (train_idx, test_idx, test_season, train_seasons) -- expanding window by season."""
    seasons_sorted = sorted(df[season_col].unique())
    for i in range(1, len(seasons_sorted)):
        train_seasons = seasons_sorted[:i]
        test_season = seasons_sorted[i]
        train_idx = df[df[season_col].isin(train_seasons)].index
        test_idx = df[df[season_col] == test_season].index
        yield train_idx, test_idx, test_season, train_seasons


# ==============================================================================
# PHASE 3: MODELING & BACKTESTING
# ==============================================================================

def run_walk_forward_moneyline(df: pd.DataFrame, feature_cols_: list, target_col: str = 'HOME_WIN'):
    """Walk-forward LogReg + XGBoost classification, with naive baseline."""
    results = []
    for train_idx, test_idx, test_season, train_seasons in walk_forward_season_splits(df):
        X_train, y_train = df.loc[train_idx, feature_cols_], df.loc[train_idx, target_col]
        X_test, y_test = df.loc[test_idx, feature_cols_], df.loc[test_idx, target_col]

        train_medians = X_train.median()
        X_train, X_test = X_train.fillna(train_medians), X_test.fillna(train_medians)

        naive_acc = accuracy_score(y_test, np.ones(len(y_test)))

        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
        logreg = LogisticRegression(max_iter=1000, C=1.0)
        logreg.fit(X_train_s, y_train)
        logreg_proba = logreg.predict_proba(X_test_s)[:, 1]

        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]

        fold_result = {
            'test_season': test_season, 'naive_accuracy': naive_acc,
            'logreg_accuracy': accuracy_score(y_test, (logreg_proba >= 0.5).astype(int)),
            'logreg_logloss': log_loss(y_test, logreg_proba),
            'xgb_accuracy': accuracy_score(y_test, (xgb_proba >= 0.5).astype(int)),
            'xgb_logloss': log_loss(y_test, xgb_proba),
        }
        results.append(fold_result)
        print(f"Season {test_season}: naive={naive_acc:.3f}  logreg={fold_result['logreg_accuracy']:.3f}  xgb={fold_result['xgb_accuracy']:.3f}")

    return pd.DataFrame(results)


def tune_xgboost_walk_forward(df: pd.DataFrame, feature_cols_: list, target_col: str = 'HOME_WIN', holdout_season: str = '2024-25'):
    """Grid search excluding holdout_season entirely (never touches final test season)."""
    param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.03, 0.05, 0.08],
                  'n_estimators': [150, 250], 'min_child_weight': [1, 5]}
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    tune_df = df[df['SEASON'] != holdout_season].reset_index(drop=True)

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        accs, lls = [], []
        for train_idx, test_idx, test_season, train_seasons in walk_forward_season_splits(tune_df):
            X_train, y_train = tune_df.loc[train_idx, feature_cols_], tune_df.loc[train_idx, target_col]
            X_test, y_test = tune_df.loc[test_idx, feature_cols_], tune_df.loc[test_idx, target_col]
            train_medians = X_train.median()
            X_train, X_test = X_train.fillna(train_medians), X_test.fillna(train_medians)

            model = XGBClassifier(**params, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            lls.append(log_loss(y_test, proba))
            accs.append(accuracy_score(y_test, (proba >= 0.5).astype(int)))

        results.append({**params, 'avg_logloss': np.mean(lls), 'avg_accuracy': np.mean(accs), 'std_accuracy': np.std(accs)})

    return pd.DataFrame(results).sort_values('avg_logloss')


def final_holdout_evaluation(df: pd.DataFrame, feature_cols_: list, best_params: dict,
                               target_col: str = 'HOME_WIN', holdout_season: str = '2024-25'):
    """Trains on all non-holdout seasons, evaluates ONCE on the untouched holdout season."""
    train_df = df[df['SEASON'] != holdout_season]
    test_df = df[df['SEASON'] == holdout_season]
    X_train, y_train = train_df[feature_cols_], train_df[target_col]
    X_test, y_test = test_df[feature_cols_], test_df[target_col]

    train_medians = X_train.median()
    X_train, X_test = X_train.fillna(train_medians), X_test.fillna(train_medians)

    model = XGBClassifier(**best_params, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print(f"=== FINAL HOLDOUT EVALUATION: {holdout_season} ===")
    print(f"Naive baseline accuracy: {accuracy_score(y_test, np.ones(len(y_test))):.3f}")
    print(f"Tuned XGBoost accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Tuned XGBoost log loss:  {log_loss(y_test, proba):.3f}")
    print(f"Tuned XGBoost precision: {precision_score(y_test, preds):.3f}")
    print(f"Tuned XGBoost recall:    {recall_score(y_test, preds):.3f}")
    print(f"Tuned XGBoost Brier:     {brier_score_loss(y_test, proba):.3f}")

    return model, X_test, y_test, proba


def run_walk_forward_spread(df: pd.DataFrame, feature_cols_: list, target_col: str = 'POINT_DIFF'):
    """Walk-forward LinReg + XGBoost regression for point spread."""
    results = []
    for train_idx, test_idx, test_season, train_seasons in walk_forward_season_splits(df):
        X_train, y_train = df.loc[train_idx, feature_cols_], df.loc[train_idx, target_col]
        X_test, y_test = df.loc[test_idx, feature_cols_], df.loc[test_idx, target_col]
        train_medians = X_train.median()
        X_train, X_test = X_train.fillna(train_medians), X_test.fillna(train_medians)

        naive_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))

        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
        linreg = LinearRegression().fit(X_train_s, y_train)
        linreg_preds = linreg.predict(X_test_s)

        xgb_reg = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.03,
                                min_child_weight=1, subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb_reg.fit(X_train, y_train)
        xgb_preds = xgb_reg.predict(X_test)

        actual_home_win = (y_test > 0).astype(int)
        fold_result = {
            'test_season': test_season, 'naive_mae': naive_mae,
            'linreg_mae': mean_absolute_error(y_test, linreg_preds),
            'linreg_direction_acc': accuracy_score(actual_home_win, (linreg_preds > 0).astype(int)),
            'xgb_mae': mean_absolute_error(y_test, xgb_preds),
            'xgb_direction_acc': accuracy_score(actual_home_win, (xgb_preds > 0).astype(int)),
        }
        results.append(fold_result)
        print(f"Season {test_season}: naive_mae={naive_mae:.2f}  linreg_mae={fold_result['linreg_mae']:.2f}  xgb_mae={fold_result['xgb_mae']:.2f}")

    return pd.DataFrame(results)


def run_walk_forward_totals(df: pd.DataFrame, feature_cols_: list, target_col: str = 'TOTAL_PTS'):
    """Walk-forward LinReg + XGBoost regression for game totals."""
    results = []
    for train_idx, test_idx, test_season, train_seasons in walk_forward_season_splits(df):
        X_train, y_train = df.loc[train_idx, feature_cols_], df.loc[train_idx, target_col]
        X_test, y_test = df.loc[test_idx, feature_cols_], df.loc[test_idx, target_col]
        train_medians = X_train.median()
        X_train, X_test = X_train.fillna(train_medians), X_test.fillna(train_medians)

        naive_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))

        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
        linreg = LinearRegression().fit(X_train_s, y_train)
        linreg_preds = linreg.predict(X_test_s)

        xgb_reg = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.03,
                                min_child_weight=1, subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb_reg.fit(X_train, y_train)
        xgb_preds = xgb_reg.predict(X_test)

        fold_result = {
            'test_season': test_season, 'naive_mae': naive_mae,
            'linreg_mae': mean_absolute_error(y_test, linreg_preds),
            'xgb_mae': mean_absolute_error(y_test, xgb_preds),
        }
        results.append(fold_result)
        print(f"Season {test_season}: naive_mae={naive_mae:.2f}  linreg_mae={fold_result['linreg_mae']:.2f}  xgb_mae={fold_result['xgb_mae']:.2f}")

    return pd.DataFrame(results)


# ==============================================================================
# PHASE 4: LIVE INFERENCE
# ==============================================================================

def fetch_schedule_for_date(game_date: str) -> pd.DataFrame:
    """Fetches scheduled games for a given date (format: 'YYYY-MM-DD')."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=game_date, timeout=API_TIMEOUT)
            games = sb.get_data_frames()[0]
            time.sleep(REQUEST_DELAY)
            games['HOME_TEAM_ABBR'] = games['HOME_TEAM_ID'].map(TEAM_ID_TO_ABBR)
            games['AWAY_TEAM_ABBR'] = games['VISITOR_TEAM_ID'].map(TEAM_ID_TO_ABBR)
            return games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ABBR', 'AWAY_TEAM_ABBR']]
        except Exception as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Schedule fetch failed: {e}")
            time.sleep(REQUEST_DELAY * (2 ** (attempt - 1)))
    raise RuntimeError(f"Failed to fetch schedule for {game_date}")


def fetch_live_team_games(season: str, as_of_date: str) -> pd.DataFrame:
    """
    Fetches the full season (ONE API call for all teams), reuses the exact
    same Phase 2 functions to get correct OPPONENT_PTS/ratings/travel/streaks,
    then filters to games strictly before as_of_date (leakage-safe).
    """
    season_long_df = fetch_season_game_log(season)
    season_game_level = build_game_level_df(season_long_df)
    tg = build_team_games_long(season_game_level)
    tg = add_possessions_and_ratings(tg)
    tg = add_travel_distance(tg)
    tg = add_streak_features(tg)

    cutoff = pd.to_datetime(as_of_date)
    tg = tg[tg['GAME_DATE'] < cutoff]
    return tg.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])


def compute_live_rolling_features(team_recent: pd.DataFrame, windows=(5, 10)) -> dict:
    """Computes rolling features (incl. ratings) from a team's pre-filtered game slice."""
    features = {}
    team_recent = team_recent.sort_values('GAME_DATE', ascending=False)

    roll_metrics = ['PTS', 'OPPONENT_PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST',
                     'TOV', 'STL', 'BLK', 'WON', 'OFF_RATING', 'DEF_RATING', 'POSSESSIONS']

    for window in windows:
        window_games = team_recent.head(window)
        for metric in roll_metrics:
            features[f'{metric}_ROLL{window}'] = (
                window_games[metric].mean() if len(window_games) > 0 and metric in window_games.columns else np.nan
            )

    # Home/away split (10-game window only, matching training)
    for is_home_val, tag in [(1, 'HOME'), (0, 'AWAY')]:
        split_games = team_recent[team_recent['IS_HOME'] == is_home_val].head(10)
        for metric in ['PTS', 'OPPONENT_PTS', 'WON']:
            features[f'_SPLIT_{tag}_{metric}'] = split_games[metric].mean() if len(split_games) > 0 else np.nan

    # Streak: most recent games, walked backward
    if len(team_recent) > 0:
        results_arr = team_recent['WON'].values
        streak_len, streak_type = 0, results_arr[0]
        for r in results_arr:
            if r == streak_type:
                streak_len += 1
            else:
                break
        features['STREAK'] = float(streak_len if streak_type == 1 else -streak_len)
        last_game = team_recent.iloc[0]
        features['_LAST_LOCATION_TEAM'] = last_game['TEAM_ABBREVIATION'] if last_game['IS_HOME'] == 1 else last_game['OPPONENT']
    else:
        features['STREAK'] = 0.0
        features['_LAST_LOCATION_TEAM'] = None

    features['_LAST_GAME_DATE'] = team_recent['GAME_DATE'].max() if len(team_recent) > 0 else None
    return features


def build_live_game_features(schedule: pd.DataFrame, as_of_date: str, season: str) -> pd.DataFrame:
    """Assembles the full HOME/AWAY feature row for every scheduled game."""
    print(f"Fetching full season data for {season} (single API call)...")
    season_team_games = fetch_live_team_games(season, as_of_date)
    cutoff = pd.to_datetime(as_of_date)

    all_rows = []
    for _, game in schedule.iterrows():
        home_abbr, away_abbr = game['HOME_TEAM_ABBR'], game['AWAY_TEAM_ABBR']
        home_recent = season_team_games[season_team_games['TEAM_ABBREVIATION'] == home_abbr]
        away_recent = season_team_games[season_team_games['TEAM_ABBREVIATION'] == away_abbr]

        home_feats = compute_live_rolling_features(home_recent)
        away_feats = compute_live_rolling_features(away_recent)

        def rest_days_from(last_date):
            return 3.0 if last_date is None or pd.isna(last_date) else float((cutoff - last_date).days)

        home_rest = rest_days_from(home_feats.pop('_LAST_GAME_DATE'))
        away_rest = rest_days_from(away_feats.pop('_LAST_GAME_DATE'))
        home_last_loc = home_feats.pop('_LAST_LOCATION_TEAM')
        away_last_loc = away_feats.pop('_LAST_LOCATION_TEAM')

        def travel_to(last_loc_team, dest_team):
            if last_loc_team is None or last_loc_team not in ARENA_LOCATIONS or dest_team not in ARENA_LOCATIONS:
                return 0.0
            lat1, lon1 = ARENA_LOCATIONS[last_loc_team]
            lat2, lon2 = ARENA_LOCATIONS[dest_team]
            return float(haversine_distance(lat1, lon1, lat2, lon2))

        row = {'GAME_ID': game['GAME_ID'], 'HOME_TEAM': home_abbr, 'AWAY_TEAM': away_abbr}

        for k, v in home_feats.items():
            if k.startswith('_SPLIT_HOME_'):
                row[k.replace('_SPLIT_HOME_', '') + '_ROLL10_SPLIT_HOME'] = v
            elif k.startswith('_SPLIT_AWAY_'):
                pass
            else:
                row[f'{k}_HOME'] = v

        for k, v in away_feats.items():
            if k.startswith('_SPLIT_AWAY_'):
                row[k.replace('_SPLIT_AWAY_', '') + '_ROLL10_SPLIT_AWAY'] = v
            elif k.startswith('_SPLIT_HOME_'):
                pass
            else:
                row[f'{k}_AWAY'] = v

        row['REST_DAYS_HOME'], row['REST_DAYS_AWAY'] = home_rest, away_rest
        row['IS_BACK_TO_BACK_HOME'] = int(home_rest <= 1)
        row['IS_BACK_TO_BACK_AWAY'] = int(away_rest <= 1)
        row['TRAVEL_DISTANCE_HOME'] = travel_to(home_last_loc, home_abbr)
        row['TRAVEL_DISTANCE_AWAY'] = travel_to(away_last_loc, home_abbr)

        all_rows.append(row)

    return pd.DataFrame(all_rows)


def generate_predictions(live_features: pd.DataFrame, feature_cols_: list, train_medians: pd.Series,
                          moneyline_model, spread_model=None, totals_model=None) -> pd.DataFrame:
    """Applies trained models to live features. Missing/absent features -> training median fill."""
    X_live = pd.DataFrame(index=live_features.index)
    missing_cols = []
    for col in feature_cols_:
        if col in live_features.columns:
            X_live[col] = live_features[col]
        else:
            X_live[col] = np.nan
            missing_cols.append(col)

    if missing_cols:
        print(f"NOTE: {len(missing_cols)} training features not available live (median-filled): {missing_cols}")

    X_live = X_live.fillna(train_medians)

    results = live_features[['GAME_ID', 'HOME_TEAM', 'AWAY_TEAM']].copy()
    results['HOME_WIN_PROB'] = moneyline_model.predict_proba(X_live)[:, 1]
    results['PREDICTED_WINNER'] = np.where(results['HOME_WIN_PROB'] >= 0.5, results['HOME_TEAM'], results['AWAY_TEAM'])

    if spread_model is not None:
        results['PREDICTED_SPREAD'] = spread_model.predict(X_live)
    if totals_model is not None:
        results['PREDICTED_TOTAL'] = totals_model.predict(X_live)

    return results


# ==============================================================================
# EXECUTION -- run this section to build everything from scratch
# ==============================================================================

if __name__ == '__main__':
    # --- Phase 1 ---
    raw_long_df = fetch_multi_season_logs(SEASONS)
    game_level_df = build_game_level_df(raw_long_df)
    game_level_df.to_csv('nba_games_raw.csv', index=False)
    print(f"Phase 1 done: {len(game_level_df)} games")

    # --- Phase 2 ---
    team_games, model_ready_df = run_full_feature_pipeline(game_level_df)
    model_ready_df.to_csv('nba_model_ready.csv', index=False)
    print(f"Phase 2 done: {model_ready_df.shape}")

    # --- Phase 3 ---
    model_df = prepare_modeling_data(model_ready_df)
    feature_cols = get_feature_columns(model_df)

    print("\n--- Moneyline (untuned baseline) ---")
    moneyline_results = run_walk_forward_moneyline(model_df, feature_cols)

    print("\n--- Tuning XGBoost (excludes 2024-25 holdout) ---")
    tuning_results = tune_xgboost_walk_forward(model_df, feature_cols)
    best_params = tuning_results.iloc[0][['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight']].to_dict()
    # Only cast the genuinely integer hyperparameters -- learning_rate MUST
    # stay a float. Blanket int()-casting every value here previously
    # truncated learning_rate (e.g. 0.03) down to 0, which silently produces
    # a degenerate model that outputs a constant prediction for every input.
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    print(f"Best params: {best_params}")

    print("\n--- Final holdout evaluation (2024-25, untouched) ---")
    final_model, X_test_final, y_test_final, proba_final = final_holdout_evaluation(model_df, feature_cols, best_params)

    print("\n--- Spread model ---")
    spread_results = run_walk_forward_spread(model_df, feature_cols)

    print("\n--- Totals model ---")
    totals_results = run_walk_forward_totals(model_df, feature_cols)

    # Train final spread/totals models on all non-holdout data for live use
    train_df = model_df[model_df['SEASON'] != '2024-25']
    X_train_full = train_df[feature_cols].fillna(train_df[feature_cols].median())
    spread_model_final = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.03,
                                       min_child_weight=1, subsample=0.8, colsample_bytree=0.8, random_state=42)
    spread_model_final.fit(X_train_full, train_df['POINT_DIFF'])
    totals_model_final = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.03,
                                       min_child_weight=1, subsample=0.8, colsample_bytree=0.8, random_state=42)
    totals_model_final.fit(X_train_full, train_df['TOTAL_PTS'])

    train_medians_full = model_df[feature_cols].median()

    # --- Phase 4 (test against a past date) ---
    TEST_DATE = '2024-01-15'
    TEST_SEASON = '2023-24'
    schedule = fetch_schedule_for_date(TEST_DATE)
    live_features = build_live_game_features(schedule, as_of_date=TEST_DATE, season=TEST_SEASON)
    predictions = generate_predictions(live_features, feature_cols, train_medians_full,
                                        moneyline_model=final_model,
                                        spread_model=spread_model_final,
                                        totals_model=totals_model_final)
    print("\n=== Predictions ===")
    print(predictions.to_string(index=False))


# ==============================================================================
# PHASE 1b: MARKET DATA INTEGRATION (closing lines, free Kaggle dataset)
# ==============================================================================

ODDS_TEAM_MAP = {
    'atl': 'ATL', 'bkn': 'BKN', 'bos': 'BOS', 'cha': 'CHA', 'chi': 'CHI',
    'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gs': 'GSW',
    'hou': 'HOU', 'ind': 'IND', 'lac': 'LAC', 'lal': 'LAL', 'mem': 'MEM',
    'mia': 'MIA', 'mil': 'MIL', 'min': 'MIN', 'no': 'NOP', 'ny': 'NYK',
    'okc': 'OKC', 'orl': 'ORL', 'phi': 'PHI', 'phx': 'PHX', 'por': 'POR',
    'sa': 'SAS', 'sac': 'SAC', 'tor': 'TOR', 'utah': 'UTA', 'wsh': 'WAS',
}


def american_odds_to_prob(ml):
    """Converts American moneyline odds to raw (vig-included) implied probability."""
    if pd.isna(ml):
        return np.nan
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def load_market_odds(path: str) -> pd.DataFrame:
    """
    Loads closing-line market data (spread, total, moneyline) from the free
    Kaggle dataset. Maps team codes to standard abbreviations, builds a
    signed home-perspective spread matching our POINT_DIFF convention
    (positive = home favored), and computes de-vigged implied win
    probabilities from moneylines.

    KNOWN LIMITATION: moneyline coverage in this free source is complete
    through the 2021-22 season, ~50% for 2022-23, and ABSENT for 2023-24
    onward. Spread and Total coverage is complete across all seasons.
    Moneyline-dependent features/evaluations will have NaN for recent
    seasons -- this is a genuine free-data gap, not a bug.
    """
    df = pd.read_csv(path)

    df['HOME_ABBR'] = df['home'].map(ODDS_TEAM_MAP)
    df['AWAY_ABBR'] = df['away'].map(ODDS_TEAM_MAP)

    unmapped = df[df['HOME_ABBR'].isna() | df['AWAY_ABBR'].isna()]
    if len(unmapped) > 0:
        print(f"WARNING: {len(unmapped)} rows have unmapped team codes -- dropping them.")
        df = df.dropna(subset=['HOME_ABBR', 'AWAY_ABBR'])

    df['CLOSING_SPREAD_HOME'] = np.where(df['whos_favored'] == 'home', df['spread'], -df['spread'])
    df['CLOSING_TOTAL'] = df['total']

    raw_prob_home = df['moneyline_home'].apply(american_odds_to_prob)
    raw_prob_away = df['moneyline_away'].apply(american_odds_to_prob)
    vig_sum = raw_prob_home + raw_prob_away
    df['IMPLIED_WIN_PROB_HOME'] = raw_prob_home / vig_sum
    df['IMPLIED_WIN_PROB_AWAY'] = raw_prob_away / vig_sum

    df['GAME_DATE'] = pd.to_datetime(df['date'])

    keep_cols = ['GAME_DATE', 'HOME_ABBR', 'AWAY_ABBR', 'CLOSING_SPREAD_HOME', 'CLOSING_TOTAL',
                 'moneyline_home', 'moneyline_away', 'IMPLIED_WIN_PROB_HOME', 'IMPLIED_WIN_PROB_AWAY']
    result = df[keep_cols].rename(columns={'moneyline_home': 'MONEYLINE_HOME', 'moneyline_away': 'MONEYLINE_AWAY'})

    # Guard against duplicate GAME_DATE+HOME+AWAY combos (e.g. rare doubleheaders
    # or data entry dupes) which would silently fan out rows during merge
    dupes = result.duplicated(subset=['GAME_DATE', 'HOME_ABBR', 'AWAY_ABBR'], keep=False)
    if dupes.sum() > 0:
        print(f"WARNING: {dupes.sum()} duplicate GAME_DATE+HOME+AWAY combos found -- keeping first occurrence.")
        result = result.drop_duplicates(subset=['GAME_DATE', 'HOME_ABBR', 'AWAY_ABBR'], keep='first')

    return result


def merge_market_data(game_level_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges closing-line market data onto our game-level dataframe, keyed on
    GAME_DATE + home/away team abbreviations. LEFT join -- games without a
    market data match (e.g. outside the odds dataset's coverage) keep NaN
    for market columns rather than being dropped.
    """
    merged = pd.merge(
        game_level_df, market_df,
        left_on=['GAME_DATE', 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY'],
        right_on=['GAME_DATE', 'HOME_ABBR', 'AWAY_ABBR'],
        how='left'
    )
    merged = merged.drop(columns=['HOME_ABBR', 'AWAY_ABBR'])

    match_rate = merged['CLOSING_SPREAD_HOME'].notna().mean()
    print(f"Market data match rate: {match_rate:.1%} ({merged['CLOSING_SPREAD_HOME'].notna().sum()} / {len(merged)} games)")

    return merged


# ==============================================================================
# PHASE 2g: MARKET-RELATIVE TARGETS & FEATURES
# ==============================================================================

def add_market_residual_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Spread/Totals targets into MARKET RESIDUALS -- what the
    model needs to predict is no longer "how many points will the home team
    win/lose by" but "how far off is the closing line from what actually
    happened." A residual of 0 means the market was exactly right; our new
    baseline (Phase 3) is "always predict residual=0" (i.e. trust the market),
    which is a MUCH harder baseline to beat than the old naive-average one.

    NOTE: this does NOT replace POINT_DIFF/TOTAL_PTS (kept for reference/
    backward compatibility) -- it ADDS the residual columns as new targets.
    """
    df = df.copy()
    df['SPREAD_RESIDUAL'] = df['POINT_DIFF'] - df['CLOSING_SPREAD_HOME']
    df['TOTALS_RESIDUAL'] = df['TOTAL_PTS'] - df['CLOSING_TOTAL']
    return df



# ==============================================================================
# PHASE 3h: PROBABILITY CALIBRATION (Isotonic Regression)
# ==============================================================================

from sklearn.isotonic import IsotonicRegression


def fit_isotonic_calibrator(raw_proba_calib: np.ndarray, y_calib: np.ndarray) -> IsotonicRegression:
    """
    Fits an isotonic regression mapping RAW model probabilities -> empirical
    win rates. MUST be fit on a calibration set the model wasn't trained on
    (e.g. a held-out prior season) -- fitting on the training set itself
    would just re-learn the model's own (possibly overconfident) curve.
    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_proba_calib, y_calib)
    return calibrator


def apply_calibration(calibrator: IsotonicRegression, raw_proba: np.ndarray) -> np.ndarray:
    """Applies a fitted calibrator to new raw probabilities."""
    return calibrator.predict(raw_proba)


# ==============================================================================
# PHASE 3i: MARKET-RELATIVE EVALUATION (beat the market, not just naive)
# ==============================================================================

def evaluate_vs_market(y_true: pd.Series, our_proba: np.ndarray, market_implied_prob: pd.Series) -> dict:
    """
    Compares OUR calibrated probabilities against the MARKET's own implied
    win probability on the SAME games. This is the real test of whether the
    model adds value -- beating a naive 'home always wins' baseline is easy;
    beating the market's own pricing is the actual bar for a system that
    could theoretically find value.

    Only evaluates on rows where market_implied_prob is available (recall:
    moneyline coverage in our free dataset ends at 2021-22) -- silently
    dropping NaN rows here would be misleading, so we report the usable
    sample size explicitly.
    """
    valid = market_implied_prob.notna() & y_true.notna()
    n_valid = valid.sum()

    if n_valid == 0:
        print("WARNING: no rows with market implied probability available -- cannot evaluate vs market.")
        return {}

    y_valid = y_true[valid].values
    our_valid = our_proba[valid.values] if isinstance(our_proba, np.ndarray) else our_proba[valid]
    market_valid = market_implied_prob[valid].values

    result = {
        'n_games': int(n_valid),
        'our_logloss': log_loss(y_valid, our_valid),
        'market_logloss': log_loss(y_valid, market_valid),
        'our_brier': brier_score_loss(y_valid, our_valid),
        'market_brier': brier_score_loss(y_valid, market_valid),
    }
    result['logloss_edge'] = result['market_logloss'] - result['our_logloss']  # positive = we beat market
    result['brier_edge'] = result['market_brier'] - result['our_brier']

    print(f"=== Model vs Market ({n_valid} games with odds available) ===")
    print(f"Log Loss  -- Ours: {result['our_logloss']:.4f}  Market: {result['market_logloss']:.4f}  "
          f"({'WE BEAT MARKET' if result['logloss_edge'] > 0 else 'market beat us'} by {abs(result['logloss_edge']):.4f})")
    print(f"Brier     -- Ours: {result['our_brier']:.4f}  Market: {result['market_brier']:.4f}  "
          f"({'WE BEAT MARKET' if result['brier_edge'] > 0 else 'market beat us'} by {abs(result['brier_edge']):.4f})")

    return result


# ==============================================================================
# PHASE 3j: RESIDUAL-TARGET WALK-FORWARD (Spread/Totals vs the market itself)
# ==============================================================================

def run_walk_forward_residual(df: pd.DataFrame, feature_cols_: list, target_col: str,
                                market_line_col: str, label: str):
    """
    Walk-forward regression on a MARKET RESIDUAL target (e.g. SPREAD_RESIDUAL).
    The new baseline is 'always predict residual = 0' -- i.e. trust the
    closing line exactly. This is a MUCH harder baseline than our old
    naive-average-margin one, because it's asking: does our model beat
    Vegas, not just beat a dumb constant.
    """
    results = []
    df_valid = df[df[target_col].notna()].copy()

    for train_idx, test_idx, test_season, train_seasons in walk_forward_season_splits(df_valid):
        X_train, y_train = df_valid.loc[train_idx, feature_cols_], df_valid.loc[train_idx, target_col]
        X_test, y_test = df_valid.loc[test_idx, feature_cols_], df_valid.loc[test_idx, target_col]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        train_medians = X_train.median()
        X_train, X_test = X_train.fillna(train_medians), X_test.fillna(train_medians)

        # "Trust the market" baseline: predicting residual=0 for every game
        market_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

        xgb_reg = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.03,
                                min_child_weight=1, subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgb_reg.fit(X_train, y_train)
        xgb_preds = xgb_reg.predict(X_test)

        fold_result = {
            'test_season': test_season, 'n_games': len(test_idx),
            'market_mae': market_mae,
            'model_mae': mean_absolute_error(y_test, xgb_preds),
        }
        fold_result['beats_market'] = fold_result['model_mae'] < fold_result['market_mae']
        results.append(fold_result)

        print(f"[{label}] Season {test_season}: market_mae={market_mae:.2f}  model_mae={fold_result['model_mae']:.2f}  "
              f"{'<- MODEL BEATS MARKET' if fold_result['beats_market'] else ''}")

    return pd.DataFrame(results)


# ==============================================================================
# PHASE 4c: KELLY CRITERION SIZING + BACKTEST SIMULATION
# ==============================================================================

def american_to_decimal(american_odds: float) -> float:
    """Converts American odds to decimal odds (total payout multiple, incl. stake)."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def kelly_fraction(win_prob: float, american_odds: float, kelly_multiplier: float = 0.25) -> float:
    """
    Fractional Kelly stake as a proportion of bankroll.
    f* = (b*p - q) / b   where b = decimal_odds - 1, q = 1 - p

    kelly_multiplier scales down full Kelly (e.g. 0.25 = quarter-Kelly) --
    standard practice given real-world edge estimates are noisy; full Kelly
    is extremely aggressive and assumes your probability estimate is exact,
    which it never is. Returns 0 (no bet) if the edge is negative.
    """
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    q = 1 - win_prob
    f_star = (b * win_prob - q) / b
    return max(f_star, 0.0) * kelly_multiplier


def simulate_kelly_backtest(predicted_probs: np.ndarray, actual_outcomes: np.ndarray,
                              american_odds: np.ndarray, starting_bankroll: float = 1000.0,
                              kelly_multiplier: float = 0.25, max_bet_pct: float = 0.05,
                              min_edge_threshold: float = 0.0) -> dict:
    """
    Simulates sequential flat-Kelly betting through a series of games.

    predicted_probs: our calibrated win probability for the bet side
    actual_outcomes: 1 if the bet won, 0 if it lost (already resolved to bet perspective)
    american_odds: the odds offered on the side we bet (e.g. -110)
    max_bet_pct: hard cap on any single bet regardless of Kelly output (risk control --
                 pure Kelly can occasionally suggest oversized bets on estimation noise)
    min_edge_threshold: only bet when our probability exceeds market breakeven by this much
                         (e.g. 0.02 = only bet when we have a 2+ point edge over breakeven)

    Returns bankroll trajectory + standard risk metrics (Sharpe, max drawdown).
    NOTE: Sharpe here is computed on PER-BET returns, not time-normalized --
    a simplification appropriate for comparing strategies within this backtest,
    not a substitute for an annualized, calendar-aware Sharpe ratio.
    """
    bankroll = starting_bankroll
    trajectory = [bankroll]
    bets_placed = 0
    bet_returns = []

    for p, outcome, odds in zip(predicted_probs, actual_outcomes, american_odds):
        if pd.isna(p) or pd.isna(odds):
            trajectory.append(bankroll)
            continue

        breakeven_prob = 1 / american_to_decimal(odds)
        edge = p - breakeven_prob

        if edge <= min_edge_threshold:
            trajectory.append(bankroll)
            continue

        f = kelly_fraction(p, odds, kelly_multiplier)
        f = min(f, max_bet_pct)
        stake = bankroll * f

        if stake <= 0:
            trajectory.append(bankroll)
            continue

        decimal_odds = american_to_decimal(odds)
        if outcome == 1:
            profit = stake * (decimal_odds - 1)
        else:
            profit = -stake

        bankroll += profit
        bet_returns.append(profit / (bankroll - profit))  # return relative to bankroll BEFORE this bet
        bets_placed += 1
        trajectory.append(bankroll)

    trajectory = np.array(trajectory)
    running_max = np.maximum.accumulate(trajectory)
    drawdowns = (trajectory - running_max) / running_max
    max_drawdown = drawdowns.min()

    bet_returns = np.array(bet_returns)
    sharpe = (bet_returns.mean() / bet_returns.std() * np.sqrt(len(bet_returns))
              if len(bet_returns) > 1 and bet_returns.std() > 0 else np.nan)

    return {
        'starting_bankroll': starting_bankroll,
        'ending_bankroll': bankroll,
        'total_return_pct': (bankroll / starting_bankroll - 1) * 100,
        'bets_placed': bets_placed,
        'total_games_seen': len(predicted_probs),
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_per_bet': sharpe,
        'trajectory': trajectory,
    }



# ==============================================================================
# PHASE 1c/2i: PLAYER AVAILABILITY (box-score-derived, fully historical & free)
# ==============================================================================

def fetch_season_player_log(season: str, season_type: str = SEASON_TYPE) -> pd.DataFrame:
    """
    Fetches ALL players' game logs for a season in ONE API call (same bulk
    pattern as fetch_season_game_log, just player-level instead of team-level).
    This is the foundation for deriving historical player availability --
    no separate injury feed needed.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=season_type,
                player_or_team_abbreviation='P', timeout=API_TIMEOUT
            )
            df = log.get_data_frames()[0]
            df['SEASON'] = season
            time.sleep(REQUEST_DELAY)
            return df
        except Exception as e:
            wait = REQUEST_DELAY * (2 ** (attempt - 1))
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] {season} player log failed: {type(e).__name__}: {e} -> retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch player log for {season} after {MAX_RETRIES} attempts.")


def fetch_multi_season_player_logs(seasons: list) -> pd.DataFrame:
    """Fetches player logs across multiple seasons, same pattern as team logs."""
    all_logs = []
    for season in seasons:
        print(f"Fetching player log {season}...")
        df = fetch_season_player_log(season)
        print(f"  -> {len(df)} player-game rows")
        all_logs.append(df)
    return pd.concat(all_logs, ignore_index=True)


def compute_expected_rotation(player_log_df: pd.DataFrame, top_n: int = 8, window: int = 10) -> pd.DataFrame:
    """
    For each player-team-season, computes a LEAKAGE-SAFE rolling average of
    minutes played over their trailing `window` games (shift(1) before
    rolling, same discipline as every other rolling feature in this
    pipeline). This defines each player's recent role -- used next to
    determine each team's "expected rotation" per game.
    """
    df = player_log_df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'SEASON', 'GAME_DATE']).reset_index(drop=True)

    grouped = df.groupby(['PLAYER_ID', 'SEASON'], group_keys=False)
    df['MIN_ROLL'] = grouped['MIN'].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    df['PTS_ROLL'] = grouped['PTS'].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    df['PLUS_MINUS_ROLL'] = grouped['PLUS_MINUS'].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    return df


def compute_missing_player_impact(player_log_with_roll: pd.DataFrame, team_games: pd.DataFrame,
                                    top_n: int = 8) -> pd.DataFrame:
    """
    For each team-game in team_games, determines which of that team's
    "expected rotation" (top_n players by rolling minutes, as of games
    STRICTLY BEFORE this one) did NOT appear in this game's actual box
    score, and quantifies the impact of their absence.

    Returns team_games with new columns:
      MISSING_PLAYERS_COUNT   -- how many expected-rotation players sat out
      MISSING_MINUTES_SHARE   -- their combined expected minutes / 240 (5 players x 48 min)
      MISSING_PTS_IMPACT      -- sum of their rolling PPG (a simple, honestly-labeled
                                  proxy for on-court impact -- NOT a real BPM/VORP,
                                  since that requires a paid/scraped advanced-stats
                                  source we don't have here)
    """
    plog = player_log_with_roll.copy()
    plog['TEAM_ABBREVIATION'] = plog['TEAM_ABBREVIATION'].astype(str)

    # Pre-build a single (TEAM, GAME_ID) -> {player_ids who played} lookup ONCE.
    # The original version re-filtered the full ~150k-row player log on EVERY
    # team-game iteration (14,000+ times) -- this dict makes that O(1) per game
    # instead of O(full player log size) per game, which was the actual
    # bottleneck (12x+ slower than benchmarked on a real 6-season dataset).
    roster_lookup = plog.groupby(['TEAM_ABBREVIATION', 'GAME_ID'])['PLAYER_ID'].apply(set).to_dict()

    results = []

    for (team, season), team_group in team_games.groupby(['TEAM_ABBREVIATION', 'SEASON']):
        team_group = team_group.sort_values('GAME_DATE')
        team_players = plog[(plog['TEAM_ABBREVIATION'] == team) & (plog['SEASON'] == season)]

        for _, game_row in team_group.iterrows():
            game_date = game_row['GAME_DATE']
            game_id = game_row['GAME_ID']

            # Expected rotation: top_n players by rolling minutes, using ONLY
            # their record as of the game immediately prior to this one
            prior_games = team_players[team_players['GAME_DATE'] < game_date]
            if len(prior_games) == 0:
                # No history yet (start of season) -- can't determine expected rotation
                results.append({'GAME_ID': game_id, 'TEAM_ABBREVIATION': team,
                                 'MISSING_PLAYERS_COUNT': np.nan, 'MISSING_MINUTES_SHARE': np.nan,
                                 'MISSING_PTS_IMPACT': np.nan})
                continue

            # Most recent known rolling values per player, right before this game
            latest_per_player = prior_games.sort_values('GAME_DATE').groupby('PLAYER_ID').last()
            expected_rotation = latest_per_player.nlargest(top_n, 'MIN_ROLL')

            # O(1) dict lookup instead of re-scanning the full player log
            actual_this_game = roster_lookup.get((team, game_id), set())

            missing = expected_rotation[~expected_rotation.index.isin(actual_this_game)]

            results.append({
                'GAME_ID': game_id, 'TEAM_ABBREVIATION': team,
                'MISSING_PLAYERS_COUNT': len(missing),
                'MISSING_MINUTES_SHARE': missing['MIN_ROLL'].sum() / 240.0,
                'MISSING_PTS_IMPACT': missing['PTS_ROLL'].sum(),
            })

    return pd.DataFrame(results)

