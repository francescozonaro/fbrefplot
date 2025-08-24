import pandas as pd


def normalize_fbref_schedule(df: pd.DataFrame, isFuture: bool) -> pd.DataFrame:
    """
    Normalize match-level data into team-level format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
        - home_team, away_team
        - home_xg, away_xg
        - home_goals, away_goals

    Returns
    -------
    pd.DataFrame
        A "long" DataFrame where each row corresponds to one team's perspective.
        Columns: team, opponent, xg, opponent_xg, goals, opponent_goals, at_home
    """

    home_cols = {
        "home_team": "team",
        "away_team": "opponent",
    }
    away_cols = {
        "home_team": "opponent",
        "away_team": "team",
    }

    if not isFuture:
        df["home_goals"], df["away_goals"] = separate_score(df["score"])

        home_cols.update(
            {
                "home_xg": "xg",
                "away_xg": "opponent_xg",
                "home_goals": "goals",
                "away_goals": "opponent_goals",
            }
        )
        away_cols.update(
            {
                "home_xg": "opponent_xg",
                "away_xg": "xg",
                "home_goals": "opponent_goals",
                "away_goals": "goals",
            }
        )

    home_df = df.rename(columns=home_cols).copy()
    home_df["at_home"] = True

    away_df = df.rename(columns=away_cols).copy()
    away_df["at_home"] = False

    return pd.concat([home_df, away_df], ignore_index=True)


def separate_score(score_series: pd.Series) -> pd.DataFrame:
    """
    Splits a score Series (e.g., '2–1') into two integer Series.
    """
    scores = score_series.str.split("–", expand=True)
    home_goals = scores[0].astype(int)
    away_goals = scores[1].astype(int)

    return home_goals, away_goals


def filter_regular_season(
    df: pd.DataFrame,
    league: str,
) -> pd.DataFrame:
    """
    Filter match data to include only regular season matches for supported leagues.
    """
    match league:
        case "BEL-Belgian Pro League":
            df = df[df["round"] == "Regular season"]
            return df
        case _:
            return df
