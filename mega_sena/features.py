import pandas as pd
import numpy as np
from .utils import digit_sum

def is_prime(n: int) -> int:
    if n < 2: return 0
    i = 2
    while i * i <= n:
        if n % i == 0: return 0
        i += 1
    return 1

BASE_BALLS = ["b1","b2","b3","b4","b5","b6"]

def _feature_frame_from_parts(date, contest):
    d = {
        "contest": contest,
        "year": date.year, "month": date.month, "day": date.day,
        "weekday": date.weekday(), "day_of_year": date.timetuple().tm_yday,
        "iso_week": date.isocalendar().week,
        "quarter": (date.month - 1)//3 + 1,
        "day_digit_sum": digit_sum(date.day),
        "month_digit_sum": digit_sum(date.month),
        "year_mod_100": date.year % 100,
        "ymd_digit_sum": digit_sum(int(f"{date.year:04d}{date.month:02d}{date.day:02d}")),
        "contest_digit_sum": digit_sum(contest),
        "contest_is_prime": is_prime(contest),
        "contest_mod_10": contest % 10,
        "contest_mod_12": contest % 12,
        "contest_mod_60": contest % 60,
        "dm_mod_60": (date.day * date.month) % 60,
        "y_mod_60": date.year % 60,
        "mix1_mod_60": (contest + date.day) % 60,
        "mix2_mod_60": (contest + date.month) % 60,
        "mix3_mod_60": (contest + date.year) % 60,
    }
    return pd.DataFrame([d])

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for _, row in df.iterrows():
        feats.append(_feature_frame_from_parts(row["date"], int(row["contest"])).iloc[0])
    feat_df = pd.DataFrame(feats)
    # attach targets
    out = pd.concat([df[["date","contest"] + BASE_BALLS].reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return out

def build_single_features(date, contest) -> pd.DataFrame:
    return _feature_frame_from_parts(date, int(contest))
