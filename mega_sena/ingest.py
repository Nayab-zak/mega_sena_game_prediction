import pandas as pd

KEEP_COLS = ["Concurso","Data do Sorteio","Bola1","Bola2","Bola3","Bola4","Bola5","Bola6"]
CANON = ["contest","date","b1","b2","b3","b4","b5","b6"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # keep only A–H
    df = df[KEEP_COLS].copy()
    df.columns = CANON

    # coerce date (Brazil format present historically -> dayfirst)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # numeric balls & contest
    for c in ["contest","b1","b2","b3","b4","b5","b6"]:
        df[c] = pd.to_numeric(df[c].str.extract(r"(\d+)")[0], errors="coerce")

    # drop invalid rows
    df = df.dropna(subset=["date","contest","b1","b2","b3","b4","b5","b6"]).copy()
    df = df.astype({"contest": int, "b1": int, "b2": int, "b3": int, "b4": int, "b5": int, "b6": int})

    # keep only valid ball range
    mask = df[["b1","b2","b3","b4","b5","b6"]].applymap(lambda x: 1 <= x <= 60).all(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    return df
