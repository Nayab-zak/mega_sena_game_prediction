from sqlalchemy import create_engine, Column, Integer, Date, MetaData, Table
from sqlalchemy.orm import sessionmaker
import pandas as pd
from .ingest import load_data

def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", future=True)

def ensure_schema(engine):
    meta = MetaData()
    Table(
        "draws", meta,
        Column("contest", Integer, primary_key=True, nullable=False),
        Column("date",    Date,    nullable=False),
        Column("b1", Integer, nullable=False),
        Column("b2", Integer, nullable=False),
        Column("b3", Integer, nullable=False),
        Column("b4", Integer, nullable=False),
        Column("b5", Integer, nullable=False),
        Column("b6", Integer, nullable=False),
    )
    meta.create_all(engine)

def seed_from_csv(db_path: str, csv_path: str):
    engine = get_engine(db_path)
    ensure_schema(engine)
    df = load_data(csv_path)  # uses columns A–H only, already cleansed
    # Upsert by contest: simple approach—replace table (fast, safe for this use)
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM draws")
        df[["contest","date","b1","b2","b3","b4","b5","b6"]].to_sql(
            "draws", conn, if_exists="append", index=False
        )

def load_history_df(db_path: str) -> pd.DataFrame:
    engine = get_engine(db_path)
    ensure_schema(engine)
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM draws ORDER BY contest", conn, parse_dates=["date"])
    return df
