data_path: "data/mega_sena.csv"

db:
  path: "mega_sena.db"

ml:
  n_estimators: 300
  max_depth: null
  random_state: 42

hybrid:
  alpha: 0.5
  pool_size: 1000
  seed: 123
  top_k: 20

api:
  host: "0.0.0.0"
  port: 8000
