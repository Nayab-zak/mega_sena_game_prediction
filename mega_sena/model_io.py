import json, os, hashlib, joblib, sys
from datetime import datetime
import sklearn

def simple_data_fingerprint(df):
    # cheap & stable: row count + max contest + last date
    n = len(df)
    max_c = int(df["contest"].max()) if n else 0
    last = str(df["date"].max().date()) if n else "NA"
    s = f"{n}|{max_c}|{last}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def save_model(path_dir, model, feature_columns, ml_config, data_hash):
    os.makedirs(path_dir, exist_ok=True)
    joblib.dump(model, os.path.join(path_dir, "model.joblib"))
    manifest = {
        "feature_columns": feature_columns,
        "ml_config": ml_config,
        "data_hash": data_hash,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(path_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def load_model(path_dir):
    model_path = os.path.join(path_dir, "model.joblib")
    manifest_path = os.path.join(path_dir, "manifest.json")
    if not (os.path.exists(model_path) and os.path.exists(manifest_path)):
        return None, None
    model = joblib.load(model_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return model, manifest
