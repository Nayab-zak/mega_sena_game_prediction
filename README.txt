# from repo root
python -m pip install -r requirements.txt

# put your CSV (A–H) at data/mega_sena.csv

# create/refresh SQLite DB
python main.py seed

#local quick check
# CLI – try each mode
python main.py suggest --date 2024-10-02 --contest 2354 --k 20 --mode hybrid
python main.py suggest --date 2024-10-02 --contest 2354 --k 20 --mode rule
python main.py suggest --date 2024-10-02 --contest 2354 --k 20 --mode ml

# API server 
uvicorn mega_sena.api:app --host 0.0.0.0 --port 8000 --reload


# example request 
curl -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"date":"2024-10-02","contest":2354,"k":20}'


Rules I applied 

The rule-based part of the system generates number probabilities from:

Date features — day, month, year, weekday, day-of-year, ISO week, digit sums, and modulo values (e.g., (day * month) % 60).

Contest number features — modulo values (% 10, % 12, % 60), digit sum, prime/composite flag, and combinations with date values.

Similarity search — we find historical draws with matching or close feature values, and count how often each number appeared in those draws.

Frequency scoring — the more often a number appears in similar draws, the higher its probability score.

Diversity constraints — encourage spread across decades (1–10, 11–20, …) and balance between even/odd numbers

