import os
import json
import time
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

import boto3
import numpy as np
import pandas as pd
import requests
import joblib
import tensorflow as tf

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ============================================================
# ENV / SETTINGS
# ============================================================
OANDA_ENV = os.getenv("OANDA_ENV", "practice").strip().lower()
OANDA_TOKEN = os.getenv("OANDA_TOKEN", "").strip()
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "").strip()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1").strip()
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()  # bucket name only (NO s3://)
S3_PREFIX = os.getenv("S3_PREFIX", "models/").strip().lstrip("/")  # e.g. models/

REFRESH_SYMBOLS_SECONDS = int(os.getenv("REFRESH_SYMBOLS_SECONDS", "60").strip())
SIGNAL_THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD", "1.0").strip())

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data")).strip()
MODELS_DIR = os.path.join(DATA_DIR, "models")
DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "primemarket.db")).strip()

# Safety checks
if not S3_BUCKET:
    raise RuntimeError("Missing S3_BUCKET env var (bucket name only, e.g. amzn-s3-primemarketai).")
if not OANDA_TOKEN:
    raise RuntimeError("Missing OANDA_TOKEN env var. Put it in .env (see .env.example).")

BASE_URL = "https://api-fxpractice.oanda.com" if OANDA_ENV == "practice" else "https://api-fxtrade.oanda.com"
OANDA_HEAD = {"Authorization": f"Bearer {OANDA_TOKEN}", "Accept": "application/json"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# DB
# ============================================================
def db_connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

DB = db_connect()

DB.execute("""
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  boundary_time_utc TEXT NOT NULL,
  pred_close REAL NOT NULL,
  last_close REAL NOT NULL,
  delta REAL NOT NULL,
  signal TEXT NOT NULL,
  created_utc TEXT NOT NULL
);
""")

DB.execute("""
CREATE TABLE IF NOT EXISTS app_state (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);
""")

DB.commit()


def db_set(k: str, v: str):
    DB.execute("INSERT INTO app_state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
    DB.commit()

def db_get(k: str) -> Optional[str]:
    cur = DB.execute("SELECT v FROM app_state WHERE k=?", (k,))
    row = cur.fetchone()
    return row[0] if row else None


# ============================================================
# OANDA HELPERS
# ============================================================
def ensure_utc_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")

def to_iso_utc(ts: pd.Timestamp) -> str:
    ts = ensure_utc_ts(ts)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

def oanda_auth_check():
    url = f"{BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary" if OANDA_ACCOUNT_ID else f"{BASE_URL}/v3/accounts"
    r = requests.get(url, headers=OANDA_HEAD, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OANDA AUTH FAILED {r.status_code}: {r.text[:300]}")
    return True

def fetch_oanda_candles(instrument: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, granularity: str, count: int = 5000) -> pd.DataFrame:
    start_ts = ensure_utc_ts(start_ts)
    end_ts = ensure_utc_ts(end_ts)

    rows = []
    cur = start_ts
    step_minutes = 1 if granularity == "M1" else 5 if granularity == "M5" else 1

    while cur < end_ts:
        url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
        params = {"granularity": granularity, "price": "M", "from": to_iso_utc(cur), "count": count}
        r = requests.get(url, headers=OANDA_HEAD, params=params, timeout=30)
        ct = r.headers.get("Content-Type", "")
        if r.status_code != 200:
            raise RuntimeError(f"OANDA error {r.status_code} CT={ct}: {r.text[:300]}")
        if "application/json" not in ct:
            raise RuntimeError(f"OANDA NON-JSON CT={ct}: {r.text[:300]}")

        js = r.json()
        candles = js.get("candles", [])
        if not candles:
            break

        added = 0
        for c in candles:
            if not c.get("complete", True):
                continue
            m = c.get("mid")
            if not m:
                continue
            rows.append([c["time"], float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"]), int(c.get("volume", 0))])
            added += 1

        if added == 0:
            break

        last_time = pd.to_datetime(rows[-1][0], utc=True)
        cur = last_time + pd.Timedelta(minutes=step_minutes)

    df = pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    return df


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)

def floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)

def is_5m_boundary(dt: datetime) -> bool:
    return dt.second == 0 and (dt.minute % 5 == 0)

def sleep_until_next_minute():
    now = utc_now_dt()
    s = 60 - now.second
    if s <= 0:
        s = 1
    time.sleep(s)

def sleep_until_next_5m_boundary():
    now = utc_now_dt().replace(microsecond=0)
    # next minute tick first
    sleep_until_next_minute()
    while True:
        now = utc_now_dt().replace(microsecond=0)
        if is_5m_boundary(now):
            return
        sleep_until_next_minute()


# ============================================================
# MODEL REGISTRY (S3 -> local cache -> load)
# ============================================================
@dataclass
class LoadedModel:
    symbol: str
    model: tf.keras.Model
    x_scaler: object
    y_scaler: object
    cfg: dict

class ModelRegistry:
    def __init__(self, bucket: str, prefix: str, local_models_dir: str, region: str):
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.local_models_dir = local_models_dir
        self.s3 = boto3.client("s3", region_name=region)
        self.models: Dict[str, LoadedModel] = {}
        self.lock = threading.Lock()

    def _s3_list_symbols(self) -> List[str]:
        # list "folders" under prefix
        paginator = self.s3.get_paginator("list_objects_v2")
        symbols = set()

        # We infer symbols by searching for config.json under models/<SYMBOL>/config.json
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                # expect models/SYMBOL/config.json
                if key.endswith("/config.json") and key.startswith(self.prefix):
                    rest = key[len(self.prefix):]
                    parts = rest.split("/")
                    if len(parts) >= 2:
                        symbols.add(parts[0])
        return sorted(symbols)

    def _download_symbol(self, symbol: str) -> str:
        # local folder
        sym_dir = os.path.join(self.local_models_dir, symbol)
        os.makedirs(sym_dir, exist_ok=True)

        needed = ["model.keras", "scalers.pkl", "config.json"]
        for fn in needed:
            key = f"{self.prefix}{symbol}/{fn}"
            dest = os.path.join(sym_dir, fn)
            self.s3.download_file(self.bucket, key, dest)
        return sym_dir

    def _load_symbol(self, symbol: str) -> LoadedModel:
        sym_dir = os.path.join(self.local_models_dir, symbol)
        model_path = os.path.join(sym_dir, "model.keras")
        scalers_path = os.path.join(sym_dir, "scalers.pkl")
        config_path = os.path.join(sym_dir, "config.json")

        with open(config_path, "r") as f:
            cfg = json.load(f)

        model = tf.keras.models.load_model(model_path)
        scalers = joblib.load(scalers_path)

        return LoadedModel(
            symbol=symbol,
            model=model,
            x_scaler=scalers["x_scaler"],
            y_scaler=scalers["y_scaler"],
            cfg=cfg
        )

    def refresh(self) -> Dict[str, str]:
        """
        Ensures all symbols in S3 are available locally and loaded in memory.
        Returns status dict.
        """
        status = {}
        symbols = self._s3_list_symbols()
        with self.lock:
            for sym in symbols:
                if sym in self.models:
                    status[sym] = "loaded"
                    continue
                try:
                    self._download_symbol(sym)
                    self.models[sym] = self._load_symbol(sym)
                    status[sym] = "downloaded+loaded"
                except Exception as e:
                    status[sym] = f"error: {e}"
        return status

    def get(self, symbol: str) -> Optional[LoadedModel]:
        with self.lock:
            return self.models.get(symbol)

    def list_loaded(self) -> List[str]:
        with self.lock:
            return sorted(self.models.keys())


REG = ModelRegistry(
    bucket=S3_BUCKET,
    prefix=S3_PREFIX,
    local_models_dir=MODELS_DIR,
    region=AWS_REGION
)


# ============================================================
# FEATURE ENGINEERING (must match training)
# ============================================================
def make_features(df_m1: pd.DataFrame, ma_periods: List[int]) -> pd.DataFrame:
    feat = df_m1.copy()
    feat["TP"] = (feat["Open"] + feat["High"] + feat["Low"] + feat["Close"]) / 4.0

    for p in ma_periods:
        feat[f"MA_TP_{p}"] = feat["TP"].rolling(p).mean()

    for p in ma_periods:
        ma_col = f"MA_TP_{p}"
        for col in ["Open", "High", "Low", "Close"]:
            feat[f"delta_{col}_{p}"] = feat[col] - feat[ma_col]

    delta_cols = [c for c in feat.columns if c.startswith("delta_")]
    feat["avg_delta_ohlc"] = feat[delta_cols].mean(axis=1)

    return feat.dropna()


def build_latest_X_for_boundary(lm: LoadedModel) -> Tuple[np.ndarray, pd.Timestamp, float, bool]:
    """
    Returns:
      X (1, seq_len, n_features),
      boundary timestamp T (UTC),
      last_close (from same boundary row),
      market_open_guess (True if data is fresh)
    """
    cfg = lm.cfg
    seq_len = int(cfg["seq_len"])
    feature_cols = cfg["feature_cols"]
    ma_periods = cfg["ma_periods"]
    warmup = int(cfg.get("feature_warmup_minutes", 300))

    now = pd.Timestamp.utcnow().floor("min")  # already tz-aware (UTC)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    # pull enough minutes: warmup + seq + buffer
    minutes_needed = warmup + seq_len + 10
    start = now - pd.Timedelta(minutes=minutes_needed)
    end = now + pd.Timedelta(minutes=1)

    df = fetch_oanda_candles(lm.symbol, start, end, "M1")
    if df.empty:
        raise RuntimeError("No candles returned.")

    # market open guess = last candle is recent
    last_ts = df.index.max()
    market_open_guess = (now - last_ts) <= pd.Timedelta(minutes=3)

    feat = make_features(df, ma_periods)

    idx = feat.index
    boundary_mask = (idx.minute % 5 == 0)
    if boundary_mask.sum() == 0:
        raise RuntimeError("No 5-min boundary points in current window.")

    T = idx[boundary_mask][-1]
    feat_upto_T = feat.loc[:T]
    if len(feat_upto_T) < seq_len:
        raise RuntimeError(f"Not enough feature rows up to {T}. Have={len(feat_upto_T)}, need={seq_len}")

    window = feat_upto_T.iloc[-seq_len:][feature_cols].values
    last_close = float(feat_upto_T.iloc[-1]["Close"])

    window_scaled = lm.x_scaler.transform(window)
    X = window_scaled.astype(np.float32)[None, :, :]
    return X, T, last_close, market_open_guess


def predict_once(symbol: str) -> dict:
    lm = REG.get(symbol)
    if lm is None:
        raise RuntimeError(f"Symbol not loaded: {symbol}")

    X, T, last_close, market_open_guess = build_latest_X_for_boundary(lm)

    y_scaled = lm.model.predict(X, verbose=0)
    pred_close = float(lm.y_scaler.inverse_transform(y_scaled)[0, 0])

    delta = pred_close - last_close
    if delta >= SIGNAL_THRESHOLD:
        signal = "BUY"
    elif delta <= -SIGNAL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": symbol,
        "boundary_time_utc": str(T),
        "pred_close": pred_close,
        "last_close": last_close,
        "delta": delta,
        "signal": signal,
        "market_open_guess": market_open_guess,
        "created_utc": utc_now_dt().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def insert_prediction(row: dict):
    DB.execute(
        """INSERT INTO predictions(symbol, boundary_time_utc, pred_close, last_close, delta, signal, created_utc)
           VALUES(?,?,?,?,?,?,?)""",
        (
            row["symbol"],
            row["boundary_time_utc"],
            row["pred_close"],
            row["last_close"],
            row["delta"],
            row["signal"],
            row["created_utc"],
        ),
    )
    DB.commit()


def get_recent_predictions(symbol: str, limit: int = 200) -> List[dict]:
    cur = DB.execute(
        """SELECT symbol, boundary_time_utc, pred_close, last_close, delta, signal, created_utc
           FROM predictions
           WHERE symbol=?
           ORDER BY id DESC
           LIMIT ?""",
        (symbol, limit),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "symbol": r[0],
            "boundary_time_utc": r[1],
            "pred_close": r[2],
            "last_close": r[3],
            "delta": r[4],
            "signal": r[5],
            "created_utc": r[6],
        })
    return out


# ============================================================
# BACKGROUND LOOPS
# ============================================================
STOP = False

def model_refresh_loop():
    while not STOP:
        try:
            status = REG.refresh()
            db_set("last_model_refresh_utc", utc_now_dt().strftime("%Y-%m-%d %H:%M:%S UTC"))
            db_set("last_model_refresh_status", json.dumps(status))
        except Exception as e:
            db_set("last_model_refresh_status", json.dumps({"error": str(e)}))
        time.sleep(REFRESH_SYMBOLS_SECONDS)

def prediction_loop():
    # wait until we are aligned to boundary; then run forever
    last_fired_key = None
    while not STOP:
        sleep_until_next_5m_boundary()
        if STOP:
            break

        now_dt = utc_now_dt().replace(microsecond=0)
        key = now_dt.strftime("%Y-%m-%d %H:%M")
        if key == last_fired_key:
            continue
        last_fired_key = key

        # predict for every loaded symbol
        symbols = REG.list_loaded()
        if not symbols:
            db_set("last_pred_status", "No symbols loaded yet.")
            continue

        for sym in symbols:
            try:
                row = predict_once(sym)
                insert_prediction(row)
                db_set("last_pred_status", f"OK {sym} @ {row['boundary_time_utc']} pred={row['pred_close']:.5f}")
            except Exception as e:
                db_set("last_pred_status", f"ERR {sym}: {e}")


# ============================================================
# FASTAPI UI
# ============================================================
app = FastAPI(title="PrimeMarketAI - Simple Inference UI")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def on_start():
    # quick check
    oanda_auth_check()
    # initial refresh now
    REG.refresh()

    # start background threads
    t1 = threading.Thread(target=model_refresh_loop, daemon=True)
    t2 = threading.Thread(target=prediction_loop, daemon=True)
    t1.start()
    t2.start()

    db_set("started_utc", utc_now_dt().strftime("%Y-%m-%d %H:%M:%S UTC"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    symbols = REG.list_loaded()
    selected = request.query_params.get("symbol") or (symbols[0] if symbols else "")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "symbols": symbols,
            "selected": selected,
            "threshold": SIGNAL_THRESHOLD,
        },
    )


@app.get("/api/status")
def api_status():
    symbols = REG.list_loaded()
    last_refresh = db_get("last_model_refresh_utc") or ""
    last_refresh_status = db_get("last_model_refresh_status") or "{}"
    last_pred_status = db_get("last_pred_status") or ""
    started = db_get("started_utc") or ""

    return JSONResponse({
        "started_utc": started,
        "symbols_loaded": symbols,
        "last_model_refresh_utc": last_refresh,
        "last_model_refresh_status": json.loads(last_refresh_status) if last_refresh_status else {},
        "last_pred_status": last_pred_status,
        "server_utc_now": utc_now_dt().strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


@app.get("/api/predictions")
def api_predictions(symbol: str, limit: int = 200):
    data = get_recent_predictions(symbol, limit=limit)
    data = list(reversed(data))  # chronological for charts
    return JSONResponse({"symbol": symbol, "rows": data})


@app.get("/api/predict_now")
def api_predict_now(symbol: str):
    row = predict_once(symbol)
    insert_prediction(row)
    return JSONResponse(row)

