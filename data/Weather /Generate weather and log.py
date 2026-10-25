"""
Precision Driven Farming AI - weather download + SYNTHETIC detection log generator.

*** IMPORTANT: the detection log produced by this script is SIMULATED (SYNTHETIC). ***
It is NOT real farm data. It exists because no public, time-stamped crop disease
dataset links to dates and field locations. It demonstrates the time series method.

What this script does
  1. Downloads daily weather from the NASA POWER agroclimatology API (REAL data).
  2. Cleans it (NASA codes missing values as -999).
  3. Generates a synthetic weekly disease/pest detection log whose outbreak risk
     depends on the real weather (humidity, rainfall, temperature), with seasonal
     patterns, persistence and random noise. Fixed seed => identical output every run.

Usage (from the repo root):
    pip install pandas numpy requests
    python generate_weather_and_log.py                  # download + generate
    python generate_weather_and_log.py --skip-download  # reuse the saved raw weather CSV

Outputs
    data/weather/nasa_power_daily_raw.csv         untouched NASA file
    data/weather/nasa_power_daily_clean.csv       cleaned daily weather
    data/simulated/simulated_detection_log.csv    SYNTHETIC weekly detection log

NASA Acknowledgement :
    "The data was obtained from the National Aeronautics and Space Administration
    (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resource
    (POWER) Project funded through the NASA Earth Science/Applied Science Program."
"""


import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
LOCATION_NAME = "Mthatha, Eastern Cape, South Africa"
LATITUDE = -31.5919
LONGITUDE = 28.777
START_DATE = "20230101"  # YYYYMMDD 
END_DATE = "20260630"
PARAMETERS = "T2M,RH2M,PRECTOTCORR"  # temperature (C), relative humidity (%), rainfall (mm/day)

SEED = 42  # fixed so the synthetic log is reproducible
N_ZONES = 4  # field zones A-D, each treated as a 10 ha block
IMAGES_PER_ZONE_PER_WEEK = 700  # baseline: 100 images per 10 ha per day x 7 days
CROP = "Maize"

BASE = Path(__file__).resolve().parent
RAW_PATH = BASE / "data" / "weather" / "nasa_power_daily_raw.csv"
CLEAN_PATH = BASE / "data" / "weather" / "nasa_power_daily_clean.csv"
LOG_PATH = BASE / "data" / "simulated" / "simulated_detection_log.csv"

# Disease/pest classes follow the CCMT maize class names.
# Coefficients are ASSUMPTIONS of the simulation (documented in the README):
#   weights act on standardised weekly weather features (z-scores)
#   t_opt/t_width: preferred temperature band (Gaussian bump); 0 width = ignore
CLASSES = {
    #  class            base   w_humid w_rain  t_opt t_width  category
    "Leaf blight":     (-3.2,   0.9,    0.6,   22.0,  5.0,   "Disease"),
    "Leaf spot":       (-3.0,   0.8,    0.7,   24.0,  6.0,   "Disease"),
    "Streak virus":    (-3.6,   0.1,   -0.2,   28.0,  5.0,   "Disease"),
    "Fall armyworm":   (-3.4,  -0.3,   -0.4,   27.0,  5.0,   "Pest"),
    "Grasshopper":     (-3.8,  -0.4,   -0.5,   29.0,  6.0,   "Pest"),
    "Leaf beetle":     (-3.6,   0.2,    0.1,   25.0,  6.0,   "Pest"),
}

# Crop calendar weight by month: maize planted ~Nov, harvested ~May (Southern Hemisphere).
# Off-season keeps a small value (volunteer plants / stubble) so the series is never all zero.
SEASON = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.7, 5: 0.35, 6: 0.05, 7: 0.05,
          8: 0.05, 9: 0.08, 10: 0.3, 11: 0.7, 12: 0.9}


def download_weather() -> None:
    import requests  # imported here so --skip-download works without it

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={PARAMETERS}&community=AG"
        f"&longitude={LONGITUDE}&latitude={LATITUDE}"
        f"&start={START_DATE}&end={END_DATE}&format=CSV"
    )
    print("Downloading NASA POWER weather ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(resp.text, encoding="utf-8")  # untouched raw file
    print(f"  saved raw file: {RAW_PATH}")


def load_and_clean_weather() -> pd.DataFrame:
    text = RAW_PATH.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # NASA CSVs start with a metadata block ending in -END HEADER-
    skip = next((i + 1 for i, ln in enumerate(lines) if "-END HEADER-" in ln), 0)
    # Handles the normal comma file AND files re-saved by Excel with semicolons
    # (where each row is duplicated after the 5th semicolon): only the first
    # 5 fields of each row are used.
    def split(ln: str):
        parts = ln.split(";") if ";" in ln else ln.split(",")
        return [x.strip() for x in parts[:5]]
    header = split(lines[skip])
    body = [split(ln) for ln in lines[skip + 1:]]
    df = pd.DataFrame(body, columns=header).apply(pd.to_numeric, errors="coerce")
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j"
    )
    df = df.rename(columns={"T2M": "temperature_c", "RH2M": "humidity_pct",
                            "PRECTOTCORR": "rainfall_mm"})
    df = df[["date", "temperature_c", "humidity_pct", "rainfall_mm"]]

    n_missing = int((df[["temperature_c", "humidity_pct", "rainfall_mm"]] == -999).sum().sum())
    df = df.replace(-999, np.nan)
    df = df.sort_values("date").set_index("date")
    df[["temperature_c", "humidity_pct", "rainfall_mm"]] = (
        df[["temperature_c", "humidity_pct", "rainfall_mm"]].interpolate(limit_direction="both")
    )
    df = df.reset_index()
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"  rows: {len(df)} | missing values (-999) found and interpolated: {n_missing}")
    print(f"  saved cleaned file: {CLEAN_PATH}")
    return df


def weekly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    w = (daily.set_index("date")
         .resample("W-MON", label="left", closed="left")
         .agg(temperature_c=("temperature_c", "mean"),
              humidity_pct=("humidity_pct", "mean"),
              rainfall_mm=("rainfall_mm", "sum"),
              n_days=("rainfall_mm", "size")))
    w = w[w["n_days"] == 7].drop(columns="n_days")  # keep complete 7-day weeks only
    w = w.dropna().reset_index().rename(columns={"date": "week_start"})
    for col in ["humidity_pct", "rainfall_mm"]:
        w["z_" + col] = (w[col] - w[col].mean()) / w[col].std()
        # disease/pest pressure responds to the PREVIOUS week's weather (1-week lag)
        w["z_" + col] = w["z_" + col].shift(1).fillna(0.0)
    w["temp_lag"] = w["temperature_c"].shift(1).fillna(w["temperature_c"].iloc[0])
    return w


def simulate_log(weekly: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    zones = [chr(ord("A") + i) for i in range(N_ZONES)]
    zone_mult = dict(zip(zones, np.exp(rng.normal(0, 0.25, N_ZONES))))  # some zones are worse
    season = weekly["week_start"].dt.month.map(SEASON).to_numpy()

    rows = []
    for zone in zones:
        for cname, (base, w_h, w_r, t_opt, t_w, category) in CLASSES.items():
            x_prev = 0.0
            for i, wk in weekly.iterrows():
                temp_bump = np.exp(-0.5 * ((wk["temp_lag"] - t_opt) / t_w) ** 2)
                risk = (base + w_h * wk["z_humidity_pct"] + w_r * wk["z_rainfall_mm"]
                        + 1.2 * temp_bump)
                # AR(1) persistence: outbreaks carry over from week to week
                x = 0.55 * x_prev + 0.45 * risk + rng.normal(0, 0.25)
                x_prev = x
                rate = IMAGES_PER_ZONE_PER_WEEK * np.exp(x) * zone_mult[zone] * season[i] * 0.05
                count = int(min(rng.poisson(rate), IMAGES_PER_ZONE_PER_WEEK))
                # severity = % of leaf area affected; rises with outbreak intensity
                sev = 0.0 if count == 0 else float(
                    np.clip(1.0 + 3.0 * np.log1p(count) + rng.normal(0, 1.5), 0.5, 60.0))
                rows.append({
                    "week_start": wk["week_start"].date(),
                    "field_zone": zone,
                    "crop": CROP,
                    "class": cname,
                    "category": category,
                    "images_scanned": IMAGES_PER_ZONE_PER_WEEK,
                    "detections": count,
                    "mean_severity_pct": round(sev, 1),
                    "data_type": "SYNTHETIC",
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse the existing raw NASA file instead of downloading")
    args = ap.parse_args()

    if not args.skip_download:
        download_weather()
    elif not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} not found - run without --skip-download first.")

    daily = load_and_clean_weather()
    log = simulate_log(weekly_weather(daily))
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)

    print(f"\nSYNTHETIC log saved: {LOG_PATH}")
    print(f"  rows: {len(log)} | weeks: {log['week_start'].nunique()} | zones: {N_ZONES}")
    print(f"  date range: {log['week_start'].min()} to {log['week_start'].max()}")
    print("  total detections by class:")
    print(log.groupby("class")["detections"].sum().sort_values(ascending=False).to_string())
    print("\nREMEMBER: label this data as simulated in every document and on the poster.")


if __name__ == "__main__":
    main()
