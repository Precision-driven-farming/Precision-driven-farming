# Weather Data, SYNTHETIC Detection Log and Time-Series Trend Analysis

> **The detection log in this project is SIMULATED (SYNTHETIC). It is not real farm data.**
> Label it as simulated in every document, figure and on the poster.

## Why it exists
The CCMT image dataset has no timestamps or field locations, and no public time-stamped crop disease dataset exists. A synthetic log, driven by real weather, is used to demonstrate the time-series trend analysis in Section 3.3 (zone health index, spray before/after comparison, untreated control). Results show the method working, not real-farm performance.

## Files
| Path | Type | Description |
|---|---|---|
| `generate_weather_and_log.py` | Script | Downloads weather and generates the simulated data (seed 42) |
| `time_series_trend_analysis.py` | Script | Health index, 3-week average, pre/post-spray slopes, level comparison, figure |
| `data/weather/nasa_power_daily_raw.csv` | REAL | NASA POWER download for Mthatha (the team's copy was re-saved with semicolons; the script reads this and the standard comma format) |
| `data/weather/nasa_power_daily_clean.csv` | REAL (cleaned) | Missing values (-999) interpolated (none were found) |
| `data/simulated/simulated_detection_log.csv` | SYNTHETIC | Week x zone x class: images sampled, detections, severity, spray flag |
| `data/simulated/simulated_zone_health.csv` | SYNTHETIC | Week x zone: health index, spray events, weather |
| `data/simulated/*_PLACEBO.csv` | SYNTHETIC | Same simulation with NO spray effect (method check) |
| `outputs/time_series/` | Results | Figure, weekly index, slope and level tables, `trend_analysis.md` |
| `outputs/time_series_placebo/` | Results | The same analysis on the placebo data |

## How to run
```
pip install pandas numpy requests scipy matplotlib
python generate_weather_and_log.py                  # or add --skip-download to reuse the saved raw file
python time_series_trend_analysis.py
python generate_weather_and_log.py --skip-download --placebo
python time_series_trend_analysis.py --placebo
```
The same weather file and seed always give identical output.

## Weather data (real)
- Source: NASA POWER agroclimatology (AG community), daily, point data
- Location: Mthatha, Eastern Cape, South Africa (lat -31.5919, lon 28.777; MERRA-2 grid cell)
- Period: 1 January 2023 to 30 June 2026 (1,277 daily rows, no gaps, no missing values)
- Variables: T2M (temperature, °C), RH2M (relative humidity, %), PRECTOTCORR (rainfall, mm/day)
- Acknowledgement: "The data was obtained from the National Aeronautics and Space Administration (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resource (POWER) Project funded through the NASA Earth Science/Applied Science Program."

## Simulation design (version 2, matches Section 3.3)
- **Field layout:** 3 zones, each made of 20 grid cells of 10 m x 10 m (0.2 ha). 10 leaf images are sampled per cell per week, so every zone-week has 200 images. Rows are stored per zone (cell-level rows are not stored).
  - **Zone A:** treated, early spray
  - **Zone B:** treated, late spray (3 weeks after Zone A)
  - **Zone C:** untreated control, never sprayed
- **Health index:** H(t) = N_healthy / N_total x 100, where N_total is the images sampled and N_healthy is the images not classified as a disease or pest.
- **Period and size:** 182 full weeks (2023-01-02 to 2026-06-22), 3,276 log rows (182 weeks x 3 zones x 6 classes) and 546 zone-week rows.
- **Classes:** maize, CCMT class names: leaf blight, leaf spot, streak virus (diseases); fall armyworm, grasshopper, leaf beetle (pests).
- **Outbreak pressure:** each class responds to the previous week's humidity and rainfall (standardised) and to a preferred temperature band, with week-to-week persistence (AR(1), 0.85) and random noise. A maize crop calendar (planted about November, harvested about May) scales detections; a small off-season value remains.
- **Spray decision:** when the field-average health index first falls below 85% in the crop season (at least 8 weeks since the last alert), Zone A is sprayed that week and Zone B 3 weeks later. 8 spray events result (4 per treated zone).
- **Spray effect (assumptions):** new infection is reduced by up to 75%, building up over about 2 weeks and fading over about 8, and the built-up infection load falls gradually over 4 weeks.
- **Counts:** binomial draws, each image has exactly one class. **Severity:** % leaf area affected, rising with detections plus noise.

## Trend analysis (what the script computes)
1. Weekly health index per zone, weekly change (spread rate) and a 3-week centred rolling average. The average is used for the figure only: it needs next week's value and blurs the spray week.
2. For each spray event (week s), a straight line is fitted to the raw weekly values for the 5 weeks up to the spray and the 6 weeks after it. The control zone is fitted over the same weeks. The comparison is the post-spray slope, treated minus control (documented method).
3. **Level comparison (recommended success measure):** change in mean health index from the 5 weeks before to the 6 weeks after the spray, treated minus control, with an approximate 95% confidence interval.
4. An early-warning flag (placeholder): a one-week fall in the health index of 10 points or more.

## Findings from the current data (simulated)
- **Level comparison:** treated zones gained on average **+8.7 points** more than the control zone over 8 events (positive in all 8; the 95% interval excludes zero in 4 of 8, all in Zone A). Zone B's later spray gains less because the control zone is already recovering.
- **Slope comparison:** mean **-1.1 points per week**, so the documented slope method does not show the effect. Over only 6 weekly points both treated and control zones rebound after a dip, and slopes are noisy.
- **Placebo run (spraying has no effect):** slope -0.04 and level -2.0 points over 12 events. The method does not invent an effect.
- Sprays follow a dip, so some rebound happens without spraying. This is why the untreated control zone is the comparison.
- The spray benefit is an assumption built into the simulation, so these numbers demonstrate the method, not that spraying works.

## Assumptions and limitations
- All risk coefficients, the crop calendar, the spray rule and the spray effect size are assumptions made by the team, not measured values.
- Health index = share of sampled leaves classed healthy. It is not the share of the zone's area, and it inherits the classifier's errors (the simulation treats detections as the model's output).
- A real farmer would not leave a whole zone untreated; a small control strip is more realistic.
- Weather values come from the MERRA-2 reanalysis grid (about 50 km cells), not a local station.
- The period covers about 3.5 years. To lengthen it, change `START_DATE` in the generator and run it without `--skip-download`.
- Only one location and one crop (maize) are simulated.
- The success-criteria targets in `time_series_trend_analysis.py` (2.0 points per week slope, 5.0 points level, 10-point alert drop) are placeholders for the team to decide.
