# wateraudit

Soft sensors for effluent BOD and COD at a municipal wastewater treatment plant, with 90% prediction intervals and a compliance risk index, tested in time order on the UCI Water Treatment Plant dataset.

![Test-period dashboard](samples/water_audit_dashboard.png)

*Test period, May to October 1991. The top two panels compare lab BOD and COD with the predicted median and the 90% conformal interval. The bottom panel is the risk index computed from the forecast alone.*

---

## Why a soft sensor

A BOD test needs five days of incubation. By the time the lab reports that effluent broke the discharge limit, that water left the plant days earlier. A soft sensor estimates the value from what is known now: online probes, same-day lab tests, and whichever earlier effluent results have already come back.

## Data

The [UCI Water Treatment Plant dataset](https://archive.ics.uci.edu/dataset/106/water+treatment+plant): 527 daily records from an urban plant between January 1990 and October 1991, with 38 attributes across the inlet (`-E`), primary settler (`-P`), secondary settler (`-D`) and outlet (`-S`). Some days are missing. The file is included in `data/`.

The raw file stores the records in monthly blocks, and the blocks are not in calendar order. The loader sorts by date before anything time-based is computed.

## Features, by when they are known

| Tier | Columns | Known |
|---|---|---|
| 0: online probes | flow `Q-E`; pH and conductivity at all four stages | continuously |
| 1: same-day lab tests | sediments `SED-*`, volatile suspended solids `SSV-*`, zinc `ZN-E` | the same day |
| 2: earlier effluent results | COD and SS from the previous record, BOD from five records back | when each test comes back |

One-record lags and 7-record rolling means of inlet flow, conductivity and pH are added on top, along with inlet flow relative to its 7-record median.

The `RD-*` columns are never used. They are removal efficiencies computed from the same-day effluent values the model is predicting, so they would leak the answer. `core/schema.py` refuses them, and a test checks that the tier-2 features only carry results that were available at prediction time.

## Method

**Soft sensors.** For each target, three LightGBM models predict the 5th, 50th and 95th percentiles of `log(1 + y)`. Conformalized quantile regression (CQR) then widens that raw interval using a held-out calibration set: each calibration record is scored by how far its true value falls outside the raw interval, and the interval is widened by the 90th-percentile score, with the usual finite-sample correction and a minimum of 0.05 on the log scale.

The 90% guarantee of CQR is marginal and assumes calibration and test records are exchangeable. A test on i.i.d. synthetic data checks that the implementation reaches it. The results below show what happens on the plant, where time order breaks that assumption.

**Risk index.** Each soft sensor converts its interval into a rough probability of exceeding the limit. The index combines those with the online pH reading as `1 − Π(1 − p)`, and pushes it higher when a predicted median is already over its limit. Bands: GREEN below 0.20, then WATCH, ACT from 0.50, and RED from 0.80.

Limits and bands are read from `configs/limits.yaml`. BOD 25 mg/L, COD 125 mg/L and SS 35 mg/L are the concentration limits of EU Directive 91/271/EEC. The directive sets no pH limit, so 6.5–8.5 is an assumed operating band.

**Stage diagnostics.** Suspended-solids removal is split into primary and secondary stages as log ratios, with a CUSUM on the secondary stage. A record whose diagnostic vector leaves the normal range is matched against hand-set signatures: hydraulic shock, sludge bulking, toxic shock, and primary settler fault. The normal range is set from the training period. The dataset has no fault labels, so this part is a rule-based sketch, not a validated classifier.

---

## Results

The records are split in time order. The first 316 (January 1990 to January 1991) are used for training, the next 105 (to May 1991) for calibration, and the last 106 (May to October 1991) for testing.

| | BOD | COD |
|---|---:|---:|
| Coverage of the 90% interval | 79.0% | 90.2% |
| Mean absolute error of the predicted median | 6.9 mg/L | 28.5 mg/L |
| Mean absolute error of the latest known lab value | 10.0 mg/L | 30.5 mg/L |

The predicted median beats carrying forward the latest available lab result: by about 30% for BOD, and only slightly for COD.

**BOD coverage falls short of 90% because time order breaks exchangeability.** The intervals are sized on January to May 1991, when effluent BOD never went above 33 mg/L, and then applied to the following five months. Those months include an event on 17–19 July 1991 with effluent BOD above 100 mg/L. With a shuffled split, which restores exchangeability, BOD coverage over five seeds is 92–95% and COD coverage is 86–95%, with means of 94% and 90%.

**As an early warning, the index is weak.** The lab later confirmed a BOD or COD breach on 14 test records. The forecast alone put 4 of them in ACT, and 6 of the 10 ACT records had no breach. The index ranks breach records above the others with an AUROC of 0.60. It reaches ACT on the second and third days of the July event, but most isolated exceedances stay at WATCH. No test record is GREEN, so at these thresholds WATCH carries no information.

These figures come from the forecast alone. An earlier version fed measured breaches into the index, which flags every breach by definition. The index is now scored without the lab result.

Stage diagnostics put 103 of the 106 test records inside the normal range, and the CUSUM does not alarm.

All figures come from `results/summary_chronological.json` and `results/summary_random_seed*.json`.

---

## Limits

- One plant, 106 test records, and 14 breaches. Every percentage above carries wide uncertainty.
- The limits are one reasonable choice, not the permit this plant operated under.
- The available measurements explain part of the effluent variation, not most of it. Aeration-tank biology, influent composition and weather are not in the data.
- The stage diagnostics are not validated against labelled faults.

## Run it

```bash
pip install -r requirements.txt
python scripts/run_audit.py                          # time-ordered split; writes the dashboard
python scripts/run_audit.py --split random --seed 1  # shuffled split, for comparison
pytest -q
```

## Tests

There are fourteen tests. These are the ones worth naming:

- **Coverage holds when its assumption does.** On i.i.d. synthetic data, the 90% interval covers between 87% and 93%.
- **A forecast is scored without the lab result.** The same prediction is RED when the measured breach is passed in, and is not RED without it.
- **Lagged features only carry results that were already available.** COD and SS are shifted by one record, BOD by five.
- **Records come back in calendar order.** Lags and the split depend on it.
- **An ordinary day comes out as normal.** Cosine similarity alone names a fault for every day. The deviation gate is what lets a nominal day through.
- **Limits come from the config file.**

## Data license

The UCI Water Treatment Plant dataset was created by Manel Poch and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [doi:10.24432/C5FS4C](https://doi.org/10.24432/C5FS4C).
