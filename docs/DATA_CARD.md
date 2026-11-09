# Data card

## Prices

- Assets: SPY, QQQ, TLT and GLD.
- Provider adapter: Yahoo via yfinance, plus canonical local CSV import.
- Validated release snapshot: 20,980 observations, 5,245 sessions per asset, 2006-01-03 through 2026-11-06.
- Convention: retrospective split/dividend-adjusted closes. Historical point-in-time price vintages are not claimed.
- Checks: unique asset/date keys, finite positive prices, sorted timezone-naive session labels, full requested XNYS coverage, suspicious jumps and stale-price warnings.
- Missing asset prices are never filled. Raw normalized Parquet snapshots are immutable and checksummed.

## Macroeconomic data

- Source: FRED/ALFRED, using a free locally configured API key.
- Series: DGS2, DGS10, UNRATE and CPIAUCSL.
- Release snapshot: 13,013 historical vintage records.
- Sparse new/revised observations are requested in four-year vintage batches to respect API limits.
- A value becomes usable on the first XNYS session strictly after its vintage date. Revisions to older observations never displace a newer observation period.
- Features include levels, changes, missingness, release age, inflation year-over-year and yield-curve changes. Missing macro values remain explicit; boosting handles missing values natively.

## Storage and access

Data, fitted models and local MLflow databases stay under ignored `data/` and `artifacts/` folders. Credentials stay in ignored `.env` or environment variables. Git contains manifests and aggregate research results rather than raw provider histories. Source terms apply; do not assume permission to redistribute raw data.

Official references: [FRED observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), [ALFRED](https://fred.stlouisfed.org/docs/api/fred/alfred.html), [yfinance download behavior](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
