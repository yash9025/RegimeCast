"""RegimeCast local dashboard, backed entirely by saved artifacts."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from regimecast.api.store import ArtifactStore, freshness

LABELS = {
    "historical": "Historical volatility",
    "ewma": "EWMA",
    "har": "HAR benchmark",
    "pooled": "Pooled model",
    "pooled_macro": "Pooled + macro",
    "gmm_mixture": "GMM experts",
    "hmm_mixture": "HMM experts",
    "gmm_pooled_probabilities": "Pooled + GMM probabilities",
    "hmm_pooled_probabilities": "Pooled + HMM probabilities",
}


def main():
    st.set_page_config(page_title="RegimeCast", page_icon="◈", layout="wide")
    st.caption("REGIMECAST / MARKET RISK RESEARCH")
    st.title("One week ahead.")
    st.write(
        "Volatility forecasts, estimated market states, and the evidence behind them."
    )
    store = ArtifactStore(
        Path(os.environ.get("REGIMECAST_CATALOG", "artifacts/catalog.json"))
    )
    try:
        periods = store.periods()
    except (OSError, ValueError, KeyError):
        st.info(
            "Local research artifacts are not ready yet. Follow the setup guide in the project README."
        )
        st.stop()
    try:
        latest = store.latest()
    except (OSError, ValueError, KeyError):
        latest = None
        st.info(
            "Latest forecasts are not ready. Historical research is available below."
        )
    if latest and latest.get("demo"):
        st.warning("DEMO — synthetic data. These values are not market forecasts.")
    with st.sidebar:
        st.subheader("Research view")
        period = st.selectbox(
            "Evaluation period",
            periods,
            format_func=lambda p: (
                "Development · 2013–2023"
                if p == "development"
                else "Holdout · 2024–2025"
            ),
        )
        st.caption(
            "Lower forecast error is better. Research results do not imply trading profitability."
        )
        st.divider()
        st.caption("Runs locally · Four assets · Five trading sessions")
    forecasts_tab, regimes_tab, research_tab, explanations_tab = st.tabs(
        ["Forecasts", "Market states", "Research evidence", "Explanations"]
    )
    with forecasts_tab:
        if latest:
            status = freshness(latest)
            st.caption(
                f"As of {latest['as_of']} · Five-session horizon ends {latest['target_end']} · Annualized volatility"
            )
            if status["stale"]:
                st.warning(
                    "Forecast data is behind the latest completed market session. Refresh the local data before relying on this view."
                )
            frame = pd.DataFrame(latest["forecasts"])
            models = frame.model.unique().tolist()
            model = st.selectbox(
                "Forecast model",
                models,
                index=models.index("har") if "har" in models else 0,
                format_func=lambda m: LABELS.get(m, m),
            )
            selected = frame[frame.model.eq(model)]
            columns = st.columns(len(selected))
            for column, row in zip(columns, selected.itertuples()):
                with column:
                    with st.container(border=True):
                        st.metric(row.asset, f"{row.volatility_pct:.1f}%")
                        st.caption("Annualized volatility")
            comparison = frame[frame.model.isin({model, "har", "ewma"})].copy()
            comparison["model"] = comparison.model.map(lambda m: LABELS.get(m, m))
            st.subheader("Forecast comparison")
            st.bar_chart(
                comparison.pivot(
                    index="asset", columns="model", values="volatility_pct"
                ),
                stack=False,
            )
            st.caption(
                "All forecasts target the same next five sessions. Market-state probabilities are not forecast confidence intervals."
            )
            st.download_button(
                "Download forecasts",
                frame.to_csv(index=False),
                "regimecast-forecasts.csv",
                "text/csv",
            )
    with regimes_tab:
        if latest and latest.get("regimes"):
            kind = st.selectbox(
                "State estimator", list(latest["regimes"]), format_func=str.upper
            )
            gate = latest["regimes"][kind]
            probability = pd.DataFrame(
                {
                    "State": [f"State {i}" for i in range(len(gate["probabilities"]))],
                    "Probability": gate["probabilities"],
                }
            )
            st.bar_chart(probability.set_index("State"))
            if not gate.get("reliable", True):
                st.warning(
                    "The current state estimate is unreliable. Forecasting uses the pooled-model fallback."
                )
            try:
                historical = store.regime_history(kind).tail(504)
                state_columns = [c for c in historical if c.startswith("regime_")]
                st.subheader("How state probabilities changed")
                st.area_chart(historical.set_index("date")[state_columns])
            except (OSError, ValueError, KeyError):
                st.info("The state timeline is not available yet.")
            st.caption(
                "States are statistical estimates. Their names do not establish an independently observed economic condition."
            )
    with research_tab:
        st.subheader("Does added complexity help?")
        try:
            metrics = store.metrics(period)
            scores = metrics.groupby("model").agg(
                QLIKE=("qlike", "mean"),
                MAE=("volatility_mae_pct_points", "mean"),
                RMSE=("volatility_rmse_pct_points", "mean"),
            )
            scores.index = scores.index.map(lambda m: LABELS.get(m, m))
            st.bar_chart(scores[["QLIKE"]].sort_values("QLIKE"))
            st.dataframe(scores.sort_values("QLIKE"), width="stretch")
            st.caption(
                "MAE and RMSE are annualized volatility percentage points. Scores average the four assets equally."
            )
            st.subheader("Errors by asset")
            st.dataframe(metrics, width="stretch", hide_index=True)
            try:
                analysis = store.analysis(period)
                annual = pd.read_csv(analysis / "annual_qlike.csv")
                st.subheader("Consistency across years")
                st.line_chart(
                    annual.pivot(index="year", columns="model", values="qlike")
                )
                uncertainty = pd.read_csv(analysis / "paired_bootstrap.csv")
                st.subheader("Uncertainty in the comparison")
                st.dataframe(
                    uncertainty[uncertainty.block_sessions.eq(21)],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Paired date-block bootstrap; negative differences favor the candidate. Intervals are exploratory and unadjusted for multiple comparisons."
                )
                st.subheader("Largest forecast errors")
                st.dataframe(
                    pd.read_csv(analysis / "worst_errors.csv").head(20),
                    width="stretch",
                    hide_index=True,
                )
            except (OSError, ValueError, KeyError):
                st.caption(
                    "Detailed uncertainty and period reports are not ready for this view."
                )
            st.download_button(
                "Download research metrics",
                metrics.to_csv(index=False),
                "regimecast-research.csv",
                "text/csv",
            )
        except (OSError, ValueError, KeyError):
            st.info("This evaluation period is not ready yet.")
    with explanations_tab:
        if latest:
            st.subheader("What contributed to the forecast?")
            st.write(
                "Each expert’s contribution depends on the estimated probability of its market state."
            )
            contributions = pd.DataFrame(latest.get("expert_contributions", []))
            if not contributions.empty:
                st.dataframe(contributions, width="stretch", hide_index=True)
            st.subheader("Observed risk context")
            context = pd.DataFrame(latest.get("feature_context", []))
            if not context.empty:
                st.dataframe(context, width="stretch", hide_index=True)
            st.caption(
                "Contributions and observed features describe the fitted models. They do not establish causal effects."
            )
            with st.expander("Data and model details"):
                st.json(
                    {
                        "data": latest.get("data_metadata", {}),
                        "models": latest.get("model_metadata", {}),
                    }
                )


if __name__ == "__main__":
    main()
