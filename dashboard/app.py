"""RegimeCast: a local market-risk research workspace."""
import os
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from regimecast.api.store import ArtifactStore, freshness

LABELS = {
    "historical": "Historical volatility", "ewma": "EWMA", "har": "HAR benchmark",
    "pooled": "Pooled model", "pooled_macro": "Pooled + macro",
    "gmm_mixture": "GMM experts", "hmm_mixture": "HMM experts",
    "gmm_pooled_probabilities": "Pooled + GMM states",
    "hmm_pooled_probabilities": "Pooled + HMM states",
}
ASSETS = {"SPY": "US large-cap equities", "QQQ": "Nasdaq-100 equities", "TLT": "Long-duration Treasuries", "GLD": "Gold"}
COLORS = ["#182f50", "#159f99", "#9282c4", "#e6ab63", "#8496b0"]
PRIMARY = ["pooled", "har", "gmm_mixture", "hmm_mixture", "pooled_macro"]

CSS = """
<style>
.stApp { background: #f5f7fb; color: #192c46; }
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stHeader"] { background: transparent; height: 2.8rem; }
.stAppDeployButton { display: none; }
.block-container { max-width: 1440px; padding: 2.8rem 2.8rem 3rem; }
[data-testid="stVerticalBlock"] { gap: 1rem; }
h1 { font-size: 2.65rem !important; letter-spacing: -.055em !important; font-weight: 700 !important; line-height: 1.15 !important; padding: .25rem 0 .4rem !important; }
h2, h3 { letter-spacing: -.025em; color: #192c46; }
h3 { font-size: 1.18rem !important; font-weight: 650 !important; padding-top: .2rem !important; }
p, label { line-height: 1.55; }
[data-testid="stCaptionContainer"] { color: #7a879a; font-size: .82rem; }
[data-testid="stSidebar"] { background: #10233e; border-right: 0; min-width: 248px; max-width: 248px; }
[data-testid="stSidebar"] > div:first-child { padding-top: 2.1rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h3 { color: #c9d4e4 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #8ea2bd !important; font-size: .78rem; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #1b3452; border-color: #34506d; color: #edf4ff; border-radius: 10px; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #b9cde3; }
[data-testid="stSidebar"] hr { border-color: #29415e; margin: 1.5rem 0; }
.brand { display:flex; align-items:center; gap:12px; margin:0 0 2.4rem; color:white; }
.brand-mark { width:36px; height:36px; display:grid; place-items:center; background:#21b4a5; border-radius:11px; font-size:23px; color:#082b34; }
.brand-name { font-size:21px; font-weight:700; letter-spacing:-.7px; }
.brand-sub { color:#88a0bd; font-size:10px; letter-spacing:1.8px; margin-top:3px; }
.side-label { font-size:10px; font-weight:650; color:#7892b3; letter-spacing:1.7px; margin-bottom:12px; }
.side-note { margin-top:14px; padding:14px; background:#172f4c; border:1px solid #2b4260; border-radius:12px; font-size:12px; color:#a6b9d0; line-height:1.7; }
.eyebrow { font-size:10px; font-weight:700; letter-spacing:2px; color:#5b788f; display:flex; align-items:center; gap:8px; }
.eyebrow-dot { width:6px; height:6px; background:#20a59b; border-radius:50%; }
.hero-meta { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.pill { display:inline-flex; align-items:center; gap:6px; border:1px solid #dce4ed; background:#fff; border-radius:30px; padding:6px 11px; font-size:11px; color:#65778e; }
.pill strong { color:#253e5b; font-weight:600; }
.hero-description { color:#718097; font-size:14px; margin:0 0 10px; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #dce3ed; padding-bottom:8px; margin:12px 0 18px; }
[data-testid="stTabs"] [data-baseweb="tab"] { height:40px; padding:0 18px; font-size:13px; color:#7b8ba0; border-radius:9px; font-weight:550; }
[data-testid="stTabs"] [aria-selected="true"] { background:#e5f2f0; color:#087a75 !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#e3e9f1 !important; border-radius:15px !important; background:white; box-shadow:0 3px 14px #223c5804; }
[data-testid="stMetricLabel"] { color:#304663; font-size:12px; font-weight:650; letter-spacing:.5px; }
[data-testid="stMetricValue"] { color:#152d4c; font-size:34px; font-weight:650; letter-spacing:-1.3px; padding:5px 0; }
[data-baseweb="select"] > div { border-color:#dce4ef; border-radius:10px; min-height:42px; background:white; }
[data-testid="stSelectbox"] label { font-size:11px; font-weight:600; color:#6f8197; }
[data-testid="stDownloadButton"] button { border-radius:9px; font-size:12px; border-color:#dce4ed; color:#3b5471; background:white; min-height:38px; }
[data-testid="stExpander"] { border-color:#e2e9f1; border-radius:12px; background:white; }
.premium-table-wrap { overflow-x:auto; border:1px solid #e5eaf1; border-radius:12px; background:white; margin:6px 0 10px; }
.premium-table { width:100%; border-collapse:collapse; font-size:12px; white-space:nowrap; }
.premium-table th { background:#f7f9fc; font-size:10px; letter-spacing:.5px; text-transform:uppercase; color:#7e8da1; font-weight:600; padding:13px 16px; text-align:right; border-bottom:1px solid #e5eaf1; }
.premium-table td { padding:13px 16px; color:#334b68; border-bottom:1px solid #eef2f6; text-align:right; font-variant-numeric:tabular-nums; }
.premium-table th:first-child, .premium-table td:first-child { text-align:left; font-weight:600; }
.premium-table tr:last-child td { border-bottom:0; }
.premium-table tr:hover td { background:#f8fafc; }
.state-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:6px 0 16px; }
.state-tile { padding:17px; background:#f7f9fc; border:1px solid #e7ecf3; border-radius:12px; }
.state-tile span { font-size:11px; color:#74869c; }
.state-tile strong { display:block; margin-top:7px; font-size:25px; color:#1a3556; letter-spacing:-.8px; font-variant-numeric:tabular-nums; }
.section-note { font-size:12px; line-height:1.7; color:#7b8ba0; margin:8px 0 14px; }
.footer { display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; border-top:1px solid #e0e7ef; padding-top:20px; margin-top:30px; color:#93a0b1; font-size:10px; letter-spacing:.3px; }
@media (max-width:1100px) { .block-container { padding-left:1.5rem; padding-right:1.5rem; } h1 { font-size:2.1rem !important; } }
@media (max-width:760px) { .block-container { padding:2rem 1rem; } .state-grid { gap:6px; } .state-tile { padding:12px; } .premium-table td,.premium-table th { padding:10px; } }
</style>
"""


def label(model):
    return LABELS.get(model, model)


def table(frame):
    st.markdown('<div class="premium-table-wrap">'+frame.to_html(index=False, border=0, classes="premium-table", escape=True)+'</div>', unsafe_allow_html=True)


def chart_style(chart, height=300):
    return chart.properties(height=height).configure_view(strokeWidth=0).configure_axis(
        labelColor="#7a8a9e", titleColor="#8796a8", labelFontSize=11,
        titleFontSize=10, domain=False, tickColor="#e6ebf2", gridColor="#edf1f6",
        labelPadding=9, titlePadding=12,
    ).configure_legend(labelColor="#65788f", labelFontSize=11, title=None, orient="bottom", padding=12)


def probability_tiles(probabilities):
    tiles=''.join(f'<div class="state-tile"><span>STATE {i:02d}</span><strong>{100*p:.1f}%</strong></div>' for i,p in enumerate(probabilities))
    st.markdown(f'<div class="state-grid">{tiles}</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="RegimeCast | Market risk", page_icon="◈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    store=ArtifactStore(Path(os.environ.get("REGIMECAST_CATALOG", "artifacts/catalog.json")))
    try:
        periods=store.periods()
    except (OSError, ValueError, KeyError):
        st.info("Your research workspace is not ready yet. Follow the setup guide in the README.")
        st.stop()
    try:
        latest=store.latest()
    except (OSError, ValueError, KeyError):
        latest=None
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">RegimeCast</div><div class="brand-sub">MARKET INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-label">WORKSPACE</div>', unsafe_allow_html=True)
        period=st.selectbox("Research sample", periods, format_func=lambda p:"Development · 2013–2023" if p=="development" else "Holdout · 2024–2025")
        st.markdown('<div class="side-note">A focused view of market risk.<br>Four liquid assets. One forecast horizon. Evidence you can inspect.</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="side-label">RESEARCH STANDARD</div>', unsafe_allow_html=True)
        st.caption("Time-ordered evaluation. Historical information only. Lower forecast error is better.")
        st.caption("Research results do not imply trading profitability.")
    st.markdown('<div class="eyebrow"><span class="eyebrow-dot"></span> THE MARKET RISK WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Risk, in perspective.")
    st.markdown('<div class="hero-description">A measured view of next-week volatility, market states, and the evidence behind every model.</div>', unsafe_allow_html=True)
    if latest:
        st.markdown(f'<div class="hero-meta"><span class="pill">AS OF <strong>{escape(latest["as_of"])}</strong></span><span class="pill">HORIZON <strong>5 trading sessions</strong></span><span class="pill">COVERAGE <strong>SPY · QQQ · TLT · GLD</strong></span></div>', unsafe_allow_html=True)
        if latest.get("demo"):
            st.warning("DEMO — synthetic data. These values are not market forecasts.")
    else:
        st.info("Latest forecasts are being prepared. You can still explore historical research below.")
    forecast_tab, state_tab, research_tab, explain_tab=st.tabs(["Overview", "Market states", "Model performance", "Forecast drivers"])

    with forecast_tab:
        if latest:
            heading, control=st.columns([2,1], vertical_alignment="bottom")
            frame=pd.DataFrame(latest["forecasts"])
            models=frame.model.unique().tolist()
            with heading:
                st.subheader("The week ahead")
                st.caption(f"Annualized volatility · Forecast horizon ends {latest['target_end']}")
            with control:
                model=st.selectbox("Forecast model", models, index=models.index("har") if "har" in models else 0, format_func=label)
            if freshness(latest)["stale"]:
                st.warning("This snapshot is behind the latest completed session. Refresh the local data for current forecasts.")
            selected=frame[frame.model.eq(model)].set_index("asset").reindex([a for a in ASSETS if a in set(frame.asset)]).reset_index()
            for column,row in zip(st.columns(len(selected)), selected.itertuples()):
                with column:
                    with st.container(border=True):
                        st.metric(row.asset, f"{row.volatility_pct:.1f}%")
                        st.caption(ASSETS.get(row.asset,row.asset))
            left,right=st.columns([1.7,1], gap="medium")
            with left:
                with st.container(border=True):
                    st.subheader("Forecasts in context")
                    st.caption("The selected model alongside familiar benchmarks")
                    comparison=frame[frame.model.isin({model,"har","ewma"})].copy()
                    comparison["Model"]=comparison.model.map(label)
                    graph=alt.Chart(comparison).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4,size=19).encode(
                        x=alt.X("asset:N",title=None,sort=list(ASSETS)), xOffset="Model:N",
                        y=alt.Y("volatility_pct:Q",title="Annualized volatility (%)"),
                        color=alt.Color("Model:N",scale=alt.Scale(range=COLORS)),
                        tooltip=[alt.Tooltip("asset:N",title="Asset"),"Model:N",alt.Tooltip("volatility_pct:Q",title="Volatility (%)",format=".1f")],
                    )
                    st.altair_chart(chart_style(graph),use_container_width=True)
            with right:
                with st.container(border=True):
                    st.subheader("At a glance")
                    st.caption("All assets · Same forecast horizon")
                    summary=selected[["asset","volatility_pct"]].copy()
                    summary.columns=["Asset","Forecast"]
                    summary["Forecast"]=summary.Forecast.map(lambda v:f"{v:.1f}%")
                    summary["Exposure"]=summary.Asset.map(ASSETS)
                    table(summary[["Asset","Exposure","Forecast"]])
                    st.markdown('<div class="section-note">Volatility measures the magnitude of potential price moves, rather than their direction.</div>',unsafe_allow_html=True)
                    st.download_button("↓  Export forecasts",frame.to_csv(index=False),"regimecast-forecasts.csv","text/csv")

    with state_tab:
        if latest and latest.get("regimes"):
            heading,control=st.columns([2,1],vertical_alignment="bottom")
            with heading:
                st.subheader("The market’s changing character")
                st.caption("Estimated probabilities across statistical market states")
            with control:
                kind=st.selectbox("State estimator",list(latest["regimes"]),format_func=str.upper)
            gate=latest["regimes"][kind]
            probability_tiles(gate["probabilities"])
            if not gate.get("reliable",True):
                st.warning("The current state estimate is unreliable. The pooled-model fallback is active.")
            with st.container(border=True):
                st.subheader("State probabilities over time")
                st.caption("Most recent 504 sessions · Each date sums to 100%")
                try:
                    history=store.regime_history(kind).tail(504)
                    cols=[c for c in history if c.startswith("regime_")]
                    long=history.melt(id_vars=["date"],value_vars=cols,var_name="State",value_name="Probability")
                    long["State"]=long.State.map(lambda s:"State "+s.rsplit("_",1)[1])
                    graph=alt.Chart(long).mark_area(opacity=.88).encode(
                        x=alt.X("date:T",title=None), y=alt.Y("Probability:Q",stack="zero",axis=alt.Axis(format="%"),title=None),
                        color=alt.Color("State:N",scale=alt.Scale(range=COLORS)),
                        tooltip=[alt.Tooltip("date:T",title="Date"),"State:N",alt.Tooltip("Probability:Q",format=".1%")],
                    )
                    st.altair_chart(chart_style(graph,340),use_container_width=True)
                except (OSError,ValueError,KeyError):
                    st.info("The state timeline is not available yet.")
            st.caption("States are model estimates. Their probabilities are not forecast confidence intervals.")

    with research_tab:
        st.subheader("Evidence over intuition")
        st.caption("Matched evaluation dates, common assets, and transparent errors")
        try:
            metrics=store.metrics(period)
            scores=metrics.groupby("model").agg(QLIKE=("qlike","mean"),MAE=("volatility_mae_pct_points","mean"),RMSE=("volatility_rmse_pct_points","mean")).reset_index()
            scores["Model"]=scores.model.map(label)
            a,b=st.columns([1.2,1],gap="medium")
            with a:
                with st.container(border=True):
                    st.subheader("Model leaderboard")
                    st.caption("Average QLIKE · Lower is better")
                    graph=alt.Chart(scores).mark_bar(cornerRadiusEnd=4,size=16,color=COLORS[1]).encode(
                        x=alt.X("QLIKE:Q",title=None),y=alt.Y("Model:N",sort="x",title=None,axis=alt.Axis(labelLimit=230)),
                        tooltip=["Model:N",alt.Tooltip("QLIKE:Q",format=".4f")],
                    )
                    st.altair_chart(chart_style(graph,330),use_container_width=True)
            with b:
                with st.container(border=True):
                    st.subheader("The scorecard")
                    st.caption("Equal asset weights · Annualized volatility errors")
                    display=scores.sort_values("QLIKE")[["Model","QLIKE","MAE","RMSE"]].copy()
                    for c in ["QLIKE","MAE","RMSE"]:
                        display[c]=display[c].map(lambda v:f"{v:.3f}")
                    table(display)
            try:
                analysis=store.analysis(period)
                with st.container(border=True):
                    st.subheader("Consistency across years")
                    annual=pd.read_csv(analysis/"annual_qlike.csv")
                    annual=annual[annual.model.isin(PRIMARY)].copy()
                    annual["Model"]=annual.model.map(label)
                    graph=alt.Chart(annual).mark_line(point=True,strokeWidth=2).encode(
                        x=alt.X("year:O",title=None),y=alt.Y("qlike:Q",title="QLIKE"),
                        color=alt.Color("Model:N",scale=alt.Scale(range=COLORS)),tooltip=["year:O","Model:N",alt.Tooltip("qlike:Q",format=".4f")],
                    )
                    st.altair_chart(chart_style(graph,280),use_container_width=True)
                with st.expander("Comparison uncertainty and largest errors"):
                    uncertainty=pd.read_csv(analysis/"paired_bootstrap.csv")
                    uncertainty=uncertainty[uncertainty.block_sessions.eq(21)][["model","mean_loss_difference","ci95_lower","ci95_upper"]].copy()
                    uncertainty["model"]=uncertainty.model.map(label)
                    uncertainty.columns=["Model","Difference","95% lower","95% upper"]
                    for c in uncertainty.columns[1:]:
                        uncertainty[c]=uncertainty[c].map(lambda v:f"{v:+.4f}")
                    table(uncertainty)
                    st.caption("Paired date-block bootstrap. Negative differences favor the candidate. Intervals are exploratory and unadjusted for multiple comparisons.")
                    worst=pd.read_csv(analysis/"worst_errors.csv").head(10).copy()
                    worst["Date"]=pd.to_datetime(worst.date).dt.strftime("%d %b %Y")
                    worst["Model"]=worst.model.map(label)
                    worst["Forecast"]=worst.forecast_variance.pow(.5).mul(100).map(lambda v:f"{v:.1f}%")
                    worst["Realized"]=worst.target_variance.pow(.5).mul(100).map(lambda v:f"{v:.1f}%")
                    table(worst.rename(columns={"asset":"Asset"})[["Date","Asset","Model","Forecast","Realized"]])
            except (OSError,ValueError,KeyError):
                st.caption("Detailed period and uncertainty reports are not yet available for this data version.")
            with st.expander("Per-asset results"):
                detail=metrics.copy()
                detail["model"]=detail.model.map(label)
                detail=detail.rename(columns={"model":"Model","asset":"Asset","observations":"Forecasts","qlike":"QLIKE","volatility_mae_pct_points":"MAE (pp)","volatility_rmse_pct_points":"RMSE (pp)"})
                st.dataframe(detail.style.format({"QLIKE":"{:.4f}","MAE (pp)":"{:.2f}","RMSE (pp)":"{:.2f}"}),width="stretch",hide_index=True)
            st.download_button("↓  Export research metrics",metrics.to_csv(index=False),"regimecast-research.csv","text/csv")
        except (OSError,ValueError,KeyError):
            st.info("This evaluation period is not ready yet.")

    with explain_tab:
        st.subheader("Inside the forecast")
        st.caption("See how each market state contributes to a single asset’s risk estimate")
        if latest:
            contributions=pd.DataFrame(latest.get("expert_contributions",[]))
            if not contributions.empty:
                c1,c2=st.columns(2)
                with c1:
                    asset=st.selectbox("Asset",[a for a in ASSETS if a in set(contributions.asset)],key="driver_asset")
                with c2:
                    gate_kind=st.selectbox("Expert model",contributions.gate.unique().tolist(),format_func=lambda s:s.upper()+" experts",key="driver_gate")
                chosen=contributions[contributions.asset.eq(asset)&contributions.gate.eq(gate_kind)].sort_values("state").copy()
                probability_tiles(chosen.probability.tolist())
                left,right=st.columns([1,1.4],gap="medium")
                with left:
                    with st.container(border=True):
                        st.subheader("Forecast composition")
                        st.caption("Share of predicted variance by state")
                        total=chosen.weighted_variance.sum()
                        chosen["Share"]=chosen.weighted_variance/total if total>0 else 0.
                        chosen["State"]=chosen.state.map(lambda s:f"State {int(s)}")
                        graph=alt.Chart(chosen).mark_bar(cornerRadiusEnd=4,size=24).encode(
                            x=alt.X("Share:Q",axis=alt.Axis(format="%"),title=None,scale=alt.Scale(domain=[0,1])),
                            y=alt.Y("State:N",title=None),color=alt.Color("State:N",scale=alt.Scale(range=COLORS),legend=None),
                            tooltip=["State:N",alt.Tooltip("Share:Q",format=".1%")],
                        )
                        st.altair_chart(chart_style(graph,210),use_container_width=True)
                with right:
                    with st.container(border=True):
                        st.subheader("Expert breakdown")
                        st.caption(f"{asset} · {gate_kind.upper()} · {latest['as_of']}")
                        detail=pd.DataFrame({"State":chosen.State,
                            "Probability":chosen.probability.map(lambda v:"<0.1%" if 0<v<.001 else f"{v:.1%}"),
                            "Expert volatility":chosen.expert_variance.pow(.5).mul(100).map(lambda v:f"{v:.1f}%"),
                            "Variance share":chosen.Share.map(lambda v:"<0.1%" if 0<v<.001 else f"{v:.1%}"),
                            "Source":chosen.fallback.map(lambda v:"Pooled fallback" if v else "State expert")})
                        table(detail)
                        st.markdown('<div class="section-note">Experts contribute to variance in proportion to their state probabilities. Volatility is calculated from the combined variance.</div>',unsafe_allow_html=True)
            context=pd.DataFrame(latest.get("feature_context",[]))
            if not context.empty:
                with st.container(border=True):
                    st.subheader("The observed backdrop")
                    st.caption("Trailing measurements available at the forecast date")
                    columns={"asset":"Asset","trailing_21_session_volatility_pct":"21-session volatility","five_session_return_pct":"5-session return","63_session_drawdown_pct":"63-session drawdown"}
                    display=context[[c for c in columns if c in context]].rename(columns=columns).copy()
                    for c in display.columns[1:]:
                        display[c]=display[c].map(lambda v:"—" if pd.isna(v) else f"{v:.1f}%")
                    table(display)
            st.caption("Model contributions describe how the forecast was assembled. They do not establish causal effects.")
            with st.expander("Data provenance and model configuration"):
                st.json({"data":latest.get("data_metadata",{}),"models":latest.get("model_metadata",{})})
    st.markdown('<div class="footer"><span>REGIMECAST · Independent market-risk research</span><span>Five sessions ahead. Evidence first.</span></div>',unsafe_allow_html=True)


if __name__=="__main__":
    main()
