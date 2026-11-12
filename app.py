"""MerchantPulse AI: PayPal Sandbox payments + Gemini-powered merchant intelligence.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

from app_state import clear_pending_order
from demo_data import make_demo_transactions
from detectors import findings_in_range, run_all_detectors, summarize_impact
from merchant_insights import (
    MerchantAIError,
    answer_merchant_question,
    answer_merchant_question_ai,
    get_gemini_model,
    gemini_key_configured,
    gemini_ready,
    gemini_sdk_available,
)
from paypal_integration import (
    REFUND_WINDOW_DAYS,
    SANDBOX_CATALOG,
    PayPalIntegrationError,
    capture_order,
    check_refund_policy,
    create_order,
    credentials_configured,
    create_decision_receipt,
    get_order,
    init_database,
    list_decision_receipts,
    list_completed_payments,
    refund_capture,
    update_decision_receipt,
)
from refund_agent import (
    MAX_COMPLAINT_LEN,
    finalize_proposal,
    refund_approval_allowed,
    triage_refund_request,
)
from security import allow_event, approval_pin_required, clean_text, mask_email, verify_approval_pin

# Chart colours: the validated categorical order (blue, orange, aqua) and the fixed
# status palette. Status colours are always paired with a text label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STATUS_COLORS = {"Completed": "#0ca30c", "Pending": "#fab219", "Failed": "#d03b3b", "Refunded": "#ec835a"}
METHOD_COLORS = {"PayPal": BLUE, "Card": ORANGE, "Pay Later": AQUA}
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#ebeae6", "#fcfcfb"
AI_CALLS_PER_10_MIN = int(os.getenv("AI_CALLS_PER_10_MIN", "20") or 20)
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501").strip().rstrip("/")
IS_LOCAL_DEPLOYMENT = APP_BASE_URL.startswith(("http://localhost", "http://127.0.0.1"))

PLAYBOOK = {
    "failure_spike": [
        "Check the payment processor / gateway status page for that day and time window.",
        "Look at decline codes for the failed attempts (they are not in this synthetic dataset).",
        "Contact affected buyers with a retry link or an alternative payment method.",
    ],
    "refund_burst": [
        "Check whether a batch of orders shipped late, damaged, or with a listing error.",
        "Read the refund reasons and customer messages for the affected category.",
        "Pause promotion of the affected products until the cause is understood.",
    ],
    "offhours_high_value": [
        "Verify these buyers and shipping addresses before fulfilling the orders.",
        "Check for repeated cards, devices or addresses across the flagged orders.",
        "Consider a manual review rule for overnight orders above this amount.",
    ],
}

st.set_page_config(page_title="MerchantPulse AI", page_icon="\U0001F4CA", layout="wide", initial_sidebar_state="expanded")

# Static styling only: no user-supplied text is ever placed into this HTML.
st.markdown(
    """
    <style>
      .block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1280px; }
      .mp-hero { padding: 18px 22px; border-radius: 14px; margin-bottom: 14px;
        background: linear-gradient(120deg, #0d366b 0%, #1c5cab 60%, #2a78d6 100%); color: #fff; }
      .mp-hero h1 { margin: 0; font-size: 1.9rem; font-weight: 700; color: #fff; }
      .mp-hero p { margin: 4px 0 10px 0; font-size: 1.02rem; color: #e6f0fd; }
      .mp-pill { display: inline-block; padding: 3px 11px; margin-right: 7px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; background: rgba(255,255,255,0.16); color: #fff;
        border: 1px solid rgba(255,255,255,0.28); }
      .mp-note { padding: 9px 14px; border-radius: 10px; border: 1px solid rgba(235,104,52,0.45);
        background: rgba(235,104,52,0.07); margin-bottom: 14px; font-size: 0.9rem; }
      button[data-baseweb="tab"] { font-weight: 600; }
    </style>
    <div class="mp-hero">
      <h1>MerchantPulse AI</h1>
      <p>Spot payment problems, understand them with AI, and fix them, with a human approving every refund.</p>
      <span class="mp-pill">PayPal Sandbox</span><span class="mp-pill">Gemini</span><span class="mp-pill">Human-approved actions</span>
    </div>
    <div class="mp-note"><strong>Data notice:</strong> the analytics dataset is synthetic demo data with planted incidents.
    PayPal Sandbox captures are real API test transactions, shown separately. Sandbox payments move no real money.</div>
    """,
    unsafe_allow_html=True,
)


def ai_allowed() -> bool:
    """Per-session cap on Gemini calls to protect quota."""
    return allow_event(
        st.session_state.setdefault("ai_events", []), max_events=AI_CALLS_PER_10_MIN, window_seconds=600
    )


def style_chart(fig, height: int = 320, legend: bool = False):
    fig.update_layout(
        title_text="", height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_SOFT, size=12), hoverlabel=dict(bgcolor=SURFACE, font_color=INK, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d9d8d3", tickcolor="#d9d8d3", title_text=None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_font=dict(size=12))
    return fig


def show_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def order_rows_top_down(fig, labels) -> None:
    """Horizontal bars: show `labels` top to bottom (plotly draws the first category at the bottom)."""
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(list(labels))), title_text=None)


@st.cache_data
def load_transactions() -> pd.DataFrame:
    return make_demo_transactions()


@st.cache_data
def load_findings(data: pd.DataFrame) -> list[dict]:
    return run_all_detectors(data)


@st.cache_data
def add_anomaly_predictions(data: pd.DataFrame, contamination_pct: int) -> pd.DataFrame:
    """Isolation Forest: flag transactions unlike the rest of the demo data."""
    result = data.copy()
    features = pd.DataFrame(
        {
            "log_amount": np.log1p(result["Amount (USD)"].astype(float)),
            "hour": result["Date"].dt.hour.astype(float),
            "weekday": result["Date"].dt.dayofweek.astype(float),
        },
        index=result.index,
    )
    features = pd.concat([features, pd.get_dummies(result[["Category", "Payment Method"]], dtype=float)], axis=1)
    model = IsolationForest(n_estimators=150, contamination=contamination_pct / 100.0, random_state=42)
    predictions = model.fit_predict(features)
    result["Anomaly Score"] = -model.decision_function(features)  # higher = more unusual
    result["ML Anomaly"] = predictions == -1
    return result


def money(value: float) -> str:
    return f"${value / 1000:.1f}K" if abs(value) >= 1000 else f"${value:,.0f}"


def pct_change(now: float, before: float) -> str | None:
    return f"{(now - before) / before * 100:+.1f}%" if before else None


def kpis(frame: pd.DataFrame) -> dict[str, float]:
    completed = frame.loc[frame["Status"] == "Completed", "Amount (USD)"]
    failed = int((frame["Status"] == "Failed").sum())
    settled = len(completed) + failed
    return {
        "volume": float(completed.sum()),
        "success": len(completed) / settled * 100 if settled else 0.0,
        "refunds": float(frame.loc[frame["Status"] == "Refunded", "Amount (USD)"].sum()),
        "count": float(len(frame)),
    }


# --- PayPal redirect handling -------------------------------------------------
# PayPal returns the approved order ID as `token`. Only orders created by this app can be captured.
init_database()
if st.query_params.get("paypal_cancel") == "1":
    clear_pending_order(st.session_state)
    st.session_state["paypal_checkout_notice"] = ("warning", "You cancelled the PayPal Sandbox checkout. No payment was captured.")
    st.query_params.clear()
elif st.query_params.get("token"):
    returned_order_id = str(st.query_params.get("token"))
    try:
        with st.spinner("Confirming the approved payment with PayPal Sandbox..."):
            capture_result = capture_order(returned_order_id)
        # A confirmed capture must not leave a stale "awaiting approval" link in the UI.
        clear_pending_order(st.session_state, returned_order_id)
        st.session_state["paypal_checkout_notice"] = (
            "success",
            f"Sandbox payment captured: {capture_result['currency']} {capture_result['amount']} · Capture ID {capture_result['capture_id']}",
        )
    except PayPalIntegrationError as exc:
        st.session_state["paypal_checkout_notice"] = ("error", str(exc))
    finally:
        st.query_params.clear()  # a browser refresh cannot repeat the flow

for key in ("paypal_checkout_notice", "refund_notice"):
    notice = st.session_state.pop(key, None)
    if notice:
        {"success": st.success, "warning": st.warning}.get(notice[0], st.error)(notice[1])

# --- Sidebar: filters + security posture --------------------------------------
base_transactions = load_transactions()
all_findings = load_findings(base_transactions)

st.sidebar.header("Filters (optional)")
st.sidebar.caption("The defaults already work. Change these only to look at a different period or category.")
today = pd.Timestamp.today().date()
date_range = st.sidebar.date_input(
    "Transaction date range",
    value=(today - timedelta(days=29), today),
    min_value=base_transactions["Date"].min().date(),
    max_value=today,
)
selected_category = st.sidebar.selectbox("Category", ["All categories"] + sorted(base_transactions["Category"].unique().tolist()))
with st.sidebar.expander("Advanced settings", expanded=False):
    contamination_pct = st.slider(
        "Unusual-transaction sensitivity (%)", 1, 10, 3,
        help="Approximate share of transactions the machine-learning view marks as unusual. A flag is not proof of fraud.",
    )
with st.sidebar.expander("What do these words mean?", expanded=False):
    st.markdown(
        "- **Sandbox**: PayPal's practice mode. No real money moves.\n"
        "- **Capture**: the moment a buyer's approved payment is actually collected.\n"
        "- **Refund**: money returned to the buyer.\n"
        "- **Incident**: a pattern the code found that looks unusual.\n"
        "- **Baseline**: this shop's normal level.\n"
        "- **z-score**: how far from normal. Above about 3 is rare."
    )
with st.sidebar.expander("Connection status", expanded=False):
    st.markdown(f"{':material/check_circle:' if credentials_configured() else ':material/cancel:'} PayPal Sandbox keys: "
                f"{'found' if credentials_configured() else 'not found (checkout is off)'}")
    st.markdown(f"{':material/check_circle:' if gemini_ready() else ':material/info:'} Gemini AI: "
                f"{'on' if gemini_ready() else 'off, using the built-in local assistant'}")
with st.sidebar.expander("Security & privacy", expanded=False):
    st.markdown(
        "- PayPal is **Sandbox only**; live URLs are refused.\n"
        "- Prices come from a **server-side catalogue**.\n"
        "- Captured amounts are **verified** against the order.\n"
        "- Gemini receives **aggregates only**: no emails, order or capture IDs.\n"
        "- AI output is **validated**; it can never move money.\n"
        "- Every refund needs **explicit human approval**"
        + (" and a **PIN**." if approval_pin_required() else " (set `MERCHANT_APPROVAL_PIN` to also require a PIN).")
    )

transactions = add_anomaly_predictions(base_transactions, contamination_pct)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range[0] if isinstance(date_range, (tuple, list)) else date_range
    end_date = start_date

day = transactions["Date"].dt.date
mask = day.between(start_date, end_date)
if selected_category != "All categories":
    mask &= transactions["Category"].eq(selected_category)
filtered = transactions.loc[mask].copy()
if filtered.empty:
    st.warning("No transactions match these filters. Try a wider date range or another category.")
    st.stop()

span = (end_date - start_date).days + 1
prev_end = start_date - timedelta(days=1)
prev_start = prev_end - timedelta(days=span - 1)
prev_mask = day.between(prev_start, prev_end)
if selected_category != "All categories":
    prev_mask &= transactions["Category"].eq(selected_category)
previous = transactions.loc[prev_mask]

scope_findings = findings_in_range(all_findings, start_date, end_date)
if selected_category != "All categories":
    scope_findings = [f for f in scope_findings if f.get("category") in (None, selected_category)]
completed = filtered.loc[filtered["Status"] == "Completed"]
sandbox_payments = list_completed_payments()

# --- Start here: three clear doors into the app --------------------------------------
def go_to(label: str) -> None:
    """Open a tab from a card. A new key makes Streamlit build a fresh tab group with `label` preselected."""
    st.session_state["nav_target"] = label
    st.session_state["nav_count"] = st.session_state.get("nav_count", 0) + 1


st.markdown("### Start here: what do you want to do?")
door_cols = st.columns(3)
doors = [
    (":material/crisis_alert:", "1. See what is wrong", "Find payment failures, refund spikes and odd orders, with the evidence.",
     "Open Problems found", ":material/crisis_alert: Problems found"),
    (":material/payments:", "2. Test payment & refund", "Pay with a PayPal test account, then handle a customer refund request.",
     "Open Payments & refunds", ":material/payments: Payments & refunds"),
    (":material/psychology:", "3. Ask MerchantPulse AI", "Type a question in plain English, e.g. \"Why are payments failing?\".",
     "Open Ask AI", ":material/psychology: Ask AI"),
]
for n, (col, (icon, title, text, button, tab_label)) in enumerate(zip(door_cols, doors)):
    with col.container(border=True):
        st.markdown(f"{icon} **{title}**")
        st.caption(text)
        st.button(button, key=f"door_{n}", width="stretch", type="primary", on_click=go_to, args=(tab_label,))

TAB_DASH = ":material/monitoring: Dashboard"
TAB_PROBLEMS = ":material/crisis_alert: Problems found"
TAB_PAY = ":material/payments: Payments & refunds"
TAB_AI = ":material/psychology: Ask AI"
tab_overview, tab_findings, tab_paypal, tab_ai = st.tabs(
    [TAB_DASH, TAB_PROBLEMS, TAB_PAY, TAB_AI],
    default=st.session_state.get("nav_target"), key=f"main_tab_{st.session_state.get('nav_count', 0)}",
)


def page_guide(what: str, todo: str) -> None:
    """A two-line box at the top of each tab: what this page is, and what to do on it."""
    with st.container(border=True):
        st.markdown(f":material/info: **What this page is:** {what}")
        st.markdown(f":material/ads_click: **What to do:** {todo}")


# --- Dashboard header (numbers + briefing) lives inside the Dashboard tab, so the tabs sit under the Start-here cards ---
with tab_overview:
    # --- KPI row -------------------------------------------------------------------
    now_k, prev_k = kpis(filtered), kpis(previous)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Completed volume", money(now_k["volume"]), pct_change(now_k["volume"], prev_k["volume"]) if len(previous) else None,
              border=True, help=f"Exact: ${now_k['volume']:,.2f}. Change is vs the previous {span} days.")
    c2.metric("Success rate", f"{now_k['success']:.1f}%",
              f"{now_k['success'] - prev_k['success']:+.1f} pts" if len(previous) else None, border=True,
              help="Completed ÷ (Completed + Failed). Pending and refunded are excluded.")
    c3.metric("Refunds", money(now_k["refunds"]), pct_change(now_k["refunds"], prev_k["refunds"]) if len(previous) else None,
              delta_color="inverse", border=True, help=f"Exact: ${now_k['refunds']:,.2f}. Lower is better.")
    c4.metric("Transactions", f"{int(now_k['count']):,}", pct_change(now_k["count"], prev_k["count"]) if len(previous) else None, border=True)
    n_critical = sum(1 for f in scope_findings if f["severity"] == "critical")
    c5.metric("Incidents detected", f"{len(scope_findings)}", f"{n_critical} critical", delta_color="off", delta_arrow="off",
              border=True, help="Statistical findings in this selection. See the Problems found tab.")

    # --- Guided briefing: the whole story on the first screen ---------------------------
    impact_now = summarize_impact(scope_findings)
    n_captures = len(sandbox_payments)
    n_refunds = sum(1 for p in sandbox_payments if p.get("refund_status"))
    top_finding = scope_findings[0] if scope_findings else None
    with st.container(border=True):
        brief_col, steps_col = st.columns([1.3, 2.7], vertical_alignment="center", gap="large")
        with brief_col:
            st.markdown("**Today's briefing**")
            if top_finding:
                badge = ":red-badge[:material/error: Critical]" if top_finding["severity"] == "critical" else ":orange-badge[:material/warning: Warning]"
                st.markdown(f"{badge} &nbsp; {top_finding['title']}")
                st.caption(top_finding["summary"])
            else:
                st.markdown(":green-badge[:material/check_circle: All clear]")
                st.caption("No unusual patterns were detected in this selection.")
        with steps_col:
            step_cols = st.columns(4)
            steps = [
                (":material/radar:", "1 · Detect", f"{len(scope_findings)} incident(s) found by code, with evidence"),
                (":material/auto_awesome:", "2 · Explain", "Gemini explains the evidence in plain language"),
                (":material/support_agent:", "3 · Triage", f"AI recommends a refund decision ({n_captures} Sandbox payment(s) captured)"),
                (":material/verified:", "4 · Approve", f"A person approves; the app enforces policy ({n_refunds} refund(s) so far)"),
            ]
            for col, (icon, title, text) in zip(step_cols, steps):
                col.markdown(f"{icon} **{title}**")
                col.caption(text)
        st.caption("New here? Use the three **Start here** cards above, or click a tab below. Money moves only after a person approves.")


# --- Overview -------------------------------------------------------------------
with tab_overview:
    page_guide("a summary of this shop's payments (demo data) with incidents marked on the chart.",
               "Look for the orange bands on the chart, then open **Problems found** for the evidence.")
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Daily payment volume")
        if completed.empty:
            st.info("There are no completed payments in this selection yet.")
        else:
            daily = (completed.assign(Day=completed["Date"].dt.normalize()).groupby("Day", as_index=False)["Amount (USD)"].sum().sort_values("Day"))
            fig = px.line(daily, x="Day", y="Amount (USD)", markers=True, labels={"Amount (USD)": "Completed payments (USD)"})
            fig.update_traces(line=dict(color=BLUE, width=2), marker=dict(size=7, color=BLUE, line=dict(width=2, color=SURFACE)),
                              hovertemplate="%{x|%d %b %Y}<br>$%{y:,.2f}<extra></extra>")
            row = 0
            for f in scope_findings:
                if f["type"] == "offhours_high_value":
                    continue
                label = "Card failures" if f["type"] == "failure_spike" else f"{f['category']} refunds"
                x0 = pd.Timestamp(f["start_date"]) - pd.Timedelta(hours=12)
                fig.add_vrect(x0=x0, x1=pd.Timestamp(f["end_date"]) + pd.Timedelta(hours=12), fillcolor=ORANGE, opacity=0.14, line_width=0)
                # Stagger labels on separate rows so neighbouring incidents never overlap.
                fig.add_annotation(x=x0, xref="x", y=1, yref="paper", text=label, showarrow=False, xanchor="left",
                                   yanchor="bottom", yshift=16 * row, font=dict(size=11, color=INK_SOFT))
                row += 1
            show_chart(style_chart(fig, 330).update_layout(margin=dict(l=8, r=8, t=12 + 18 * max(row, 1), b=8)))
            if scope_findings:
                st.caption("Shaded bands mark incidents found by the detectors. Open the Problems found tab for the evidence.")
    with right:
        st.subheader("Transaction status")
        status_counts = filtered["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        fig = px.bar(status_counts, x="Count", y="Status", orientation="h", color="Status", color_discrete_map=STATUS_COLORS, text="Count")
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:,}<extra></extra>", marker_line=dict(width=0))
        order_rows_top_down(fig, status_counts["Status"])
        fig.update_xaxes(title_text=None, showgrid=False, showticklabels=False)
        show_chart(style_chart(fig, 330))

    left2, right2 = st.columns(2)
    for column, name, group_col in ((left2, "Revenue by category", "Category"), (right2, "Revenue by payment method", "Payment Method")):
        with column:
            st.subheader(name)
            if completed.empty:
                st.info("No completed payments to chart for this selection.")
            else:
                grouped = completed.groupby(group_col, as_index=False)["Amount (USD)"].sum().sort_values("Amount (USD)", ascending=False)
                fig = px.bar(grouped, x="Amount (USD)", y=group_col, orientation="h", text="Amount (USD)")
                fig.update_traces(marker_color=BLUE, marker_line=dict(width=0), texttemplate="$%{x:,.0f}", textposition="outside",
                                  cliponaxis=False, hovertemplate="%{y}: $%{x:,.2f}<extra></extra>")
                order_rows_top_down(fig, grouped[group_col])
                # Head-room on the right so the outside value labels are never clipped.
                fig.update_xaxes(title_text=None, showgrid=False, showticklabels=False,
                                 range=[0, float(grouped["Amount (USD)"].max()) * 1.3])
                show_chart(style_chart(fig, 260))

    st.subheader("Transaction explorer")
    statuses = sorted(filtered["Status"].unique().tolist())
    status_filter = st.multiselect("Show statuses", options=statuses, default=statuses)
    table = filtered.loc[filtered["Status"].isin(status_filter), ["Transaction ID", "Date", "Amount (USD)", "Status", "Category", "Payment Method", "ML Anomaly"]].copy()
    table["Date"] = table["Date"].dt.strftime("%d %b %Y, %H:%M")
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "Amount (USD)": st.column_config.NumberColumn(format="$%.2f"),
        "ML Anomaly": st.column_config.CheckboxColumn("Flagged by model"),
    })

    with st.expander("Who this is for and why it matters", icon=":material/storefront:"):
        st.markdown(
            "**Who it is for.** Small online merchants who accept digital payments but lack dedicated payment-operations teams "
            "to investigate payment failures, refund patterns and customer complaints.\n\n"
            "**Why it matters (published context).**\n"
            "- US merchandise returns reached **USD 743 billion in 2023, 14.5% of sales**, with an **online return rate of 17.6%** "
            "(NRF and Appriss Retail, [as reported by Shop! Association](https://shopassociation.org/many-unhappy-returns-743b-in-merchandise-sent-back-in-2023/)).\n"
            "- In a 2023 survey of 200 subscription-business executives, **four in five failed payments were not the customer's fault** "
            "([PYMNTS and Flex Pay](https://www.pymnts.com/subscriptions/2023/80percent-of-failed-subscription-payments-not-the-customers-fault/)). "
            "That survey covers subscription businesses, so it is context, not a measurement of this app's users.\n\n"
            "**What MerchantPulse changes.** Instead of scanning a table, a merchant gets the incident, the evidence behind it, "
            "a plain-language explanation and a controlled action, with a person approving anything that moves money.\n\n"
            f"**In this demo.** On synthetic data the detectors surface {impact_now['incident_count']} incident(s). "
            f"They associate USD {impact_now['excess_failed_value_usd']:,.0f} with failures above baseline and "
            f"USD {impact_now['excess_refund_value_usd']:,.0f} with refunds above baseline; flagged transactions "
            f"total USD {impact_now['value_under_review_usd']:,.0f} under review. These are benchmark-style measurements, "
            "not realised loss, recovered revenue, or real-merchant outcomes.\n\n"
            "**Not built yet.** Webhook-based capture, PayPal reporting data in place of the synthetic sample, partial refunds and merchant sign-in."
        )

# --- Findings ---------------------------------------------------------------------
with tab_findings:
    page_guide("problems found automatically by code, with the transactions behind each one.",
               "Open **Evidence** on a card, then press **Explain with Gemini** for a plain-language explanation.")
    st.subheader("What needs your attention")
    st.caption(
        "Found by code, explained by AI. Each detector compares a short window with this merchant's own baseline "
        "(a one-sided proportion z-test). Gemini never calculates these numbers; it only explains them."
    )
    impact = summarize_impact(scope_findings)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Incidents", f"{impact['incident_count']}", border=True)
    m2.metric("Excess failed-payment value", f"${impact['excess_failed_value_usd']:,.0f}", border=True,
              help="Value of failed payments above the expected baseline. Not confirmed lost revenue.")
    m3.metric("Excess refund value", f"${impact['excess_refund_value_usd']:,.0f}", border=True,
              help="Refund value above the expected baseline. Not proof of avoidable refunds.")
    m4.metric("Transactions under review", f"${impact['value_under_review_usd']:,.0f}", border=True,
              help="Value requiring investigation, including flagged overnight orders. Not confirmed fraud or loss.")

    failed_daily = (filtered.loc[filtered["Status"] == "Failed"].assign(Day=lambda d: d["Date"].dt.normalize())
                    .groupby(["Day", "Payment Method"], as_index=False).size())
    if not failed_daily.empty:
        st.markdown("**Failed payments per day, by payment method**")
        fig = px.bar(failed_daily, x="Day", y="size", color="Payment Method", color_discrete_map=METHOD_COLORS,
                     category_orders={"Payment Method": list(METHOD_COLORS)}, labels={"size": "Failed payments"})
        fig.update_traces(marker_line=dict(width=2, color=SURFACE), hovertemplate="%{x|%d %b}: %{y} failed<extra></extra>")
        fig.update_layout(barmode="stack")
        show_chart(style_chart(fig, 280, legend=True))

    if not scope_findings:
        st.success("No unusual patterns were detected in this selection.")
    explanations = st.session_state.setdefault("finding_explanations", {})
    for idx, f in enumerate(scope_findings):
        with st.container(border=True):
            badge = ":red-badge[:material/error: Critical]" if f["severity"] == "critical" else ":orange-badge[:material/warning: Warning]"
            st.markdown(f"{badge} &nbsp; **{f['title']}**")
            st.write(f["summary"])
            a, b, c = st.columns(3)
            if f["type"] == "failure_spike":
                a.metric("Failure rate", f"{f['failure_rate_pct']:.0f}%", f"{f['failure_rate_pct'] - f['baseline_rate_pct']:+.0f} pts vs baseline", delta_color="inverse")
                b.metric("Excess failed value", f"${f['excess_failed_value_usd']:,.0f}")
                c.metric("How unusual (z-score)", f"{f['z_score']:.1f}")
            elif f["type"] == "refund_burst":
                a.metric("Refund rate", f"{f['refund_rate_pct']:.0f}%", f"{f['refund_rate_pct'] - f['baseline_rate_pct']:+.0f} pts vs baseline", delta_color="inverse")
                b.metric("Excess refund value", f"${f['excess_refund_value_usd']:,.0f}")
                c.metric("How unusual (z-score)", f"{f['z_score']:.1f}")
            else:
                a.metric("Orders flagged", f"{f['flagged_count']}")
                b.metric("Total value", f"${f['flagged_value_usd']:,.0f}")
                c.metric("Amount limit", f"≥ ${f['amount_threshold_usd']:,.0f}")
            with st.expander("Evidence: the transactions behind this finding"):
                rows = transactions.loc[transactions["Transaction ID"].isin(f["transaction_ids"]), ["Transaction ID", "Date", "Amount (USD)", "Status", "Category", "Payment Method"]].copy()
                rows["Date"] = rows["Date"].dt.strftime("%d %b %Y, %H:%M")
                st.dataframe(rows, width="stretch", hide_index=True, column_config={"Amount (USD)": st.column_config.NumberColumn(format="$%.2f")})
            with st.expander("Suggested checks"):
                st.markdown("\n".join(f"- {item}" for item in PLAYBOOK[f["type"]]))
            key = f"{f['type']}|{f['start_date']}|{f.get('payment_method') or f.get('category') or ''}"
            if st.button("Explain with Gemini", key=f"explain_{idx}", icon=":material/auto_awesome:"):
                question = f"Explain this detected incident in plain language and recommend what the merchant should check and do next: {f['title']}."
                try:
                    if gemini_ready() and ai_allowed():
                        with st.spinner("Asking Gemini to explain the evidence..."):
                            explanations[key] = answer_merchant_question_ai(question, filtered, sandbox_payments, findings=[f])
                    else:
                        explanations[key] = "Gemini is not available right now. Suggested checks:\n\n" + "\n".join(f"- {i}" for i in PLAYBOOK[f["type"]])
                except MerchantAIError as exc:
                    explanations[key] = f"{exc}\n\nSuggested checks:\n\n" + "\n".join(f"- {i}" for i in PLAYBOOK[f["type"]])
            if key in explanations:
                st.markdown(str(explanations[key]).replace("$", "USD "))

    with st.expander("ML anomaly detection (Isolation Forest)"):
        st.write("A second, model-based view: transactions whose amount, timing, category or payment method differ from the rest. These are review flags, not fraud verdicts.")
        flagged = filtered.loc[filtered["ML Anomaly"]].sort_values("Anomaly Score", ascending=False)
        x1, x2, x3 = st.columns(3)
        x1.metric("Flagged transactions", f"{len(flagged):,}")
        x2.metric("Flagged share", f"{len(flagged) / len(filtered) * 100:.1f}%")
        x3.metric("Highest flagged amount", f"${flagged['Amount (USD)'].max():,.2f}" if not flagged.empty else "—")
        if flagged.empty:
            st.info("No transactions were flagged for the current filters.")
        else:
            show = flagged[["Transaction ID", "Date", "Amount (USD)", "Status", "Category", "Payment Method", "Anomaly Score"]].head(15).copy()
            show["Date"] = show["Date"].dt.strftime("%d %b %Y, %H:%M")
            st.dataframe(show, width="stretch", hide_index=True, column_config={
                "Amount (USD)": st.column_config.NumberColumn(format="$%.2f"),
                "Anomaly Score": st.column_config.NumberColumn(format="%.4f"),
            })

# --- PayPal & refunds -----------------------------------------------------------------
with tab_paypal:
    page_guide("a practice PayPal checkout, a list of recorded payments, and an assistant for customer refund requests.",
               "Work top to bottom: **Step 1** take a test payment, **Step 2** see it recorded, **Step 3** refund it.")
    st.subheader("Step 1 · Take a test payment (PayPal Sandbox)")
    st.caption("Create a small test order, approve it with a **Sandbox Personal buyer account**, and the server captures it. Sandbox only: never use real credentials or money.")
    if not credentials_configured():
        st.warning("PayPal credentials were not found. Check the project-root .env file and restart Streamlit.")
    else:
        app_base_url = os.getenv("APP_BASE_URL", "http://localhost:8501").strip().rstrip("/")
        skus = list(SANDBOX_CATALOG.keys())
        col_pick, col_btn = st.columns([2, 1], vertical_alignment="bottom")
        selected_sku = col_pick.selectbox("Fixed-price demo item", skus, key="paypal_demo_sku",
                                          format_func=lambda s: f"{SANDBOX_CATALOG[s][0]} — ${SANDBOX_CATALOG[s][1]:.2f} USD")
        if col_btn.button("Create PayPal Sandbox order", type="primary", key="create_paypal_order", width="stretch"):
            try:
                with st.spinner("Creating a test order with PayPal..."):
                    st.session_state["paypal_pending_order"] = create_order(
                        selected_sku, return_url=f"{app_base_url}/?paypal_return=1", cancel_url=f"{app_base_url}/?paypal_cancel=1")
            except PayPalIntegrationError as exc:
                st.error(str(exc))
        pending_order = st.session_state.get("paypal_pending_order")
        if pending_order:
            st.info(f"Order {pending_order['order_id']} is awaiting buyer approval: {pending_order['currency']} {pending_order['amount']}.")
            st.link_button("Continue to PayPal Sandbox approval", pending_order["approval_url"], width="stretch")

    st.divider()
    st.subheader("Step 2 · Check the payment was recorded")
    if sandbox_payments:
        refunded_payments = [p for p in sandbox_payments if p.get("refund_status")]
        gross = sum(float(p["amount"]) for p in sandbox_payments)
        refunded_total = sum(float(p["amount"]) for p in refunded_payments)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Test payments", f"{len(sandbox_payments):,}", border=True)
        p2.metric("Captured volume", f"${gross:,.2f}", border=True)
        p3.metric("Refunded", f"${refunded_total:,.2f}", f"{len(refunded_payments)} refund(s)", delta_color="off", delta_arrow="off", border=True)
        p4.metric("Net volume", f"${gross - refunded_total:,.2f}", border=True)
        ledger = pd.DataFrame(
            {
                "Item": [p["description"] for p in sandbox_payments],
                "Amount": [float(p["amount"]) for p in sandbox_payments],
                "Refund": [("Refunded" if p.get("refund_status") == "COMPLETED" else p["refund_status"].title()) if p.get("refund_status") else "—" for p in sandbox_payments],
                "Captured at": [p["captured_at"] for p in sandbox_payments],
                "Buyer (masked)": [mask_email(p.get("payer_email")) for p in sandbox_payments],
                "PayPal order ID": [p["order_id"] for p in sandbox_payments],
                "Capture ID": [p["capture_id"] for p in sandbox_payments],
            }
        )
        st.dataframe(ledger, width="stretch", hide_index=True, column_config={"Amount": st.column_config.NumberColumn(format="$%.2f")})
    else:
        st.info("No payments recorded yet. Do Step 1 above: create a test order, then approve it with your Sandbox buyer account.")

    st.divider()
    st.subheader("Step 3 · Handle a customer refund request")
    st.caption(
        f"A customer complains. Gemini **recommends** what to do. You **approve**. The app **enforces** the policy "
        f"(full refunds only, within {REFUND_WINDOW_DAYS} days, once per payment). Nothing is ever refunded automatically."
    )
    if not approval_pin_required() and not IS_LOCAL_DEPLOYMENT:
        st.warning(
            "Approval PIN is not configured. Before exposing this demo publicly, "
            "set MERCHANT_APPROVAL_PIN in the hosting environment."
        )
    refundable = [p for p in sandbox_payments if not p.get("refund_status")]
    progress_slot = st.empty()
    if not refundable:
        st.info("Nothing to refund yet. Finish Step 1 (take a test payment) first, then come back here.")
    else:
        options = {f"{p['description']} · ${float(p['amount']):.2f} · …{p['order_id'][-6:]}": p for p in refundable}
        payment = options[st.selectbox("Payment the customer is complaining about", list(options))]
        order_id = payment["order_id"]

        examples = {
            "Charged twice": "Hi, I was charged twice for the same order. Please fix this.",
            "Never arrived": "My order never arrived and the tracking has not updated in two weeks.",
            "Changed my mind": "I changed my mind and don't want it anymore.",
        }
        ex_cols = st.columns(len(examples))
        for col, (label, text) in zip(ex_cols, examples.items()):
            col.button(label, key=f"ex_{label}", width="stretch", on_click=lambda t=text: st.session_state.update(refund_complaint=t))
        complaint = st.text_area("Customer complaint", key="refund_complaint", max_chars=MAX_COMPLAINT_LEN, height=90,
                                 placeholder="Paste or type what the customer wrote...")
        if st.button("Ask AI to triage", type="primary", icon=":material/support_agent:"):
            checks = check_refund_policy(get_order(order_id))
            use_ai = gemini_ready() and ai_allowed()
            with st.spinner("Triaging the request..."):
                proposal = triage_refund_request(complaint, payment, use_ai=use_ai)
            st.session_state["refund_proposal"] = {
                "order_id": order_id, "complaint": clean_text(complaint, MAX_COMPLAINT_LEN),
                "proposal": finalize_proposal(proposal, checks), "checks": checks,
            }
            final_proposal = st.session_state["refund_proposal"]["proposal"]
            st.session_state["refund_proposal"]["receipt_id"] = create_decision_receipt(
                order_id,
                evidence_summary=f"Customer report: {clean_text(complaint, MAX_COMPLAINT_LEN)}",
                ai_reasoning=f"{final_proposal.label}: {final_proposal.rationale}",
                safety_status=("Held by Complaint Safety Gate; no Gemini request sent"
                               if final_proposal.source == "safety_gate" else "Passed Complaint Safety Gate"),
                policy_checks=checks,
            )

        state = st.session_state.get("refund_proposal")
        if state and state["order_id"] == order_id and state["complaint"] == clean_text(complaint, MAX_COMPLAINT_LEN):
            proposal, checks = state["proposal"], state["checks"]
            with st.container(border=True):
                source_labels = {
                    "gemini": "Gemini",
                    "local_rules": "Local rules",
                    "safety_gate": "Pre-AI safety gate",
                }
                source = source_labels.get(proposal.source, "Manual review")
                st.markdown(f"**{proposal.label}** &nbsp; :gray-badge[{source}] :gray-badge[Confidence: {proposal.confidence}]")
                st.write(proposal.rationale)
                if proposal.source == "safety_gate":
                    st.error("Complaint held before Gemini. No AI request was sent.")
                    st.info("Review the customer's actual issue manually. Do not approve a refund based on instruction-like text in the complaint.")
                elif proposal.notice:
                    st.caption(proposal.notice)
                if proposal.customer_message:
                    st.markdown("**Draft reply to the customer**")
                    st.markdown(f"> {proposal.customer_message}")
                st.markdown("**Policy checks (enforced by the app)**")
                for passed, message in checks:
                    st.markdown(f"{':material/check_circle:' if passed else ':material/cancel:'} {message}")

                receipt_id = state.get("receipt_id")
                with st.expander("Decision Receipt · structured evidence record", expanded=True):
                    st.caption(f"Receipt {receipt_id} · Order …{order_id[-6:]}")
                    st.markdown(f"**Customer report (unverified)**  \n{clean_text(state['complaint'], MAX_COMPLAINT_LEN)}")
                    st.markdown(f"**AI reasoning**  \n{proposal.label}: {proposal.rationale}")
                    st.markdown(f"**Security**  \n{'Held before AI' if proposal.source == 'safety_gate' else 'Complaint Safety Gate passed'}")
                    st.markdown("**Policy**  \n" + "  \n".join(
                        f"{'Passed' if passed else 'Failed'}: {message}" for passed, message in checks
                    ))
                    st.markdown("**Human authority**  \nPending explicit approval")
                    st.markdown("**Execution**  \nNot executed")

                allowed = refund_approval_allowed(proposal, checks)
                if allowed:
                    st.markdown("**Human approval**")
                    reason = st.text_input(
                        "Refund reason (shown to the buyer)",
                        key=f"reason_{order_id}",
                        value=f"Refund approved: {clean_text(state['complaint'], 120)}",
                        max_chars=200,
                    )
                    confirm = st.checkbox(
                        f"I approve refunding {payment['currency']} {payment['amount']} to the buyer's Sandbox account",
                        key=f"confirm_{order_id}",
                    )
                    pin = st.text_input("Approval PIN", type="password", key=f"pin_{order_id}") if approval_pin_required() else ""
                    approve_col, dismiss_col = st.columns(2)
                    if approve_col.button(
                        "Approve & issue Sandbox refund",
                        type="primary",
                        disabled=not confirm,
                        width="stretch",
                        icon=":material/verified:",
                    ):
                        pin_events = st.session_state.setdefault("pin_events", [])
                        if approval_pin_required() and not allow_event(pin_events, max_events=5, window_seconds=300):
                            st.error("Too many approval attempts. Wait a few minutes and try again.")
                        elif not verify_approval_pin(pin):
                            st.error("Incorrect approval PIN.")
                        else:
                            try:
                                update_decision_receipt(
                                    receipt_id,
                                    human_authority="Explicit approval provided in dashboard",
                                    execution_status="Submitted to PayPal Sandbox; awaiting confirmation",
                                )
                                result = refund_capture(
                                    order_id,
                                    reason=reason,
                                    approved_by="merchant (dashboard)",
                                    proposed_by="ai" if proposal.source == "gemini" else "manual",
                                    ai_summary=f"{proposal.label}: {proposal.rationale}",
                                )
                                try:
                                    update_decision_receipt(
                                        receipt_id,
                                        execution_status=f"PayPal confirmed {result['status']}",
                                        execution_reference=result.get("refund_id") or "Refund ID unavailable",
                                    )
                                except Exception:  # the refund already happened; a receipt write must not hide that
                                    pass
                                st.session_state["refund_notice"] = (
                                    "success",
                                    f"Sandbox refund {result['status'].lower()}: "
                                    f"{result['currency']} {result['amount']} · Refund ID {result['refund_id']}",
                                )
                                st.session_state.pop("refund_proposal", None)
                                st.rerun()
                            except PayPalIntegrationError as exc:
                                update_decision_receipt(
                                    receipt_id,
                                    human_authority="Explicit approval provided in dashboard",
                                    execution_status="PayPal outcome not confirmed; check Sandbox ledger",
                                )
                                st.error(str(exc))
                    if dismiss_col.button("Dismiss", width="stretch"):
                        st.session_state.pop("refund_proposal", None)
                        st.rerun()
                else:
                    if proposal.action != "refund_full":
                        st.info("Refund approval is unavailable: this proposal is not an eligible full-refund recommendation.")
                    elif not checks or not all(passed for passed, _ in checks):
                        st.info("Refund approval is unavailable until every policy check passes.")
                    if st.button("Dismiss review", key=f"dismiss_review_{order_id}", width="stretch"):
                        st.session_state.pop("refund_proposal", None)
                        st.rerun()


    # Progress line is filled last, so it reflects the click that just happened.
    _names = ["Pick the payment", "Write the complaint", "Ask AI to triage", "Approve the refund"]
    if not refundable:
        _now = 0
    elif st.session_state.get("refund_proposal"):
        _now = 3
    elif str(st.session_state.get("refund_complaint", "")).strip():
        _now = 2
    else:
        _now = 1
    progress_slot.markdown("  →  ".join(f"**{'①②③④'[i]} {n}**" if i == _now else f"{'①②③④'[i]} {n}" for i, n in enumerate(_names)))

    with st.expander("Recent Decision Receipts"):
        receipts = list_decision_receipts(limit=10)
        if not receipts:
            st.caption("No refund decisions have been triaged yet.")
        for receipt in receipts:
            with st.container(border=True):
                st.caption(f"{receipt['created_at']} · Receipt {receipt['receipt_id']} · Order …{receipt['order_id'][-6:]}")
                st.markdown(f"**Evidence:** {receipt['evidence_summary']}")
                st.markdown(f"**AI reasoning:** {receipt['ai_reasoning']}")
                st.markdown(f"**Security:** {receipt['safety_status']}")
                st.markdown("**Policy:** " + "; ".join(
                    f"{'passed' if passed else 'failed'} — {message}"
                    for passed, message in receipt['policy_checks']
                ))
                st.markdown(f"**Human authority:** {receipt['human_authority']}")
                execution = receipt['execution_status']
                if receipt.get("execution_reference"):
                    execution += f" · Refund ID {receipt['execution_reference']}"
                st.markdown(f"**Execution:** {execution}")

# --- Ask MerchantPulse ------------------------------------------------------------------
with tab_ai:
    page_guide("a question box where you ask about your payments in plain English.",
               "Click an example question, or type your own, then press **Ask MerchantPulse**.")
    st.subheader("Ask MerchantPulse AI")
    st.write("Ask about revenue, failed payments, refunds, incidents or your PayPal Sandbox captures. Gemini explains figures the app calculated; it never recalculates them.")
    if gemini_ready():
        st.caption(f"Gemini is enabled ({get_gemini_model()}). It receives aggregate summaries only: no buyer emails, PayPal order IDs, capture IDs or raw rows.")
    elif gemini_key_configured() and not gemini_sdk_available():
        st.warning("A Gemini API key is configured, but the OpenAI-compatible Python SDK is missing. Run `python -m pip install openai`. Local analysis remains available.")
    else:
        st.caption("Local analytical mode is active. Add GEMINI_API_KEY to the project-root .env file to enable Gemini.")

    samples = ["How can I improve sales?", "What needs attention first?", "Why are payments failing?", "Summarise my PayPal tests"]
    s_cols = st.columns(len(samples))
    for col, text in zip(s_cols, samples):
        col.button(text, key=f"sample_{text}", width="stretch", on_click=lambda t=text: st.session_state.update(merchant_question=t))
    assistant_scope = (str(start_date), str(end_date), selected_category, contamination_pct)
    with st.form("merchant_insights_form", clear_on_submit=False):
        merchant_question = st.text_area("Ask MerchantPulse", key="merchant_question", max_chars=600, height=100,
                                         placeholder="Ask about revenue, failed payments, refunds, incidents, or ways to improve...")
        ask_submitted = st.form_submit_button("Ask MerchantPulse", type="primary")

    if ask_submitted and not merchant_question.strip():
        st.warning("Please enter a question before submitting.")
    elif ask_submitted:
        notice_text = None
        try:
            if gemini_ready() and ai_allowed():
                with st.spinner("Analysing the selected data with MerchantPulse AI..."):
                    merchant_answer = answer_merchant_question_ai(merchant_question, filtered, paypal_payments=sandbox_payments, findings=scope_findings)
                response_mode = f"Gemini generative AI · {get_gemini_model()}"
            else:
                merchant_answer = answer_merchant_question(merchant_question, filtered)
                response_mode = "Local analytical assistant"
                if gemini_ready():
                    notice_text = f"Session limit of {AI_CALLS_PER_10_MIN} Gemini questions per 10 minutes reached, so local analysis answered."
        except MerchantAIError as exc:
            merchant_answer = answer_merchant_question(merchant_question, filtered)
            response_mode, notice_text = "Local analytical fallback", f"Generative AI was unavailable, so the local assistant answered. Details: {exc}"
        except (TypeError, ValueError, KeyError) as exc:
            merchant_answer = "The selected data could not be analysed. Check the dashboard filters and try again."
            response_mode, notice_text = "Analysis error", f"{type(exc).__name__}: {exc}"
        st.session_state["merchant_insights_last_result"] = {
            "scope": assistant_scope, "question": merchant_question.strip(), "answer": merchant_answer,
            "mode": response_mode, "notice": notice_text,
        }

    last_insight = st.session_state.get("merchant_insights_last_result")
    if last_insight and last_insight.get("scope") == assistant_scope and str(last_insight.get("question", "")).strip():
        st.markdown(f"**You asked:** {last_insight['question']}")
        with st.container(border=True):
            st.caption(f"RESPONSE ENGINE: {last_insight.get('mode', 'Local analytical assistant')}")
            st.divider()
            st.markdown(str(last_insight["answer"]).replace("$", "USD "))
        if last_insight.get("notice"):
            st.warning(last_insight["notice"])
    elif last_insight:
        st.caption("Your previous answer used different filters. Submit the question again to analyse the current selection.")

st.caption("MerchantPulse AI • PayPal AI Hackathon prototype • Synthetic analytics sample + separately verified PayPal Sandbox captures")
