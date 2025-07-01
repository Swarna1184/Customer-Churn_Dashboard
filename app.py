# app.py – Enhanced Customer Churn Dashboard
# ==========================================

import pathlib, joblib, numpy as np, pandas as pd, shap, requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------



# -----------------------------------------------------------------
# 1. Load model & data
# -----------------------------------------------------------------
MODEL_PATH = pathlib.Path("churn_model.pkl")
DATA_PATH  = pathlib.Path("telco_churn_clusters.csv")

@st.cache_data(show_spinner=False)
def load_model(path):
    return joblib.load(path) if path.exists() else None

@st.cache_data(show_spinner=True)
def load_data(path):
    if not path.exists():
        st.error("CSV not found.")
        st.stop()
    return pd.read_csv(path)

clf = load_model(MODEL_PATH)
df  = load_data(DATA_PATH)

# -----------------------------------------------------------------
# 2. Global SHAP explainer & column list
# -----------------------------------------------------------------
if clf is not None:
    explainer     = shap.TreeExplainer(clf.named_steps["model"])
    feature_names = clf.named_steps["prep"].get_feature_names_out()
else:
    explainer, feature_names = None, []

try:
    orig_cols = list(clf.named_steps["prep"].feature_names_in_)
except AttributeError:
    orig_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

# ---- global SHAP importance (top‑10)
if explainer:
    sample_glob = clf.named_steps["prep"].transform(
        df.sample(min(1000, len(df)), random_state=42))
    if hasattr(sample_glob, "toarray"):
        sample_glob = sample_glob.toarray()
    shap_vals_glob = explainer.shap_values(sample_glob, check_additivity=False)
    mean_abs_glob  = np.abs(shap_vals_glob).mean(axis=0)
    shap_df = (pd.DataFrame({"feature": feature_names,
                             "importance": mean_abs_glob})
               .sort_values("importance", ascending=False)
               .head(10))
else:
    shap_df = pd.DataFrame()

# -------------------------------------------------
# 3.  Sidebar filters
# -------------------------------------------------
with st.sidebar:
    st.header("🔎 Filters")
    gender_filter   = st.multiselect("Gender",   df["gender"].unique(),   default=list(df["gender"].unique()))
    contract_filter = st.multiselect("Contract", df["Contract"].unique(), default=list(df["Contract"].unique()))
    cluster_filter  = st.multiselect("Cluster",  sorted(df["Cluster"].unique()),
                                     default=sorted(df["Cluster"].unique()))

mask        = (df["gender"].isin(gender_filter) &
               df["Contract"].isin(contract_filter) &
               df["Cluster"].isin(cluster_filter))
filtered_df = df[mask]

# -------------------------------------------------
# 4.  KPI cards
# -------------------------------------------------
st.title("📊 Customer Churn Dashboard")

if "CustomerID" not in df.columns:
    df.insert(0, "CustomerID", [f"C{i:04d}" for i in range(1, len(df)+1)])

total_customers = len(filtered_df)
churn_rate      = filtered_df["Churn"].mean()
high_risk       = filtered_df[filtered_df["Churn"] == 1].shape[0]
avg_monthly_revenue = 70  # assumption for demo

col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Customers", f"{total_customers}")
col2.metric("⚠️ Churn Rate",      f"{churn_rate*100:.2f}%")
col3.metric("💰 Est. Monthly Loss",
            f"${churn_rate*total_customers*avg_monthly_revenue:,.0f}")

# Global 12‑month revenue‑loss curve
months     = np.arange(1, 13)
lost_cust  = churn_rate * total_customers * months
rev_loss   = lost_cust * avg_monthly_revenue   # used later
st.divider()

# -------------------------------------------------
# 5.  Live Churn Prediction Sandbox
# -------------------------------------------------
# ---------------------------------------------
#  WHAT‑IF SCENARIO  (replacement)
# ---------------------------------------------
with st.expander("🔄 What‑If Scenario Simulator"):
    default_row = df[orig_cols].mode().iloc[0]  # baseline
    with st.form("whatif_form"):
        st.write("Adjust fields to see impact on churn probability.")
        tenure_inp  = st.slider("Tenure (months)", 0, 72, int(default_row["tenure"]))
        monthly_inp = st.slider("Monthly Charges ($)", 10, 120, int(default_row["MonthlyCharges"]))
        contract_inp  = st.selectbox("Contract Type", sorted(df["Contract"].unique()),
                                     index=list(df["Contract"].unique()).index(default_row["Contract"]))
        internet_inp  = st.selectbox("Internet Service", sorted(df["InternetService"].unique()),
                                     index=list(df["InternetService"].unique()).index(default_row["InternetService"]))
        go = st.form_submit_button("Compare")

    if go and clf:
        # Build BEFORE and AFTER rows
        before = default_row.copy()
        after  = before.copy()
        after.update({"tenure": tenure_inp,
                      "MonthlyCharges": monthly_inp,
                      "Contract": contract_inp,
                      "InternetService": internet_inp})
        df_before = pd.DataFrame([before])[orig_cols]
        df_after  = pd.DataFrame([after])[orig_cols]

        prob_before = clf.predict_proba(df_before)[0,1]
        prob_after  = clf.predict_proba(df_after)[0,1]

        colA, colB = st.columns(2)
        colA.metric("Before", f"{prob_before*100:.1f}%")
        colB.metric("After",  f"{prob_after*100:.1f}%")

        diff = (prob_after - prob_before)*100
        st.write(f"**Δ Churn Probability:** `{diff:+.1f}%`")



# ---------------------------------------------
#  CHURN RISK ALERT  (after churn_rate calc)
# ---------------------------------------------
SLACK_WEBHOOK = "https://hooks.slack.com/services/XXXX/YYY/ZZZ"  # replace

def send_slack_alert(msg):
    try:
        requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=5)
    except Exception as e:
        st.warning(f"Slack alert failed: {e}")

# Fire alert if churn > 30 %
ALERT_THRESHOLD = 0.30
if churn_rate > ALERT_THRESHOLD:
    send_slack_alert(f"⚠️  High churn rate detected: {churn_rate*100:.2f}%")


# -------------------------------------------------
# 6.  Cohort heat‑map by tenure
# -------------------------------------------------
with st.expander("📊 Cohort Churn Heat‑Map (Tenure Bands)"):
    bins = pd.cut(df["tenure"], [0,6,12,24,48,72],
                  labels=["0‑6m","6‑12m","1‑2y","2‑4y","4‑6y"])
    cohort = df.assign(TenureBand=bins).groupby("TenureBand")["Churn"].mean()
    fig, ax = plt.subplots(figsize=(6,2.5))
    sns.heatmap(cohort.to_frame().T*100, annot=True, fmt=".1f",
                cmap="YlOrRd", cbar=False, ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# -------------------------------------------------
# 7.  Cluster radar plot
# -------------------------------------------------
with st.expander("🕸️ Cluster Persona Radar Plot"):
    metrics = ["AvgTenure","AvgCharges","ChurnRate"]
    cluster_stats = (df.groupby("Cluster")
                       .agg(AvgTenure=("tenure","mean"),
                            AvgCharges=("MonthlyCharges","mean"),
                            ChurnRate=("Churn","mean"))
                       .reset_index())
    for m in metrics:
        cluster_stats[m] = (cluster_stats[m]-cluster_stats[m].min())/(cluster_stats[m].max()-cluster_stats[m].min())
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))
    fig = plt.figure(figsize=(6,4))
    ax  = plt.subplot(111, polar=True)
    for _, row in cluster_stats.iterrows():
        vals = row[metrics].tolist()+row[metrics].tolist()[:1]
        ax.plot(angles, vals, label=f"Cluster {int(row['Cluster'])}")
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics); ax.set_yticklabels([])
    ax.set_title("Cluster Personas (scaled 0‑1)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))
    st.pyplot(fig)

# -------------------------------------------------
# 8.  Retention Strategy A/B Simulator
# -------------------------------------------------
with st.expander("🧪 Retention Strategy A/B Simulator"):
    sel_cluster = st.selectbox("Target Cluster", sorted(df["Cluster"].unique()))
    red_pct     = st.slider("Expected Churn Reduction (%)", 0, 50, 10)
    base        = df[df["Cluster"]==sel_cluster]
    base_churn  = base["Churn"].mean()
    new_churn   = max(0, base_churn - red_pct/100)
    saved_rev   = (base_churn-new_churn)*len(base)*avg_monthly_revenue
    st.metric("Cluster Churn ↓", f"{base_churn*100:.1f}% → {new_churn*100:.1f}%")
    st.metric("Est. Monthly Revenue Saved", f"${saved_rev:,.0f}")

# -------------------------------------------------
# 9.  Root‑Cause SHAP per Cluster
# -------------------------------------------------
if explainer:
    with st.expander("🔎 Root‑Cause Explorer (SHAP by Cluster)"):
        tgt_cluster = st.selectbox("Select Cluster", sorted(df["Cluster"].unique()), key="rc_cluster")
        rows = filtered_df[filtered_df["Cluster"] == tgt_cluster]
        if len(rows) < 10:
            st.info("Need ≥10 rows for stable SHAP plot.")
        else:
            rows_X = clf.named_steps["prep"].transform(rows)
            if hasattr(rows_X,"toarray"): rows_X = rows_X.toarray()
            shap_vals = explainer.shap_values(rows_X, check_additivity=False)
            fig, ax = plt.subplots(figsize=(6,3))
            shap.summary_plot(shap_vals, pd.DataFrame(rows_X, columns=feature_names),
                              plot_type="bar", max_display=10, show=False)
            st.pyplot(fig)

# -------------------------------------------------
# 10.  Visualizations (contract & cluster)
# -------------------------------------------------
col4, col5 = st.columns((2,1))
with col4:
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Contract", hue="Churn", ax=ax)
    st.pyplot(fig)

with col5:
    st.subheader("Churn by Cluster")
    churn_by_cluster = filtered_df.groupby("Cluster")["Churn"].mean()
    st.bar_chart(churn_by_cluster)

with st.expander("📈 Churn Trend by Tenure"):
    churn_trend = filtered_df.groupby("tenure")["Churn"].mean()
    st.line_chart(churn_trend)

# -------------------------------------------------
# 11.  SHAP Feature Importance (global top‑10)
# -------------------------------------------------
if not shap_df.empty:
    st.subheader("🔍 Top Features Driving Churn (SHAP)")
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(5,3))
    sns.barplot(data=shap_df, x="importance", y="feature",
                palette="viridis", ax=ax)
    ax.set_xlabel("Mean |SHAP|")
    st.pyplot(fig, use_container_width=True)


# ---------------------------------------------
#  CUSTOMER DRILL‑DOWN  (search by ID)
# ---------------------------------------------
with st.expander("🔎 Customer Drill‑Down"):
    cust_id = st.text_input("Enter CustomerID")
    if cust_id:
        row = df[df["CustomerID"].astype(str) == cust_id]
        if row.empty:
            st.error("CustomerID not found.")
        else:
            st.write("### Customer Record")
            st.dataframe(row)

            # Predict & show churn prob for this customer
            raw_row = row[orig_cols]
            prob_single = clf.predict_proba(raw_row)[0, 1]
            st.metric("Predicted Churn Probability", f"{prob_single*100:.1f}%")

            # SHAP waterfall explanation (single instance)
            if explainer:
                X1 = clf.named_steps["prep"].transform(raw_row)
                if hasattr(X1, "toarray"):
                    X1 = X1.toarray()
                shap_val = explainer.shap_values(X1, check_additivity=False)
                st.write("#### SHAP Waterfall (Top 10 features)")
                fig = plt.figure(figsize=(5,3))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_val[0], feature_names=feature_names),
                    max_display=10, show=False)
                st.pyplot(fig, bbox_inches="tight")

# -------------------------------------------------
# 12.  Cluster‑Level Summary table
# -------------------------------------------------
with st.expander("📦 Cluster‑Level Summary"):
    summary = (
        df.groupby("Cluster").agg(ChurnRate=("Churn","mean"),
                                  AvgTenure=("tenure","mean"),
                                  AvgCharges=("MonthlyCharges","mean"),
                                  Count=("Churn","size"))
        .reset_index().round(2)
    )
    summary["ChurnRate"] = (summary["ChurnRate"]*100).astype(str)+"%"
    st.dataframe(summary, use_container_width=True)

# -------------------------------------------------
# 13.  Suggested Retention Strategy
# -------------------------------------------------
strategies = {
    0: "💡 Loyalty bonuses, exclusive offers",
    1: "🎁 Welcome discount bundles",
    2: "💸 Fixed‑rate or bundled savings",
    3: "🎯 Personalized support + discounts",
}
st.subheader("💡 Suggested Retention Strategy")
strategy_df = (
    filtered_df.groupby("Cluster")["Churn"]
    .agg(["mean","count"]).reset_index()
    .rename(columns={"mean":"ChurnRate","count":"Customers"})
)
strategy_df["ChurnRate"] = (strategy_df["ChurnRate"]*100).round(2).astype(str)+"%"
strategy_df["Strategy"]  = strategy_df["Cluster"].map(strategies)
st.dataframe(strategy_df, use_container_width=True)

# -------------------------------------------------
# 14.  Executive Summary CSV download
# -------------------------------------------------
with st.expander("📥 Download Executive CSV Summary"):
    summ_df = pd.DataFrame({
        "TotalCustomers":[total_customers],
        "ChurnRate":[round(churn_rate*100,2)],
        "HighRiskCust":[high_risk],
        "MonthlyRevenueLoss":[round(churn_rate*total_customers*avg_monthly_revenue,0)]
    })
    st.download_button("Download Summary CSV",
                       summ_df.to_csv(index=False).encode("utf-8"),
                       "churn_executive_summary.csv",
                       "text/csv")

# -------------------------------------------------
# 15.  Executive Insights Markdown
# -------------------------------------------------
with st.expander("📑 Executive Summary"):
    st.markdown(f"""
**Key Takeaways**

* **Total Customers:** `{total_customers}`
* **Overall Churn:** `{churn_rate*100:.2f}%`
* **High‑Risk Customers:** `{high_risk}`
* **Projected 12‑mo Revenue Loss:** `${rev_loss[-1]:,.0f}`

Top churn drivers (SHAP):
1. `{shap_df.iloc[0]['feature']}`
2. `{shap_df.iloc[1]['feature']}`
3. `{shap_df.iloc[2]['feature']}`

**Next Steps**

* Focus retention on cluster with highest churn (see radar plot).
* Use A/B simulator to estimate ROI of discount offers.
""")

# -------------------------------------------------
# 16.  Footer
# -------------------------------------------------
st.caption("© 2025 Enhanced Churn Dashboard by Swarna Suganthi S")
