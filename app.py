# ============================================================
# Beijing Air Quality Dashboard — app.py
# Run: python -m streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beijing Air Quality",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
SC  = {"Dongsi":"#E74C3C","Tiantan":"#E67E22","Shunyi":"#2ECC71","Huairou":"#3498DB"}
TC  = {"Urban":"#E74C3C","Suburban":"#3498DB"}
SO  = ["Spring","Summer","Autumn","Winter"]
FC  = ["PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","WSPM","month","hour"]
sns.set_theme(style="whitegrid", font_scale=1.05)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    padding: 32px; border-radius: 14px; color: white; margin-bottom: 20px;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 6px; }
.hero p  { font-size: 0.9rem; opacity: 0.75; margin: 0; }
.card {
    background: #f9fafb; border-left: 4px solid #1D9E75;
    border-radius: 10px; padding: 16px 18px; margin-bottom: 10px;
}
.card h3 { color: #1D9E75; font-size: 0.95rem; margin: 0 0 4px; }
.card p  { color: #6b7280; font-size: 0.83rem; margin: 0; }
.insight {
    background: #f0fdf4; border-left: 4px solid #1D9E75;
    padding: 10px 14px; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #374151; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper: add engineered columns ────────────────────────────────────────────
def prep(df):
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df[["year","month","day","hour"]])
    if "season" not in df.columns:
        sm = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",
              5:"Spring",6:"Summer",7:"Summer",8:"Summer",
              9:"Autumn",10:"Autumn",11:"Autumn"}
        df["season"] = pd.Categorical(
            df["month"].map(sm), categories=SO, ordered=True)
    if "station_type" not in df.columns:
        df["station_type"] = df["station"].map(
            {"Dongsi":"Urban","Tiantan":"Urban",
             "Shunyi":"Suburban","Huairou":"Suburban"})
    if "day_type" not in df.columns:
        df["day_type"] = df["date"].dt.dayofweek.apply(
            lambda d: "Weekend" if d >= 5 else "Weekday")
    if "AQI_Level" not in df.columns:
        df["AQI_Level"] = pd.cut(
            df["PM2.5"], bins=[0,35,75,115,float("inf")],
            labels=["Good","Moderate","Unhealthy","Hazardous"], right=False)
    return df

# ── Load / cache model ────────────────────────────────────────────────────────
@st.cache_resource
def get_model(n):
    df = st.session_state["df"]
    if all(os.path.exists(p) for p in
           ["rf_pm25_model.pkl","rf_pm25_scaler.pkl","rf_feature_cols.pkl"]):
        rf = joblib.load("rf_pm25_model.pkl")
        sc = joblib.load("rf_pm25_scaler.pkl")
        fc = joblib.load("rf_feature_cols.pkl")
        return rf, sc, fc
    dm = df[FC+["PM2.5"]].dropna()
    X  = dm[FC]; y = dm["PM2.5"]
    Xt, _, yt, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(sc.fit_transform(Xt), yt)
    return rf, sc, FC

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Beijing Air Quality")
    st.caption("CMP7005 PRAC1 · Student 20318851")
    st.divider()

    page = st.radio("", [
        "🏠  Home",
        "📊  Dataset Explorer",
        "📈  Visualisations",
        "🔍  Urban vs Suburban",
        "🤖  Model & Predictor"
    ], label_visibility="collapsed")

    st.divider()
    up = st.file_uploader("Upload cleaned CSV", type=["csv"])
    if up:
        df_raw = prep(pd.read_csv(up))
        df_raw.to_csv("uploaded_data.csv", index=False)
        st.session_state["df"] = df_raw
        st.success(f"{len(df_raw):,} rows loaded ✓")
    elif "df" not in st.session_state and os.path.exists("uploaded_data.csv"):
        st.session_state["df"] = prep(pd.read_csv("uploaded_data.csv"))
        st.caption("Using saved dataset")

    if "df" in st.session_state:
        st.caption(f"{len(st.session_state['df']):,} records · 4 stations")

# =============================================================================
# HOME
# =============================================================================
if page.startswith("🏠"):
    st.markdown("""
    <div class="hero">
      <h1>🌍 Beijing Air Quality Analytics</h1>
      <p>Hourly data · 4 monitoring stations · March 2013 – February 2017</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card"><h3>📊 Dataset Explorer</h3>
        <p>Filter by station, season and year. View AQI distribution,
           summary metrics, descriptive statistics and raw data.</p></div>
        <div class="card"><h3>📈 Visualisations</h3>
        <p>8 interactive charts: distributions, seasonal boxplots,
           violin plots, monthly trend, heatmap, CO, pie, hourly heatmap.</p></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card"><h3>🔍 Urban vs Suburban</h3>
        <p>Side-by-side comparison of Dongsi & Tiantan (urban) vs
           Shunyi & Huairou (suburban) across all pollutants and seasons.</p></div>
        <div class="card"><h3>🤖 Model & Predictor</h3>
        <p>Random Forest performance, feature importance, and a live
           PM2.5 predictor with instant AQI classification.</p></div>
        """, unsafe_allow_html=True)

    st.divider()
    with st.expander("📖 Research Questions"):
        st.markdown("""
        | # | Question |
        |---|----------|
        | RQ1 | How do PM2.5 levels differ between urban and suburban stations? |
        | RQ2 | What seasonal patterns exist across station types? |
        | RQ3 | Are there meaningful weekday vs weekend differences? |
        | RQ4 | Which meteorological variables most influence PM2.5? |
        | RQ5 | How does CO vary spatially — implications for capture materials? |

        **Data:** Zhang et al. (2017). Beijing Multi-Site Air Quality Data.
        UCI ML Repository. https://doi.org/10.24432/C5RK5G
        """)

    st.info("⬅️ Upload your cleaned CSV using the sidebar, then navigate between pages.")

# =============================================================================
# REQUIRE DATA FOR ALL OTHER PAGES
# =============================================================================
elif "df" not in st.session_state:
    st.warning("⬅️ Please upload your cleaned CSV using the sidebar first.")
    st.stop()

else:
    df = st.session_state["df"]

    # =========================================================================
    # DATASET EXPLORER
    # =========================================================================
    if page.startswith("📊"):
        st.title("📊 Dataset Explorer")

        f1,f2,f3,f4 = st.columns(4)
        sel_stn  = f1.multiselect("Station",  sorted(df["station"].unique()),
                                   default=sorted(df["station"].unique()))
        sel_sea  = f2.multiselect("Season",   SO, default=SO)
        sel_type = f3.multiselect("Type", ["Urban","Suburban"],
                                   default=["Urban","Suburban"])
        yr       = f4.slider("Year",int(df["year"].min()),int(df["year"].max()),
                              (int(df["year"].min()),int(df["year"].max())))

        dff = df[df["station"].isin(sel_stn) & df["season"].isin(sel_sea) &
                 df["station_type"].isin(sel_type) & df["year"].between(yr[0],yr[1])]

        st.divider()
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Records",    f"{len(dff):,}")
        m2.metric("Mean PM2.5", f"{dff['PM2.5'].mean():.1f} µg/m³")
        m3.metric("Max PM2.5",  f"{dff['PM2.5'].max():.0f} µg/m³")
        m4.metric("Mean CO",    f"{dff['CO'].mean():.0f} µg/m³")
        m5.metric("Mean Temp",  f"{dff['TEMP'].mean():.1f} °C")
        m6.metric("Mean Wind",  f"{dff['WSPM'].mean():.1f} m/s")

        st.divider()
        ca, cb = st.columns(2)
        with ca:
            st.subheader("AQI Distribution")
            ac  = dff["AQI_Level"].value_counts().reindex(
                ["Good","Moderate","Unhealthy","Hazardous"]).fillna(0)
            pct = (ac/ac.sum()*100).round(1)
            fig,ax = plt.subplots(figsize=(6,4))
            bars = ax.bar(ac.index,ac.values,
                          color=["#2ECC71","#F1C40F","#E67E22","#E74C3C"],
                          edgecolor="white",width=0.6)
            for b,p in zip(bars,pct):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+200,
                        f"{p}%", ha="center", fontsize=10, fontweight="bold")
            ax.set_ylabel("Hours"); ax.set_title("AQI Counts",fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with cb:
            st.subheader("Mean PM2.5 by Station")
            mp = dff.groupby(["station","station_type"])["PM2.5"].mean().round(2).reset_index()
            mp.columns = ["Station","Type","Mean PM2.5 (µg/m³)"]
            st.dataframe(mp.style.background_gradient(
                cmap="YlOrRd",subset=["Mean PM2.5 (µg/m³)"]),
                use_container_width=True)
            st.subheader("Records per Station")
            rc = dff["station"].value_counts().reset_index()
            rc.columns = ["Station","Records"]
            st.dataframe(rc.style.background_gradient(
                cmap="Blues",subset=["Records"]),use_container_width=True)

        st.divider()
        st.subheader("Descriptive Statistics")
        pc = ["PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","WSPM"]
        st.dataframe(dff[pc].describe().T.round(2).style.background_gradient(
            cmap="Blues",subset=["mean","50%"]),use_container_width=True)

        st.divider()
        mv = pd.DataFrame({"Missing Count":dff.isnull().sum(),
                            "Missing %":(dff.isnull().sum()/len(dff)*100).round(2)})
        mv = mv[mv["Missing Count"]>0]
        st.subheader("Missing Values")
        if len(mv):
            st.dataframe(mv.style.background_gradient(
                cmap="Oranges",subset=["Missing %"]),use_container_width=True)
        else:
            st.success("✅ No missing values in filtered dataset")

        st.divider()
        st.subheader("Raw Data Viewer")
        cols_sel = st.multiselect("Columns", list(dff.columns),
            default=["date","station","station_type","season","day_type",
                     "PM2.5","CO","NO2","TEMP","WSPM","AQI_Level"])
        n = st.slider("Rows", 10, 500, 50, step=10)
        st.dataframe(dff[cols_sel].head(n), use_container_width=True)

    # =========================================================================
    # VISUALISATIONS
    # =========================================================================
    elif page.startswith("📈"):
        st.title("📈 Visualisations")
        chart = st.selectbox("Select chart", [
            "1 — PM2.5 Distribution by Station",
            "2 — Seasonal Boxplot (Urban vs Suburban)",
            "3 — Weekday vs Weekend Violin Plot",
            "4 — Monthly PM2.5 Trend",
            "5 — Correlation Heatmap",
            "6 — CO Analysis",
            "7 — Pollutant Composition Pie",
            "8 — Hourly PM2.5 Heatmap",
        ])
        st.divider()

        if chart.startswith("1"):
            bins = st.slider("Bins", 20, 100, 60)
            fig,axes = plt.subplots(2,2,figsize=(13,8)); axes=axes.flatten()
            for i,s in enumerate(["Dongsi","Tiantan","Shunyi","Huairou"]):
                ax = axes[i]; d = df[df["station"]==s]["PM2.5"].dropna()
                ax.hist(d,bins=bins,color=SC[s],alpha=0.6,edgecolor="white",density=True)
                d.plot.kde(ax=ax,color=SC[s],lw=2.5)
                ax.axvline(d.mean(),color="black",ls="--",lw=1.5,label=f"Mean:{d.mean():.1f}")
                ax.axvline(d.median(),color="grey",ls=":",lw=1.5,label=f"Median:{d.median():.1f}")
                ax.set_title(f"{s} ({df[df['station']==s]['station_type'].iloc[0]})",fontweight="bold")
                ax.set_xlabel("PM2.5 (µg/m³)"); ax.legend(fontsize=8)
            fig.suptitle("Figure 1: PM2.5 Distribution by Station",fontsize=13,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">Right-skewed distributions reflect episodic '
                'extreme pollution events. Urban stations show heavier tails — more '
                'frequent severe episodes (Crouse et al., 2016).</div>',unsafe_allow_html=True)

        elif chart.startswith("2"):
            poll = st.selectbox("Pollutant",["PM2.5","PM10","CO","NO2","SO2","O3"])
            fig,ax = plt.subplots(figsize=(12,6))
            sns.boxplot(data=df,x="season",y=poll,hue="station_type",palette=TC,
                        order=SO,ax=ax,flierprops=dict(marker="o",markersize=2,alpha=0.3))
            if poll=="PM2.5":
                ax.axhline(75,color="orange",ls="--",lw=1.5,label="75 µg/m³"); ax.legend()
            ax.set_title(f"Figure 2: {poll} by Season and Station Type",fontweight="bold",fontsize=13)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">Winter consistently highest (coal heating). '
                'Urban stations exceed suburban every season — urban-suburban gradient '
                'confirmed (Chen et al., 2013).</div>',unsafe_allow_html=True)

        elif chart.startswith("3"):
            sf = st.selectbox("Season filter",["All"]+SO)
            ds = df if sf=="All" else df[df["season"]==sf]
            fig,axes = plt.subplots(1,2,figsize=(13,6),sharey=True)
            for i,st_t in enumerate(["Urban","Suburban"]):
                ax=axes[i]; sub=ds[ds["station_type"]==st_t]
                sns.violinplot(data=sub,x="day_type",y="PM2.5",
                               palette={"Weekday":"#E74C3C","Weekend":"#3498DB"},
                               ax=ax,inner="box",cut=0)
                wk=sub[sub["day_type"]=="Weekday"]["PM2.5"].mean()
                we=sub[sub["day_type"]=="Weekend"]["PM2.5"].mean()
                ax.axhline(wk,color="#E74C3C",ls="--",lw=1.5,label=f"Weekday:{wk:.1f}")
                ax.axhline(we,color="#3498DB",ls="--",lw=1.5,label=f"Weekend:{we:.1f}")
                ax.set_title(f"{st_t} ({sf})\nReduction: {((wk-we)/wk*100):.1f}%",fontweight="bold")
                ax.set_xlabel("Day Type"); ax.set_ylabel("PM2.5 (µg/m³)" if i==0 else "")
                ax.legend(fontsize=9)
            fig.suptitle("Figure 3: Weekday vs Weekend PM2.5",fontsize=13,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">Weekend effect more pronounced at urban stations — '
                'reduced weekday traffic and industrial activity '
                '(Qin et al., 2004).</div>',unsafe_allow_html=True)

        elif chart.startswith("4"):
            show_t = st.checkbox("Show 75 µg/m³ threshold",value=True)
            df2 = df.copy(); df2["d2"] = pd.to_datetime(df2[["year","month","day","hour"]])
            monthly = df2.set_index("d2").groupby("station_type")["PM2.5"].resample("ME").mean().reset_index()
            fig,ax = plt.subplots(figsize=(14,5))
            for st_t,grp in monthly.groupby("station_type"):
                ax.plot(grp["d2"],grp["PM2.5"],color=TC[st_t],lw=2.5,label=st_t)
                ax.fill_between(grp["d2"],grp["PM2.5"],alpha=0.1,color=TC[st_t])
            if show_t: ax.axhline(75,color="orange",ls="--",lw=1.5,label="75 µg/m³")
            ax.set_title("Figure 4: Monthly Mean PM2.5 — Urban vs Suburban 2013–2017",
                         fontsize=13,fontweight="bold")
            ax.set_xlabel("Date"); ax.set_ylabel("Mean PM2.5 (µg/m³)"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">Clear annual cycles: Winter peaks (coal heating '
                'Nov–Feb), Summer troughs. Urban–suburban gap widens every winter '
                '(Huang et al., 2014).</div>',unsafe_allow_html=True)

        elif chart.startswith("5"):
            sf2 = st.selectbox("Station filter",["All"]+sorted(df["station"].unique()))
            dc  = df if sf2=="All" else df[df["station"]==sf2]
            nc  = ["PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","WSPM"]
            fig,ax = plt.subplots(figsize=(11,8))
            sns.heatmap(dc[nc].corr(),annot=True,fmt=".2f",cmap="RdBu_r",center=0,
                        vmin=-1,vmax=1,linewidths=0.5,ax=ax,annot_kws={"size":9})
            ax.set_title(f"Figure 5: Correlation Matrix — {sf2}",fontsize=13,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">PM2.5–CO r≈0.74 (shared combustion origin — '
                'Oyegoke et al., 2020). PM2.5–WSPM r≈−0.31 and PM2.5–TEMP r≈−0.42 '
                '(wind and warmth disperse pollutants).</div>',unsafe_allow_html=True)

        elif chart.startswith("6"):
            fig,axes = plt.subplots(1,2,figsize=(13,5))
            co_s = df.groupby(["season","station_type"],observed=True)["CO"].mean().unstack()
            co_s.reindex([s for s in SO if s in co_s.index]).plot(
                kind="bar",ax=axes[0],color=["#E74C3C","#3498DB"],edgecolor="white",width=0.7)
            axes[0].set_title("Seasonal Mean CO by Station Type",fontweight="bold")
            axes[0].tick_params(axis="x",rotation=0); axes[0].legend(title="Type")
            sns.boxplot(data=df,x="station",y="CO",
                        palette=["#E74C3C","#E67E22","#2ECC71","#3498DB"],ax=axes[1],
                        flierprops=dict(marker="o",markersize=2,alpha=0.3))
            axes[1].set_title("CO Distribution by Station",fontweight="bold")
            fig.suptitle("Figure 6: CO Concentrations — Beijing 2013–2017",
                         fontsize=13,fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">Urban Winter CO ~1,500–2,000 µg/m³ — 2–3× higher '
                'than suburban Summer. Directly relevant to Oyegoke et al. (2020) '
                'metallic oxide CO capture research.</div>',unsafe_allow_html=True)

        elif chart.startswith("7"):
            pc2 = ["PM2.5","PM10","SO2","NO2","CO","O3"]
            ml  = df[pc2].mean().round(2)
            fig = plt.figure(figsize=(10,7))
            sc2 = ["#E74C3C","#E67E22","#F1C40F","#2ECC71","#3498DB","#9B59B6"]
            w,t,at = plt.pie(ml.values,explode=[0.12,0.08,0,0,0,0],colors=sc2,
                              autopct="%1.1f%%",pctdistance=0.75,shadow=True,startangle=140)
            plt.axis("equal")
            plt.title("Figure 7: Dominant Pollutants — Beijing 2013–2017",
                      fontsize=14,fontweight="bold",pad=20)
            plt.legend(w,[f"{n} ({v:.1f} µg/m³)" for n,v in zip(pc2,ml.values)],
                       title="Pollutants",loc="center left",
                       bbox_to_anchor=(1,0,0.5,1),fontsize=9)
            plt.setp(at,size=11,weight="bold",color="white")
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">CO dominates by concentration scale. '
                'PM2.5 and PM10 are the principal health-relevant pollutants '
                '(WHO, 2021).</div>',unsafe_allow_html=True)

        elif chart.startswith("8"):
            s_sel = st.selectbox("Station",sorted(df["station"].unique()))
            pivot = df[df["station"]==s_sel].groupby(["hour","month"])["PM2.5"].mean().unstack()
            fig,ax = plt.subplots(figsize=(14,6))
            sns.heatmap(pivot,cmap="YlOrRd",ax=ax,linewidths=0.3,
                        cbar_kws={"label":"Mean PM2.5 (µg/m³)"})
            ax.set_title(f"Figure 8: Hourly PM2.5 Heatmap — {s_sel}",
                         fontsize=13,fontweight="bold")
            ax.set_xlabel("Month"); ax.set_ylabel("Hour of Day (0–23)")
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('<div class="insight">Darkest cells = worst pollution — winter months '
                '(Nov–Feb) during morning (7–9am) and evening (6–9pm) rush '
                'hours.</div>',unsafe_allow_html=True)

    # =========================================================================
    # URBAN VS SUBURBAN
    # =========================================================================
    elif page.startswith("🔍"):
        st.title("🔍 Urban vs Suburban Comparison")
        st.caption("Dongsi & Tiantan (Urban)  ·  Shunyi & Huairou (Suburban)")

        pollutants = ["PM2.5","PM10","SO2","NO2","CO","O3"]
        urban    = df[df["station_type"]=="Urban"]
        suburban = df[df["station_type"]=="Suburban"]

        st.subheader("Mean Pollutant Delta")
        cols = st.columns(6)
        for col,poll in zip(cols,pollutants):
            u = urban[poll].mean(); s = suburban[poll].mean()
            col.metric(poll,f"{u:.1f}",f"{((u-s)/s*100):+.1f}% vs Sub",
                       delta_color="inverse")

        st.divider()
        st.subheader("All Pollutants Side by Side")
        comp = df.groupby("station_type")[pollutants].mean().round(2)
        fig,axes = plt.subplots(2,3,figsize=(15,8)); axes=axes.flatten()
        for i,poll in enumerate(pollutants):
            ax = axes[i]
            vals = [comp.loc["Urban",poll],comp.loc["Suburban",poll]]
            bars = ax.bar(["Urban","Suburban"],vals,
                          color=["#E74C3C","#3498DB"],edgecolor="white",width=0.5)
            for b,v in zip(bars,vals):
                ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,
                        f"{v:.1f}",ha="center",fontsize=10,fontweight="bold")
            ax.set_title(poll,fontweight="bold",fontsize=12); ax.set_ylabel("Mean (µg/m³)")
        fig.suptitle("Mean Pollutant Levels: Urban vs Suburban",fontsize=14,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        st.subheader("Seasonal Comparison")
        p_sel = st.selectbox("Pollutant",pollutants)
        seas = df.groupby(["season","station_type"],observed=True)[p_sel].mean().unstack()
        seas = seas.reindex([s for s in SO if s in seas.index])
        fig,ax = plt.subplots(figsize=(10,5))
        x = np.arange(len(seas)); w=0.35
        b1=ax.bar(x-w/2,seas.get("Urban",[0]*len(seas)),w,label="Urban",
                  color="#E74C3C",edgecolor="white")
        b2=ax.bar(x+w/2,seas.get("Suburban",[0]*len(seas)),w,label="Suburban",
                  color="#3498DB",edgecolor="white")
        for b in list(b1)+list(b2):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,
                    f"{b.get_height():.1f}",ha="center",fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(seas.index)
        ax.set_title(f"Seasonal {p_sel} — Urban vs Suburban",fontweight="bold",fontsize=13)
        ax.set_ylabel(f"Mean {p_sel} (µg/m³)"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        t1,t2,t3 = st.tabs(["Full Stats","Weekday/Weekend","Season Pivot"])
        with t1:
            tab = df.groupby("station_type")[pollutants].agg(
                ["mean","median","std","max"]).round(2)
            st.dataframe(tab.style.background_gradient(cmap="RdYlGn_r"),
                         use_container_width=True)
        with t2:
            wd = df.pivot_table(values="PM2.5",index="day_type",columns="station_type",
                                 aggfunc=["mean","median","std"],observed=True).round(2)
            st.dataframe(wd.style.background_gradient(cmap="RdYlGn_r"),
                         use_container_width=True)
        with t3:
            piv = df.pivot_table(values="PM2.5",index="season",columns="station_type",
                                  aggfunc="mean",observed=True,
                                  margins=True,margins_name="Overall").round(2)
            st.dataframe(piv.style.background_gradient(cmap="YlOrRd"),
                         use_container_width=True)

    # =========================================================================
    # MODEL & PREDICTOR
    # =========================================================================
    elif page.startswith("🤖"):
        st.title("🤖 Model & Predictor — PM2.5 Prediction")

        with st.spinner("Loading model (first run trains automatically)..."):
            rf, sc, fc = get_model(len(df))

        dm = df[fc+["PM2.5"]].dropna()
        X  = dm[fc]; y = dm["PM2.5"]
        _,Xte,_,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        yp  = rf.predict(sc.transform(Xte))
        mae = mean_absolute_error(yte,yp)
        rmse= np.sqrt(mean_squared_error(yte,yp))
        r2  = r2_score(yte,yp)

        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Model","Random Forest")
        mc2.metric("MAE",f"{mae:.2f} µg/m³")
        mc3.metric("RMSE",f"{rmse:.2f} µg/m³")
        mc4.metric("R²",f"{r2:.4f}")

        if r2>=0.85: st.success(f"Excellent — explains {r2*100:.1f}% of PM2.5 variance")
        elif r2>=0.70: st.info(f"Good — explains {r2*100:.1f}% of PM2.5 variance")
        else: st.warning("Moderate performance")

        st.divider()
        c_s,c_f = st.columns(2)
        with c_s:
            st.subheader("Actual vs Predicted")
            si  = np.random.choice(len(yte),size=min(3000,len(yte)),replace=False)
            yA  = np.array(yte); yP = np.array(yp)
            fig,ax = plt.subplots(figsize=(6,5))
            ax.scatter(yA[si],yP[si],alpha=0.25,s=8,color="#3498DB")
            mv=max(yA.max(),yP.max())
            ax.plot([0,mv],[0,mv],"r--",lw=2,label="Perfect (y=x)")
            ax.set_xlabel("Actual PM2.5 (µg/m³)"); ax.set_ylabel("Predicted PM2.5 (µg/m³)")
            ax.set_title(f"Actual vs Predicted — R²={r2:.4f}",fontweight="bold")
            ax.text(0.05,0.90,f"MAE={mae:.2f}\nRMSE={rmse:.2f}",
                    transform=ax.transAxes,fontsize=10,
                    bbox=dict(boxstyle="round",facecolor="lightyellow",alpha=0.8))
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

        with c_f:
            st.subheader("Feature Importance")
            imp=pd.Series(rf.feature_importances_,index=fc).sort_values()
            ci=["#E74C3C" if v>0.15 else "#3498DB" if v>0.08 else "#95A5A6"
                for v in imp.values]
            fig,ax = plt.subplots(figsize=(6,5))
            imp.plot(kind="barh",ax=ax,color=ci,edgecolor="white")
            ax.set_title("Feature Importance (Gini)",fontweight="bold")
            for i,v in enumerate(imp.values):
                ax.text(v+0.001,i,f"{v:.3f}",va="center",fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="insight">PM10 and CO are the strongest predictors — '
            'shared combustion origins with PM2.5 (Huang et al., 2014). CO capture '
            'at combustion sources reduces co-emitted PM2.5 (Oyegoke et al., 2020).'
            '</div>',unsafe_allow_html=True)

        st.divider()
        st.subheader("🔮 Live PM2.5 Predictor")
        st.caption("Enter atmospheric conditions for an instant PM2.5 prediction:")

        with st.form("pred"):
            st.markdown("**Co-pollutants (µg/m³)**")
            r1=st.columns(5)
            pm10v=r1[0].number_input("PM10",0.0,2000.0,80.0,step=5.0)
            so2v =r1[1].number_input("SO2",0.0,500.0,15.0,step=1.0)
            no2v =r1[2].number_input("NO2",0.0,500.0,60.0,step=1.0)
            cov  =r1[3].number_input("CO",0.0,20000.0,1000.0,step=50.0)
            o3v  =r1[4].number_input("O3",0.0,500.0,50.0,step=1.0)
            st.markdown("**Meteorological**")
            r2b=st.columns(4)
            tv=r2b[0].number_input("Temp (°C)",-30.0,45.0,10.0,step=0.5)
            pv=r2b[1].number_input("Pressure (hPa)",900.0,1100.0,1010.0,step=0.5)
            dv=r2b[2].number_input("Dew Point (°C)",-30.0,30.0,5.0,step=0.5)
            wv=r2b[3].number_input("Wind (m/s)",0.0,20.0,2.0,step=0.5)
            st.markdown("**Time**")
            r3b=st.columns(2)
            mv4=r3b[0].slider("Month",1,12,1)
            hv =r3b[1].slider("Hour",0,23,8)
            go =st.form_submit_button("🔮 Predict PM2.5",use_container_width=True)

        if go:
            pred=max(0,rf.predict(sc.transform(
                np.array([[pm10v,so2v,no2v,cov,o3v,tv,pv,dv,wv,mv4,hv]])))[0])
            if pred<=35:   lvl,col,em,h="Good","#2ECC71","🟢","Satisfactory — little or no health risk."
            elif pred<=75: lvl,col,em,h="Moderate","#F1C40F","🟡","Acceptable — sensitive groups limit exertion."
            elif pred<=115:lvl,col,em,h="Unhealthy","#E67E22","🟠","Sensitive groups affected."
            else:          lvl,col,em,h="Hazardous","#E74C3C","🔴","Health alert — avoid outdoor activity."
            p1,p2,p3=st.columns(3)
            p1.metric("Predicted PM2.5",f"{pred:.1f} µg/m³")
            p2.metric("AQI Category",f"{em} {lvl}")
            p3.metric("Health Risk","Low" if lvl=="Good" else "Moderate" if lvl=="Moderate" else "High" if lvl=="Unhealthy" else "Severe")
            st.markdown(
                f"<div style='background:{col}22;border-left:5px solid {col};"
                f"padding:14px;border-radius:10px;margin-top:10px'>"
                f"<strong>{em} {lvl}</strong> — {h}<br>"
                f"<small>Standard: China GB 3095-2012 (MEE China, 2012)</small></div>",
                unsafe_allow_html=True)
