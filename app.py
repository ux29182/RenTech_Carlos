import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import time

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(
    page_title="RenTech Quantum COMMANDER", 
    layout="wide", 
    page_icon="🧠"
)

st.markdown("""
<style>
    .stSelectbox {font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 1.1rem;} 
    div[data-testid="stMetricLabel"] {font-size: 0.8rem; font-weight: bold; color: #aaaaaa;}
    .big-font {font-size: 20px !important; font-weight: bold;}
    .daily-card {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 RenTech Quantum: THE COMMANDER")

# --- 2. BARRA LATERAL ---
st.sidebar.header("🕹️ Centro de Control")

if st.sidebar.button("🔄 Actualizar Sistemas"):
    st.cache_data.clear()

activos_disponibles = {
    "💶 EUR/USD (Euro)": "EURUSD=X",
    "💷 GBP/USD (Libra)": "GBPUSD=X",
    "💴 USD/JPY (Yen)": "USDJPY=X",
    "🍁 USD/CAD (Loonie)": "USDCAD=X",
    "🌎 AUD/USD (Aussie)": "AUDUSD=X",
    "🥝 NZD/USD (Kiwi)": "NZDUSD=X",
    "📀 ORO (Gold)": "GC=F",
    "💻 NASDAQ 100": "^NDX",
    "🇺🇸 S&P 500": "^GSPC",
    "₿ BITCOIN": "BTC-USD"
}

nombre_activo = st.sidebar.selectbox("Activo Principal", list(activos_disponibles.keys()))
ticker = activos_disponibles[nombre_activo]

context_options = {
    "📉 VIX (Miedo)": "^VIX",
    "🏛️ BONOS 10Y (Tasas)": "^TNX",
    "💵 DXY (Dólar)": "DX-Y.NYB",
    "🇺🇸 S&P 500": "^GSPC",
    "🛢️ PETRÓLEO": "CL=F"
}
nombre_contexto = st.sidebar.selectbox("Contexto Macro", list(context_options.keys()))
context_ticker = context_options[nombre_contexto]

st.sidebar.markdown("---")
periodo = st.sidebar.selectbox("Memoria Histórica", ["2y", "5y", "10y"], index=0)

# --- 3. FUNCIONES DE DATOS ---
@st.cache_data
def get_data_pro(symbol, context_symbol, period):
    try:
        t = yf.download(symbol, period=period, interval="1d", progress=False)
        c = yf.download(context_symbol, period=period, interval="1d", progress=False)
        
        if isinstance(t.columns, pd.MultiIndex): t.columns = t.columns.get_level_values(0)
        if isinstance(c.columns, pd.MultiIndex): c.columns = c.columns.get_level_values(0)
        
        if t.empty or c.empty: return pd.DataFrame()
        
        df = pd.DataFrame(index=t.index)
        df["Close"] = t["Close"]
        df["High"] = t["High"]
        df["Low"] = t["Low"]
        df["Context"] = c["Close"]
        df.dropna(inplace=True)
        return df
    except: return pd.DataFrame()

@st.cache_data
def get_all_assets_data(assets_dict, period="1y"):
    df_all = pd.DataFrame()
    for name, sym in assets_dict.items():
        time.sleep(0.05)
        try:
            d = yf.download(sym, period=period, interval="1d", progress=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            if not d.empty:
                df_all[name.split(" ")[1]] = np.log(d["Close"] / d["Close"].shift(1))
        except: pass
    return df_all.dropna()

def run_hmm_robust(data):
    df = data.copy()
    df["Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol"] = (df["High"] - df["Low"]) / df["Close"]
    df["Context_Ret"] = np.log(df["Context"] / df["Context"].shift(1))
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    if len(df) < 50: return df, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["Ret", "Vol", "Context_Ret"]].values)

    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X_scaled)
    hidden_states = model.predict(X_scaled)
    
    real_vol = [np.mean(df[["Ret", "Vol", "Context_Ret"]].values[hidden_states == i, 1]) for i in range(model.n_components)]
    state_map = {old: new for new, old in enumerate(np.argsort(real_vol))}
    df["Regime"] = [state_map[s] for s in hidden_states]
    return df, model

def get_mindset(regime):
    if regime == 0: return "⚔️ AGRESIVO", "Swing / Dejar correr"
    elif regime == 1: return "🛡️ CONSERVADOR", "Scalping / Rango"
    else: return "💣 DEFENSIVO", "Mitad Lote / Breakout"

def analyze_seasonality_summary(df):
    df_s = df.copy()
    df_s["Day"] = df_s.index.day_name()
    df_s["Ret"] = df_s["Ret"] * 100
    stats = df_s.groupby("Day")["Ret"].agg(["mean", lambda x: (x > 0).mean() * 100])
    
    best_day = stats["mean"].idxmax()
    worst_day = stats["mean"].idxmin()
    today_name = pd.Timestamp.now().day_name()
    today_stats = stats.loc[today_name] if today_name in stats.index else None
    
    return best_day, worst_day, today_stats, stats

def plot_interactive(df, ticker, context_ticker):
    color_map = {0: 'rgba(0, 255, 127, 0.2)', 1: 'rgba(255, 215, 0, 0.2)', 2: 'rgba(255, 69, 0, 0.2)'}
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df['block'] = (df['Regime'] != df['Regime'].shift(1)).cumsum()
    blocks = df.groupby(['block', 'Regime']).agg(start=('Close', lambda x: x.index[0]), end=('Close', lambda x: x.index[-1]))
    shapes = []
    for idx, row in blocks.iterrows():
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=row['start'], x1=row['end'] + pd.Timedelta(hours=12), y0=0, y1=1, fillcolor=color_map[idx[1]], opacity=0.5, layer="below", line_width=0))
    fig.update_layout(shapes=shapes)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"{ticker}", line=dict(color='white', width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA 50", line=dict(color='yellow', width=1, dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df['Context'], name=f"{context_ticker}", line=dict(color='cyan', width=1, dash='dot'), opacity=0.7), secondary_y=True)
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.02, x=1))
    return fig

# --- 4. PRE-CÁLCULO GLOBAL ---
with st.spinner('Inicializando COMMANDER AI...'):
    raw_df = get_data_pro(ticker, context_ticker, periodo)
    processed_df = None
    model_hmm = None
    
    if not raw_df.empty and len(raw_df) > 50:
        processed_df, model_hmm = run_hmm_robust(raw_df)

# --- 5. INTERFAZ DE PESTAÑAS ---
tab0, tab1, tab2, tab3 = st.tabs(["🏠 COMMAND CENTER", "🚀 TÁCTICA", "🌐 ESPACIO", "⏳ TIEMPO"])

# === TAB 0: MASTER CONTROLLER ===
with tab0:
    if processed_df is not None:
        res = processed_df
        last_st = res["Regime"].iloc[-1]
        last_px = res["Close"].iloc[-1]
        last_sma = res["SMA50"].iloc[-1]
        
        if last_px > last_sma: trend_emoji, trend_txt = "🐂", "BULL (Alcista)"
        else: trend_emoji, trend_txt = "🐻", "BEAR (Bajista)"
        
        names = {0: "🟢 CALMA", 1: "🟡 TRANSICIÓN", 2: "🔴 CAOS"}
        risk_txt = names[last_st]
        
        best_d, worst_d, today_st, _ = analyze_seasonality_summary(res)
        today_name = pd.Timestamp.now().day_name()
        
        today_verdict = "NEUTRO"
        if today_st is not None:
            if today_st["mean"] > 0 and today_st["<lambda_0>"] > 53: today_verdict = "✅ DÍA DE ORO"
            elif today_st["<lambda_0>"] < 47: today_verdict = "🩸 DÍA TRAMPA"
            else: today_verdict = "💤 DÍA LENTO"

        st.markdown(f"## Resumen Ejecutivo: {nombre_activo}")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("PRECIO ACTUAL", f"{last_px:.4f}", f"{res['Ret'].iloc[-1]:.2%}")
        with col2: st.metric("TENDENCIA MAESTRA", trend_txt, trend_emoji)
        with col3: st.metric("RÉGIMEN DE RIESGO", risk_txt, "HMM AI")
        with col4: st.metric(f"ESTADÍSTICA DE HOY", today_verdict, today_name)

        st.markdown("---")
        
        c_scan, c_space = st.columns(2)
        with c_scan:
            st.subheader("📡 Radar Global (Rápido)")
            context_dict = {k:v for k,v in context_options.items() if v != context_ticker}
            for k, v in context_dict.items():
                try:
                    time.sleep(0.1)
                    d = yf.download(v, period="5d", progress=False)
                    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                    pct = (d["Close"].iloc[-1] / d["Close"].iloc[-2]) - 1
                    color = "inverse" if "^VIX" in v or "DX-Y" in v or "^TNX" in v else "normal"
                    st.metric(k.split(" ")[1], f"{d['Close'].iloc[-1]:.2f}", f"{pct:.2%}", delta_color=color)
                except: pass

        with c_space:
            st.subheader("🧬 Aliados y Enemigos")
            if st.button("Buscar Relaciones"):
                with st.spinner("Analizando..."):
                    df_all = get_all_assets_data(activos_disponibles, period="6mo")
                    if not df_all.empty:
                        corr = df_all.corr()
                        my_name = nombre_activo.split(" ")[1]
                        if my_name in corr.columns:
                            s = corr[my_name].sort_values(ascending=False)
                            st.success(f"🤝 **Aliado:** {s.index[1]} ({s.iloc[1]:.2f})")
                            st.error(f"⚔️ **Enemigo:** {s.index[-1]} ({s.iloc[-1]:.2f})")
    else:
        st.error("Inicializando sistema... Espera un momento.")

# === TAB 1: TÁCTICA (CON ESCÁNER COMPLETO RESTAURADO) ===
with tab1:
    if processed_df is not None:
        last_px = res["Close"].iloc[-1]
        last_sma = res["SMA50"].iloc[-1]
        mindset_title, mindset_desc = get_mindset(last_st)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Brújula", trend_txt)
        with c2: st.metric("Mentalidad", mindset_title, mindset_desc)
        with c3: st.metric("Contexto", f"{res['Context'].iloc[-1]:.2f}")
        with c4: st.metric("Velocímetro", risk_txt)

        st.plotly_chart(plot_interactive(res, ticker, context_ticker), use_container_width=True)
        
        # --- AQUÍ ESTÁ EL ESCÁNER COMPLETO QUE PEDISTE ---
        st.markdown("---")
        st.subheader("🌐 Escáner Global: Probabilidades de Régimen")
        
        scan_results = []
        context_dict = {k:v for k,v in context_options.items() if v != context_ticker}
        scan_bar = st.progress(0)
        
        for i, (c_name, c_code) in enumerate(context_dict.items()):
            time.sleep(0.5) # Pausa para evitar bloqueo de Yahoo
            try:
                d_scan = get_data_pro(ticker, c_code, period)
                if len(d_scan) > 50:
                    _, m_scan = run_hmm_robust(d_scan)
                    if m_scan:
                        r_scan, _ = run_hmm_robust(d_scan)
                        l_reg = r_scan["Regime"].iloc[-1]
                        probs = m_scan.transmat_[l_reg]
                        scan_results.append({
                            "Factor": c_name, 
                            "Régimen": l_reg, 
                            "Calma": probs[0], 
                            "Caos": probs[2]
                        })
            except: pass
            scan_bar.progress((i + 1) / len(context_dict))
        
        scan_bar.empty()
        
        if scan_results:
            df_scan = pd.DataFrame(scan_results)
            regime_map = {0: "🟢 Calma", 1: "🟡 Transición", 2: "🔴 Caos"}
            df_scan["Estado"] = df_scan["Régimen"].map(regime_map)
            
            st.dataframe(
                df_scan[["Factor", "Estado", "Calma", "Caos"]],
                use_container_width=True,
                column_config={
                    "Calma": st.column_config.ProgressColumn("Prob. Calma", format="%.1f%%", min_value=0, max_value=1),
                    "Caos": st.column_config.ProgressColumn("Prob. Caos", format="%.1f%%", min_value=0, max_value=1),
                }, hide_index=True
            )
        else:
            st.warning("No se pudo cargar la tabla detallada (Yahoo Limit). Intenta en 1 minuto.")

# === TAB 2: ESPACIO ===
with tab2:
    st.subheader("🌐 La Red Neuronal del Mercado")
    if st.button("Generar Mapa de Calor Completo"):
        with st.spinner("Procesando..."):
            df_all = get_all_assets_data(activos_disponibles)
            if not df_all.empty:
                corr_matrix = df_all.corr()
                fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)

# === TAB 3: TIEMPO ===
with tab3:
    if processed_df is not None:
        best_d, worst_d, _, stats = analyze_seasonality_summary(res)
        
        st.subheader("📅 Tu Agenda Semanal de Trading")
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        stats = stats.reindex(days_order)
        
        cols = st.columns(5)
        for i, day in enumerate(days_order):
            if day in stats.index:
                row = stats.loc[day]
                ret = row["mean"]
                win = row["<lambda_0>"]
                
                border_color = "#555"
                status = "NEUTRO"
                if ret > 0 and win > 53: 
                    border_color = "#00FF00"
                    status = "✅ ORO"
                elif win < 47: 
                    border_color = "#FF4B4B"
                    status = "🩸 TRAMPA"
                
                with cols[i]:
                    st.markdown(f"""
                    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 8px; border-top: 5px solid {border_color}; text-align: center;">
                        <h4 style="margin:0;">{day[:3]}</h4>
                        <p style="font-size: 12px; color: #888; margin:0;">{status}</p>
                        <hr style="margin: 5px 0; border-color: #333;">
                        <h2 style="margin:0; color: white;">{win:.0f}%</h2>
                        <p style="font-size: 12px;">Win Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: st.success(f"🏆 **Mejor Día:** {best_d}")
        with c2: st.error(f"💀 **Peor Día:** {worst_d}")

        with st.expander("Ver Gráficos Detallados"):
            colors = ['#00FF00' if v > 0 else '#FF4B4B' for v in stats["mean"]]
            fig_day = go.Figure(data=[go.Bar(x=stats.index, y=stats["mean"], marker_color=colors)])
            fig_day.update_layout(template="plotly_dark", title="Rentabilidad Promedio", height=300)
            st.plotly_chart(fig_day, use_container_width=True)