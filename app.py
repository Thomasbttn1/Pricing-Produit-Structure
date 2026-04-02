

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from models import RateCurve, BlackScholes, MonteCarlo
from products import (
    ZeroCouponBond, CouponBond,
    InterestRateSwap, BasisSwap,
    EuropeanOption, BarrierOption,
    CallSpread, PutSpread, Butterfly,
    AutocallProduct, StructuredNote,
)
from portfolio import load_market_data, load_portfolio, price_portfolio, risk_matrix


st.set_page_config(
    page_title="Pricer – Produits Structurés",
    page_icon="📈",
    layout="wide",
)

@st.cache_data(show_spinner="Chargement des données de marché…")
def get_market():
    return load_market_data()


@st.cache_data(show_spinner="Pricing du portefeuille…")
def get_portfolio_df():
    ptfs = load_portfolio()
    df   = price_portfolio(ptfs)
    dm   = risk_matrix(ptfs, "delta")
    dv   = risk_matrix(ptfs, "vega")
    return df, dm, dv, ptfs


st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choisir une page :",
    ["🌍 Marché & Calibration", "🔧 Pricer Unitaire", "💼 Portefeuille"],
)

mkt = get_market()
S   = mkt["spot"]
sig = mkt["sigma"]
q   = mkt["div_yield"]
r   = mkt["r"]
curve: RateCurve = mkt["curve"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Sous-jacent :** AAPL")
st.sidebar.markdown(f"**Spot :** ${S:.2f}")
st.sidebar.markdown(f"**Vol 1Y :** {sig:.1%}")
st.sidebar.markdown(f"**Taux 3M :** {r:.2%}")
st.sidebar.markdown(f"**Div yield :** {q:.3%}")
st.sidebar.markdown(f"*Date de valorisation : 27/02/2026*")


if page == "🌍 Marché & Calibration":
    st.title("🌍 Données de Marché & Calibration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Courbe de taux zéro-coupon (US Treasury)")

        rate_df = pd.read_csv("docs/rate_curve.csv").sort_values("maturity_years")
        fine_mats = np.linspace(0.08, 30, 200)
        fine_rates = [curve.zero_rate(t) for t in fine_mats]

        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(
            x=fine_mats, y=[r * 100 for r in fine_rates],
            mode="lines", name="Nelson-Siegel", line=dict(color="#1f77b4", width=2)
        ))
        fig_rate.add_trace(go.Scatter(
            x=rate_df["maturity_years"], y=rate_df["zero_rate"] * 100,
            mode="markers", name="FRED (marché)", marker=dict(size=8, color="red")
        ))
        fig_rate.update_layout(
            xaxis_title="Maturité (années)", yaxis_title="Taux (%)",
            height=350, margin=dict(t=20)
        )
        st.plotly_chart(fig_rate, use_container_width=True)

        # Tableau des taux
        rate_df["zero_rate"] = (rate_df["zero_rate"] * 100).round(3)
        rate_df.columns = ["Maturité (Y)", "Taux (%)"]
        st.dataframe(rate_df, use_container_width=True, hide_index=True)

    # ── Prix AAPL & volatilité 
    with col2:
        st.subheader("Historique AAPL (2 ans)")

        prices = pd.read_csv("docs/AAPL_prices.csv", parse_dates=["date"], index_col="date")
        prices = prices[prices.index >= prices.index.max() - pd.DateOffset(years=2)]

        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(
            x=prices.index, y=prices["Close"],
            mode="lines", name="Close", line=dict(color="#2ca02c")
        ))
        fig_px.update_layout(xaxis_title="Date", yaxis_title="Prix ($)", height=350,
                              margin=dict(t=20))
        st.plotly_chart(fig_px, use_container_width=True)

        st.subheader("Volatilité réalisée glissante (30j / 90j)")
        log_ret = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
        vol_30  = log_ret.rolling(30).std() * np.sqrt(252)
        vol_90  = log_ret.rolling(90).std() * np.sqrt(252)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=vol_30.index, y=vol_30 * 100,
                                      mode="lines", name="Vol 30j", line=dict(color="orange")))
        fig_vol.add_trace(go.Scatter(x=vol_90.index, y=vol_90 * 100,
                                      mode="lines", name="Vol 90j", line=dict(color="purple")))
        fig_vol.update_layout(xaxis_title="Date", yaxis_title="Vol annualisée (%)",
                               height=300, margin=dict(t=20))
        st.plotly_chart(fig_vol, use_container_width=True)

    #  Surface de volatilité implicite (paramétrique, terme structure) 
    st.subheader("Structure par terme de volatilité (réalisée)")
    snap = mkt["snap"]
    vol_ts = snap.get("vol_term_structure", {})
    if vol_ts:
        mats_vol = [float(k) for k in vol_ts.keys()]
        vols     = [float(v) * 100 for v in vol_ts.values()]
        fig_vts = go.Figure(go.Bar(x=[f"{m:.0%}" for m in mats_vol], y=vols,
                                    marker_color="#ff7f0e"))
        fig_vts.update_layout(xaxis_title="Maturité", yaxis_title="Vol (%)",
                               height=280, margin=dict(t=20))
        st.plotly_chart(fig_vts, use_container_width=True)



elif page == "🔧 Pricer Unitaire":
    st.title("🔧 Pricer Unitaire")

    product_type = st.selectbox("Type de produit", [
        "Obligation Zéro-Coupon",
        "Obligation à Coupons",
        "Swap Fixe / Variable",
        "Call / Put Vanille",
        "Call Spread",
        "Put Spread",
        "Butterfly",
        "Option à Barrière (KO/KI)",
        "Autocallable",
        "Note Structurée",
    ])

    # Paramètres communs (sidebar-like dans le corps principal)
    with st.expander("🔧 Paramètres de marché (modifiables)", expanded=False):
        S_ui   = st.number_input("Spot AAPL ($)", value=float(S), step=1.0)
        sig_ui = st.number_input("Volatilité (%)", value=float(sig * 100), step=0.5) / 100
        r_ui   = st.number_input("Taux sans risque (%)", value=float(r * 100), step=0.1) / 100
        q_ui   = st.number_input("Div. yield (%)", value=float(q * 100), step=0.01) / 100

    st.markdown("---")

    # Obligation Zéro-Coupon 
    if product_type == "Obligation Zéro-Coupon":
        nominal = st.number_input("Nominal (€)", value=1_000_000, step=100_000)
        T_ui    = st.number_input("Maturité (années)", value=2.0, step=0.25)
        if st.button("Pricer"):
            p = ZeroCouponBond(nominal, T_ui, curve)
            st.metric("Prix", f"€ {p.price():,.2f}")
            g = p.greeks()
            st.json(g)

    #  Obligation à Coupons
    elif product_type == "Obligation à Coupons":
        nominal   = st.number_input("Nominal (€)", value=1_000_000, step=100_000)
        cpn_rate  = st.number_input("Taux coupon (%)", value=5.0, step=0.25) / 100
        freq      = st.selectbox("Fréquence", [1, 2, 4, 12], index=1)
        T_ui      = st.number_input("Maturité (années)", value=5.0, step=0.5)
        if st.button("Pricer"):
            p = CouponBond(nominal, cpn_rate, freq, T_ui, curve)
            c1, c2, c3 = st.columns(3)
            c1.metric("Prix", f"€ {p.price():,.2f}")
            c2.metric("YTM", f"{p.ytm():.3%}")
            c3.metric("Duration", f"{p.duration():.2f} ans")

    #  Swap 
    elif product_type == "Swap Fixe / Variable":
        nominal   = st.number_input("Nominal (€)", value=1_000_000, step=100_000)
        fixed_rt  = st.number_input("Taux fixe (%)", value=5.0, step=0.1) / 100
        freq      = st.selectbox("Fréquence fixe", [1, 2, 4, 12], index=3)
        T_ui      = st.number_input("Maturité (années)", value=2.0, step=0.25)
        payer     = st.radio("Position", ["Payeur fixe", "Receveur fixe"]) == "Payeur fixe"
        if st.button("Pricer"):
            p = InterestRateSwap(nominal, fixed_rt, freq, T_ui, curve, payer)
            c1, c2 = st.columns(2)
            npv = p.price()
            c1.metric("NPV", f"€ {npv:,.2f}")
            c2.metric("Taux de marché (par)", f"{p.par_rate():.3%}")
            st.metric("DV01", f"€ {p.greeks()['dv01']:,.1f}")

    #  Options Vanilles 
    elif product_type == "Call / Put Vanille":
        opt_type = st.radio("Type", ["call", "put"])
        K_ui     = st.number_input("Strike", value=260.0, step=5.0)
        T_ui     = st.number_input("Maturité (années)", value=0.5, step=0.1)
        qty      = st.number_input("Quantité", value=1, step=1)
        if st.button("Pricer"):
            p = EuropeanOption(S_ui, K_ui, T_ui, r_ui, q_ui, sig_ui, opt_type, qty)
            g = p.greeks()
            c1, c2, c3 = st.columns(3)
            c1.metric("Prix unitaire", f"${p.price()/qty:.4f}")
            c2.metric("Prix total", f"${p.price():,.2f}")
            c3.metric("Vol implicite utilisée", f"{sig_ui:.1%}")

            cols = st.columns(5)
            for i, (k, v) in enumerate(g.items()):
                cols[i % 5].metric(k.capitalize(), f"{v:.4f}")

            # Profit & Loss à maturité
            st.subheader("Profil de payoff à maturité")
            Ss   = np.linspace(S_ui * 0.5, S_ui * 1.5, 200)
            pnl  = [qty * (max(s - K_ui, 0) if opt_type == "call" else max(K_ui - s, 0))
                    - p.price() for s in Ss]
            fig_pnl = go.Figure(go.Scatter(x=Ss, y=pnl, mode="lines",
                                            line=dict(color="#1f77b4")))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_pnl.add_vline(x=S_ui, line_dash="dot", line_color="red",
                              annotation_text=f"Spot={S_ui:.0f}")
            fig_pnl.update_layout(xaxis_title="S_T", yaxis_title="P&L",
                                   height=300, margin=dict(t=10))
            st.plotly_chart(fig_pnl, use_container_width=True)

    #  Call Spread 
    elif product_type == "Call Spread":
        K1   = st.number_input("Strike bas (K1)", value=240.0, step=5.0)
        K2   = st.number_input("Strike haut (K2)", value=270.0, step=5.0)
        T_ui = st.number_input("Maturité (années)", value=0.5, step=0.1)
        qty  = st.number_input("Quantité", value=1, step=1)
        if st.button("Pricer"):
            p = CallSpread(S_ui, K1, K2, T_ui, r_ui, q_ui, sig_ui, qty)
            g = p.greeks()
            st.metric("Prix", f"${p.price():,.4f}")

            Ss  = np.linspace(S_ui * 0.7, S_ui * 1.3, 200)
            pnl = [qty * min(max(s - K1, 0), K2 - K1) - p.price() for s in Ss]
            fig = go.Figure(go.Scatter(x=Ss, y=pnl, mode="lines", line=dict(color="green")))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(xaxis_title="S_T", yaxis_title="P&L", height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

            st.json({k: round(v, 4) for k, v in g.items()})

    #  Put Spread 
    elif product_type == "Put Spread":
        K1   = st.number_input("Strike bas (K1)", value=200.0, step=5.0)
        K2   = st.number_input("Strike haut (K2)", value=230.0, step=5.0)
        T_ui = st.number_input("Maturité (années)", value=0.5, step=0.1)
        qty  = st.number_input("Quantité", value=1, step=1)
        if st.button("Pricer"):
            p = PutSpread(S_ui, K1, K2, T_ui, r_ui, q_ui, sig_ui, qty)
            st.metric("Prix", f"${p.price():,.4f}")
            Ss  = np.linspace(S_ui * 0.7, S_ui * 1.3, 200)
            pnl = [qty * min(max(K2 - s, 0), K2 - K1) - p.price() for s in Ss]
            fig = go.Figure(go.Scatter(x=Ss, y=pnl, mode="lines", line=dict(color="orange")))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(xaxis_title="S_T", yaxis_title="P&L", height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    #  Butterfly 
    elif product_type == "Butterfly":
        K1   = st.number_input("Strike bas (K1)", value=220.0, step=5.0)
        K2   = st.number_input("Strike central (K2)", value=250.0, step=5.0)
        K3   = st.number_input("Strike haut (K3)", value=280.0, step=5.0)
        T_ui = st.number_input("Maturité (années)", value=0.5, step=0.1)
        qty  = st.number_input("Quantité", value=1, step=1)
        if st.button("Pricer"):
            p = Butterfly(S_ui, K1, K2, K3, T_ui, r_ui, q_ui, sig_ui, qty)
            st.metric("Prix", f"${p.price():,.4f}")
            Ss  = np.linspace(S_ui * 0.7, S_ui * 1.3, 200)
            pnl = [qty * (max(s - K1, 0) - 2 * max(s - K2, 0) + max(s - K3, 0)) - p.price()
                   for s in Ss]
            fig = go.Figure(go.Scatter(x=Ss, y=pnl, mode="lines", line=dict(color="purple")))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(xaxis_title="S_T", yaxis_title="P&L", height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    #  Barrière 
    elif product_type == "Option à Barrière (KO/KI)":
        opt_type  = st.radio("Type", ["call", "put"])
        K_ui      = st.number_input("Strike", value=260.0, step=5.0)
        barrier   = st.number_input("Barrière", value=290.0, step=5.0)
        bar_type  = st.selectbox("Type barrière", ["up-out", "up-in", "down-out", "down-in"])
        T_ui      = st.number_input("Maturité (années)", value=0.5, step=0.1)
        qty       = st.number_input("Quantité", value=1, step=1)
        if st.button("Pricer (Monte Carlo)"):
            with st.spinner("Simulation…"):
                p = BarrierOption(S_ui, K_ui, T_ui, r_ui, q_ui, sig_ui, barrier,
                                  bar_type, opt_type, qty, n_paths=30_000)
                vanilla = EuropeanOption(S_ui, K_ui, T_ui, r_ui, q_ui, sig_ui, opt_type, qty)
            c1, c2 = st.columns(2)
            c1.metric(f"Barrière ({bar_type})", f"${p.price():,.4f}")
            c2.metric("Vanille équivalente", f"${vanilla.price():,.4f}")
            st.caption("Prix calculé par Monte Carlo (50 000 trajectoires)")

    #  Autocall 
    elif product_type == "Autocallable":
        st.markdown("**Structure :** bi-mensuel, niveau recall 100% → -5%/an (février), cpn=8%/an")
        n_obs   = st.slider("Nombre d'observations", 6, 30, 12)
        cpn_rt  = st.number_input("Taux de coupon (%/an)", value=8.0, step=0.5) / 100
        recall0 = st.number_input("Niveau de recall initial (%)", value=100.0, step=5.0) / 100

        obs = [i * 2 / 12 for i in range(1, n_obs + 1)]
        recalls = []
        lvl = recall0
        for i, t in enumerate(obs):
            recalls.append(lvl)
            if round(t % 1, 2) in [0.08, 0.17]:  # ~Feb chaque année
                lvl = max(lvl - 0.05, 0.5)

        if st.button("Pricer (Monte Carlo)"):
            with st.spinner("Simulation…"):
                p = AutocallProduct(S_ui, 1.0, obs, recalls, cpn_rt, 0.0,
                                    r_ui, q_ui, sig_ui, n_paths=30_000)
            st.metric("Prix (pour nominal=1)", f"{p.price():.6f}")

            # Afficher le schedule
            sched = pd.DataFrame({"T (années)": obs, "Recall level": recalls})
            st.dataframe(sched, hide_index=True, use_container_width=True)

    #  Note Structurée 
    elif product_type == "Note Structurée":
        sspa   = st.selectbox("Code SSPA", [1100, 1130, 1220, 1320])
        nominal = st.number_input("Nominal unitaire ($)", value=1.0, step=0.1)
        qty    = st.number_input("Quantité", value=100, step=10)
        T_ui   = st.number_input("Maturité (années)", value=2.0, step=0.25)
        partcp = st.number_input("Participation (%)", value=100.0, step=10.0) / 100

        cap_val = bar1_val = bar2_val = None
        if sspa == 1130:
            bar1_val = st.number_input("Barrière KO ($)", value=290.0, step=5.0)
        if sspa == 1220:
            cap_val  = st.number_input("Cap ($)", value=280.0, step=5.0)
        if sspa == 1320:
            bar1_val = st.number_input("Barrière basse ($)", value=220.0, step=5.0)
            bar2_val = st.number_input("Barrière haute ($)", value=250.0, step=5.0)

        if st.button("Pricer (Monte Carlo)"):
            with st.spinner("Simulation…"):
                p = StructuredNote(sspa, nominal, qty, S_ui, T_ui, r_ui, q_ui, sig_ui,
                                   partcp, cap_val, bar1_val, bar2_val, n_paths=30_000)
            st.metric("Prix total", f"${p.price():,.2f}")
            st.metric("Prix unitaire", f"${p.price()/qty:.4f}")
            g = p.greeks()
            st.json({k: round(v, 4) for k, v in g.items()})


elif page == "💼 Portefeuille":
    st.title("💼 Vue Portefeuille – Inventaire Complet")

    with st.spinner("Pricing en cours (Monte Carlo inclus)…"):
        df, dm_delta, dm_vega, ptfs = get_portfolio_df()

    #  Résumé agrégé 
    st.subheader("📊 Prix agrégé par portefeuille")

    summary = df.groupby("Portefeuille")["Prix"].agg(["sum", "count"])
    summary.columns = ["Valeur totale", "Nb positions"]
    summary["Valeur totale"] = summary["Valeur totale"].map("${:,.2f}".format)
    st.dataframe(summary, use_container_width=True)

    grand_total = df["Prix"].sum()
    st.metric("💰 Valeur totale du portefeuille", f"${grand_total:,.2f}")

    #  Tableau détaillé 
    st.subheader("📋 Positions détaillées")
    ptf_filter = st.multiselect(
        "Filtrer par portefeuille",
        options=df["Portefeuille"].unique().tolist(),
        default=df["Portefeuille"].unique().tolist(),
    )
    df_show = df[df["Portefeuille"].isin(ptf_filter)].copy()

    # Formater les colonnes numériques
    num_cols = df_show.select_dtypes(include="number").columns
    st.dataframe(
        df_show.style.format({c: "{:.4f}" for c in num_cols}),
        use_container_width=True, hide_index=True,
    )

    #  Matrice de risque Delta 
    st.subheader("🎯 Matrice de risque – Delta par pilier de maturité")
    fig_delta = px.imshow(
        dm_delta.astype(float),
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Delta agrégé par portefeuille × maturité",
    )
    fig_delta.update_layout(height=350, margin=dict(t=40))
    st.plotly_chart(fig_delta, use_container_width=True)

    #  Matrice de risque Vega 
    st.subheader("📉 Matrice de risque – Vega par pilier de maturité")
    fig_vega = px.imshow(
        dm_vega.fillna(0).astype(float),
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="Vega agrégé par portefeuille × maturité",
    )
    fig_vega.update_layout(height=350, margin=dict(t=40))
    st.plotly_chart(fig_vega, use_container_width=True)

    #  Distribution des prix 
    st.subheader("📊 Distribution de la valeur par portefeuille")
    fig_bar = px.bar(
        df.groupby("Portefeuille")["Prix"].sum().reset_index(),
        x="Portefeuille", y="Prix", color="Portefeuille",
        text_auto=".2s",
    )
    fig_bar.update_layout(showlegend=False, height=300, margin=dict(t=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    #  Ajouter un nouveau sous-jacent 
    st.markdown("---")
    st.subheader("➕ Ajouter un nouveau sous-jacent")
    with st.form("new_underlying"):
        new_ticker = st.text_input("Ticker (ex: MSFT, TSLA…)", value="MSFT")
        new_spot   = st.number_input("Spot ($)", value=400.0, step=10.0)
        new_sigma  = st.number_input("Vol (%)", value=25.0, step=1.0) / 100
        new_q      = st.number_input("Div yield (%)", value=0.5, step=0.1) / 100
        new_K      = st.number_input("Strike pour un Call ATM", value=400.0, step=10.0)
        new_T      = st.number_input("Maturité (ans)", value=1.0, step=0.25)
        submitted  = st.form_submit_button("Calculer Call ATM")
        if submitted:
            call_px = BlackScholes.price(new_spot, new_K, new_T, r, new_q, new_sigma, "call")
            g       = BlackScholes.all_greeks(new_spot, new_K, new_T, r, new_q, new_sigma, "call")
            st.success(f"Call ATM {new_ticker} → ${call_px:.4f}")
            st.json({k: round(v, 4) for k, v in g.items()})
