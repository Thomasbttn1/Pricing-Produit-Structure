
import json
import numpy as np
import pandas as pd
from datetime import datetime, date

from models import RateCurve, BlackScholes
from products import (
    ZeroCouponBond, CouponBond,
    InterestRateSwap, BasisSwap,
    EuropeanOption, BarrierOption,
    CallSpread, PutSpread, Butterfly,
    AutocallProduct, StructuredNote,
)


VALUATION_DATE = datetime(2026, 2, 27)


def _years(dt) -> float:
    """Convertit une date ou datetime en nombre d'années depuis VALUATION_DATE."""
    if dt is None:
        return 0.0
    if isinstance(dt, (datetime, date)):
        d = dt if isinstance(dt, datetime) else datetime.combine(dt, datetime.min.time())
    else:
        return float(dt)
    delta = (d - VALUATION_DATE).days / 365.25
    return max(delta, 0.0)


def _freq_from_str(freq_str) -> float:
    """Convertit une fréquence textuelle en nombre de paiements par an."""
    if freq_str is None:
        return 1.0
    mapping = {"1M": 12, "3M": 4, "6M": 2, "12M": 1, "1Y": 1}
    return float(mapping.get(str(freq_str).strip().upper(), 1))



def load_market_data() -> dict:
    """
    Charge le snapshot de marché depuis docs/ et construit la courbe de taux.
    """
    # Snapshot AAPL
    with open("docs/market_snapshot.json") as f:
        snap = json.load(f)

    # Courbe de taux (bootstrap depuis FRED)
    rate_df = pd.read_csv("docs/rate_curve.csv")
    rate_dict = dict(zip(rate_df["maturity_years"], rate_df["zero_rate"]))

    # Calibration Nelson-Siegel pour une courbe lisse
    mats  = sorted(rate_dict.keys())
    rates = [rate_dict[m] for m in mats]
    try:
        curve = RateCurve.nelson_siegel(mats, rates)
    except Exception:
        curve = RateCurve.from_dict(rate_dict)

    r_short = curve.zero_rate(0.25)   # taux 3M comme taux sans risque de référence

    return {
        "spot":       snap["spot"],
        "sigma":      snap["hist_vol_1y"],    # vol historique 1Y
        "div_yield":  snap["dividend_yield"],
        "r":          r_short,
        "curve":      curve,
        "snap":       snap,
    }



def load_portfolio(xlsx_path: str = "Inventaire.xlsx") -> dict:
    """
    Lit Inventaire.xlsx et retourne un dict { portefeuille: [list of BaseProduct] }.
    """
    mkt = load_market_data()
    S   = mkt["spot"]
    sig = mkt["sigma"]
    q   = mkt["div_yield"]
    curve = mkt["curve"]

    portfolios = {}

    #  Portefeuille 1 : Swaps 
    df_swap = pd.read_excel(xlsx_path, sheet_name="Swap", header=0)
    swaps = []
    for _, row in df_swap.iterrows():
        mat_dt = row.get("Maturité")
        T      = _years(mat_dt)
        if T <= 0:
            continue
        nominal    = float(row["Nominal"])
        fixed_rate = row.get("Taux fixe")
        freq_str   = row.get("Fréquence fixe")
        var1       = row.get("Taux Variable 1")
        var2       = row.get("Taux Variable 2")
        r_ref      = curve.zero_rate(0.25)   # taux variable court

        if pd.notna(var1) and pd.isna(var2):
            # Swap fixe / variable
            if pd.isna(fixed_rate):
                continue
            freq = _freq_from_str(freq_str)
            swaps.append({
                "product":  InterestRateSwap(nominal, float(fixed_rate), freq, T, curve),
                "label":    f"IRS {nominal/1e6:.1f}M  mat={mat_dt}  fix={fixed_rate:.2%}",
                "nominal":  nominal,
                "maturity": T,
            })
        elif pd.notna(var1) and pd.notna(var2):
            # Basis swap (deux jambes variables)
            freq1 = _freq_from_str(str(var1))
            freq2 = _freq_from_str(str(var2))
            swaps.append({
                "product":  BasisSwap(nominal, freq1, freq2, T, curve),
                "label":    f"Basis {nominal/1e6:.1f}M  {var1} vs {var2}  mat={mat_dt}",
                "nominal":  nominal,
                "maturity": T,
            })

    portfolios["Swap"] = swaps

    #  Portefeuille 2 : Options 
    df_opt = pd.read_excel(xlsx_path, sheet_name="Options", header=0)
    options = []
    for _, row in df_opt.iterrows():
        produit = str(row.get("Produit", "")).strip()
        qty     = float(row.get("Quantité", 1))
        mat_dt  = row.get("Maturité")
        T       = _years(mat_dt)
        if T <= 0:
            continue
        K1  = float(row["Strike 1"])
        K2  = float(row["Strike 2"]) if pd.notna(row.get("Strike 2")) else None
        K3  = float(row["Strike 3"]) if pd.notna(row.get("Strike 3")) else None
        bar_type = row.get("Type Barrière")
        bar_lvl  = float(row["Niveau Barrière"]) if pd.notna(row.get("Niveau Barrière")) else None
        r_ref    = curve.zero_rate(T)

        if produit == "Call Spread":
            p = CallSpread(S, K1, K2, T, r_ref, q, sig, quantity=abs(qty))
            if qty < 0:  # inversion signe
                p = CallSpread(S, K1, K2, T, r_ref, q, sig, quantity=-abs(qty))

        elif produit == "Put Spread":
            p = PutSpread(S, K1, K2, T, r_ref, q, sig, quantity=abs(qty))

        elif produit == "Butterfly":
            K2_use = K2 if K2 else (K1 + K3) / 2
            p = Butterfly(S, K1, K2_use, K3, T, r_ref, q, sig, quantity=abs(qty))

        elif produit == "Call":
            bt = None
            if pd.notna(bar_type) and bar_lvl:
                direction = "up" if bar_lvl > S else "down"
                inout     = "out" if str(bar_type).upper() == "OUT" else "in"
                p = BarrierOption(S, K1, T, r_ref, q, sig, bar_lvl,
                                  f"{direction}-{inout}", "call", qty)
            else:
                p = EuropeanOption(S, K1, T, r_ref, q, sig, "call", qty)

        elif produit == "Put":
            if pd.notna(bar_type) and bar_lvl:
                direction = "up" if bar_lvl > S else "down"
                inout     = "out" if str(bar_type).upper() == "OUT" else "in"
                p = BarrierOption(S, K1, T, r_ref, q, sig, bar_lvl,
                                  f"{direction}-{inout}", "put", qty)
            else:
                p = EuropeanOption(S, K1, T, r_ref, q, sig, "put", qty)
        else:
            continue

        options.append({
            "product":  p,
            "label":    f"{produit}  qty={qty}  K={K1}  mat={mat_dt}",
            "nominal":  qty,
            "maturity": T,
        })

    portfolios["Options"] = options

    #  Portefeuille 3 : Autocall 
    df_ac = pd.read_excel(xlsx_path, sheet_name="Autocall", header=0)

    # Construction du schedule d'observation à partir de la première ligne réelle
    # (les formules Excel sont évaluées statiquement → on recalcule)
    ref_date = datetime(2026, 1, 31)
    final_obs = datetime(2028, 2, 28)

    obs_dates_py = []
    recall_levels = []
    d = ref_date
    recall = 1.0
    month_count = 0
    while d <= final_obs:
        month_count += 2
        # avancer de 2 mois
        m = d.month + 2
        y = d.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        try:
            d = datetime(y, m, 28 if m == 2 else min(d.day, 30))
        except ValueError:
            d = datetime(y, m, 28)

        if d > final_obs:
            break
        obs_dates_py.append(_years(d))

        # Baisse de 5% chaque mois de février
        if d.month == 2:
            recall -= 0.05
        recall_levels.append(max(recall, 0.5))   # plancher à 50%

    autocalls = []
    if obs_dates_py:
        ac = AutocallProduct(
            S0           = S,
            nominal      = 1.0,       # normalisé à 1, multiplié par quantity côté portfolio
            obs_dates    = obs_dates_py,
            recall_levels = recall_levels,
            coupon_rate  = 0.08,
            ref_date_years = _years(ref_date),
            r            = curve.zero_rate(max(obs_dates_py)),
            q            = q,
            sigma        = sig,
            n_paths      = 30_000,
        )
        autocalls.append({
            "product":  ac,
            "label":    "Autocall AAPL  cpn=8%  mat=2028-02",
            "nominal":  1.0,
            "maturity": max(obs_dates_py),
        })

    portfolios["Autocall"] = autocalls

    #  Portefeuille 4 : Notes structurées 
    df_ns = pd.read_excel(xlsx_path, sheet_name="Notes structurées", header=0)
    notes = []
    SSPA_EXPECTED = {
        1100: dict(barrier1=None, barrier2=None, cap=None),
        1130: dict(barrier1=280.0, barrier2=None, cap=None),
        1220: dict(barrier1=None, barrier2=None, cap=280.0),
        1320: dict(barrier1=220.0, barrier2=250.0, cap=None),
    }
    for _, row in df_ns.iterrows():
        code = int(row.get("Code produit SSPA", 0))
        if code not in SSPA_EXPECTED:
            continue
        qty     = float(row.get("Quantité", 1))
        mat_dt  = row.get("Maturité")
        T       = _years(mat_dt)
        if T <= 0:
            continue
        partcp  = float(row["Taux de participation"]) if pd.notna(row.get("Taux de participation")) else 1.0
        cap     = float(row["Cap"]) if pd.notna(row.get("Cap")) else SSPA_EXPECTED[code]["cap"]
        bar1    = float(row["Barrière 1"]) if pd.notna(row.get("Barrière 1")) else SSPA_EXPECTED[code]["barrier1"]
        bar2    = float(row["Barrière 2"]) if pd.notna(row.get("Barrière 2")) else SSPA_EXPECTED[code]["barrier2"]
        r_ref   = curve.zero_rate(T)

        p = StructuredNote(
            sspa_code     = code,
            nominal       = 1.0,
            quantity      = qty,
            S0            = S,
            T             = T,
            r             = r_ref,
            q             = q,
            sigma         = sig,
            participation = partcp,
            cap           = cap,
            barrier1      = bar1,
            barrier2      = bar2,
            n_paths       = 30_000,
        )
        notes.append({
            "product":  p,
            "label":    f"Note SSPA {code}  qty={qty}  mat={mat_dt}",
            "nominal":  qty,
            "maturity": T,
        })

    portfolios["Notes structurées"] = notes

    return portfolios



def price_portfolio(portfolios: dict) -> pd.DataFrame:
    """
    Price tous les produits et retourne un DataFrame.

    Colonnes : portefeuille, label, maturity, price, delta, gamma, vega, theta, rho
    """
    rows = []
    for ptf_name, positions in portfolios.items():
        for pos in positions:
            prod  = pos["product"]
            try:
                px    = prod.price()
                greeks = prod.greeks()
            except Exception as e:
                px, greeks = float("nan"), {}

            row = {
                "Portefeuille": ptf_name,
                "Produit":      pos["label"],
                "Maturité (Y)": round(pos["maturity"], 2),
                "Prix":         round(px, 4),
                **{k.capitalize(): round(v, 6) for k, v in greeks.items()},
            }
            rows.append(row)

    return pd.DataFrame(rows)



def risk_matrix(portfolios: dict, greek: str = "delta") -> pd.DataFrame:
    """
    Construit la matrice de risque agrégée par pilier de maturité.

    Parameters
    ----------
    greek : str   Nom du greek à agréger ('delta', 'vega', 'dv01', …)

    Returns
    -------
    pd.DataFrame  index=portefeuille, columns=piliers de maturité
    """
    piliers = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    data = {ptf: {p: 0.0 for p in piliers} for ptf in portfolios}

    for ptf_name, positions in portfolios.items():
        for pos in positions:
            T    = pos["maturity"]
            prod = pos["product"]
            try:
                g = prod.greeks().get(greek, 0.0)
            except Exception:
                g = 0.0
            # Attribution au pilier le plus proche
            closest = min(piliers, key=lambda p: abs(p - T))
            data[ptf_name][closest] += g

    df = pd.DataFrame(data).T
    df.columns = [f"{p}Y" for p in piliers]
    return df




if __name__ == "__main__":
    print("Chargement du portefeuille…")
    ptfs = load_portfolio()
    for name, positions in ptfs.items():
        print(f"\n{'─'*60}\n  {name} ({len(positions)} position(s))")

    print("\nPricing en cours…")
    df = price_portfolio(ptfs)
    print(df.to_string())

    print("\nMatrice Delta :")
    print(risk_matrix(ptfs, "delta").to_string())
