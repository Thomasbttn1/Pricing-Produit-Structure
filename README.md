# Pricer Multi-Produits – Master 272 Dauphine

Application de pricing de produits financiers structurés, développée dans le cadre du cours *Modélisation et Pricing de Produits Structurés* (L. Davoust, 2026).

---

## Structure du projet

```
├── models.py        # Modèles : RateCurve (Nelson-Siegel), Black-Scholes, Monte Carlo
├── products.py      # Tous les produits financiers (12 classes)
├── portfolio.py     # Chargement de l'inventaire, pricing agrégé, matrice de risque
├── app.py           # Dashboard Streamlit (3 pages)
├── Inventaire.xlsx  # Inventaire des 4 portefeuilles
└── docs/
    ├── AAPL_prices.csv          # Historique de prix AAPL (5 ans, yfinance)
    ├── rate_curve.csv           # Courbe de taux US Treasury (FRED)
    ├── rate_curve_history.csv   # Historique 3 ans de la courbe
    └── market_snapshot.json     # Spot, vol 1Y, dividend yield, term structure
```

---

## Produits pricés

| Catégorie | Produits |
|-----------|----------|
| **Élémentaires** | Obligation Zéro-Coupon, Obligation à Coupons, Call / Put Européen |
| **Réplication statique** | IRS Fixe/Variable, Basis Swap, Call Spread, Put Spread, Butterfly |
| **Path-dependent** | Option à Barrière KO/KI, Autocallable, Notes Structurées (SSPA 1100/1130/1220/1320) |

---

## Modèles

- **Black-Scholes** : pricing analytique + Greeks (Δ, Γ, ν, Θ, ρ) pour les options vanilles et stratégies
- **Monte Carlo** : 50 000 trajectoires GBM avec variantes antithétiques, pour barrières, autocall et notes structurées
- **Courbe de taux** : bootstrap depuis FRED (US Treasury) + calibration Nelson-Siegel
- **Volatilité** : volatilité réalisée historique + structure par terme (1M → 24M)

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Lancer le dashboard

```bash
streamlit run app.py
```

---

## Dashboard – 3 pages

**1. Marché & Calibration**
- Courbe de taux Nelson-Siegel vs points FRED
- Historique AAPL et volatilité réalisée glissante (30j / 90j)
- Structure par terme de la volatilité

**2. Pricer Unitaire**
- Saisie manuelle de n'importe quel produit (12 types)
- Paramètres de marché modifiables (spot, vol, taux, dividende)
- Prix, Greeks, profil de payoff à maturité

**3. Portefeuille**
- Inventaire complet pricé (15 positions, 4 portefeuilles)
- Matrice de risque Delta et Vega par pilier de maturité
- Ajout d'un nouveau sous-jacent à la volée

---

## Données

| Source | Données | Utilisation |
|--------|---------|-------------|
| [yfinance](https://github.com/ranaroussi/yfinance) | Prix AAPL (OHLCV, 5 ans) | Spot, volatilité historique |
| [FRED](https://fred.stlouisfed.org/) | US Treasury (DGS1MO → DGS30) | Courbe de taux zéro-coupon |

Les données sont stockées statiquement dans `docs/` — aucune clé API n'est requise pour faire tourner l'application.

---

## Sous-jacent principal

**AAPL (Apple Inc.)** — Date de valorisation : 27 février 2026

| Paramètre | Valeur |
|-----------|--------|
| Spot | $253.79 |
| Vol 1Y réalisée | 31.3% |
| Dividend yield | 0.41% |
| Taux sans risque (3M) | ~3.7% |
