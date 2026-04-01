
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from models import BlackScholes, RateCurve, MonteCarlo



class BaseProduct(ABC):
    """Classe abstraite dont héritent tous les produits."""

    @abstractmethod
    def price(self) -> float:
        """Retourne le prix (valeur de marché) du produit."""
        ...

    def greeks(self) -> dict:
        """Retourne les Greeks (par défaut vide si non applicables)."""
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}(price={self.price():.4f})"



class ZeroCouponBond(BaseProduct):
    """
    Obligation Zéro-Coupon.

    """

    def __init__(self, nominal: float, T: float, curve: RateCurve):
        
        self.nominal = nominal
        self.T       = T
        self.curve   = curve

    def price(self) -> float:
        return self.nominal * self.curve.discount_factor(self.T)

    def duration(self) -> float:
        """Duration (= maturité pour un ZCB)."""
        return self.T

    def greeks(self) -> dict:
        """Rho : sensibilité au taux (DV01 approx)."""
        price_up   = self.nominal * np.exp(-(self.curve.zero_rate(self.T) + 0.0001) * self.T)
        price_down = self.nominal * np.exp(-(self.curve.zero_rate(self.T) - 0.0001) * self.T)
        dv01 = (price_down - price_up) / 2
        return {"dv01": dv01, "duration": self.T}


class CouponBond(BaseProduct):
    """
    Obligation à coupons périodiques.
    """

    def __init__(self, nominal: float, coupon_rate: float, freq_per_year: float,
                 T: float, curve: RateCurve):
       
    
        self.nominal       = nominal
        self.coupon_rate   = coupon_rate
        self.freq_per_year = freq_per_year
        self.T             = T
        self.curve         = curve

        # Dates des coupon (en années depuis aujourd'hui)
        step = 1.0 / freq_per_year
        self.payment_dates = np.arange(step, T + 1e-9, step)

    def price(self) -> float:
        coupon  = self.nominal * self.coupon_rate / self.freq_per_year
        pv      = sum(coupon * self.curve.discount_factor(t) for t in self.payment_dates)
        pv     += self.nominal * self.curve.discount_factor(self.T)
        return pv

    def ytm(self, tol=1e-8, max_iter=200) -> float:
        """Rendement actuariel (Yield to Maturity) par dichotomie."""
        target = self.price()
        coupon = self.nominal * self.coupon_rate / self.freq_per_year
        dates  = self.payment_dates

        def pv_at(y):
            pv = sum(coupon * np.exp(-y * t) for t in dates)
            pv += self.nominal * np.exp(-y * self.T)
            return pv

        lo, hi = 0.0, 1.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if pv_at(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return (lo + hi) / 2

    def duration(self) -> float:
        """Duration de Macaulay."""
        coupon = self.nominal * self.coupon_rate / self.freq_per_year
        p      = self.price()
        d      = sum(t * coupon * self.curve.discount_factor(t) for t in self.payment_dates)
        d     += self.T * self.nominal * self.curve.discount_factor(self.T)
        return d / p

    def greeks(self) -> dict:
        r0     = self.curve.zero_rate(self.T)
        p_up   = CouponBond(self.nominal, self.coupon_rate, self.freq_per_year, self.T,
                            RateCurve.flat(r0 + 0.0001)).price()
        p_down = CouponBond(self.nominal, self.coupon_rate, self.freq_per_year, self.T,
                            RateCurve.flat(r0 - 0.0001)).price()
        dv01 = (p_down - p_up) / 2
        return {"dv01": dv01, "duration": self.duration(), "ytm": self.ytm()}


class InterestRateSwap(BaseProduct):
    """
    Swap de taux d'intérêt (fixe / variable).
    """

    def __init__(self, nominal: float, fixed_rate: float, freq_per_year: float,
                 T: float, curve: RateCurve, payer: bool = True):
        
        self.nominal       = nominal
        self.fixed_rate    = fixed_rate
        self.freq_per_year = freq_per_year
        self.T             = T
        self.curve         = curve
        self.payer         = payer

        step = 1.0 / freq_per_year
        self.payment_dates = np.arange(step, T + 1e-9, step)

    def _pv_fixed_leg(self) -> float:
        tau    = 1.0 / self.freq_per_year
        coupon = self.nominal * self.fixed_rate * tau
        return sum(coupon * self.curve.discount_factor(t) for t in self.payment_dates)

    def _pv_float_leg(self) -> float:
        """PV jambe variable ≈ Nominal × (1 - DF(T)) via réplication par obligation."""
        return self.nominal * (1.0 - self.curve.discount_factor(self.T))

    def price(self) -> float:
        """NPV du swap (du côté payeur fixe si payer=True)."""
        npv = self._pv_float_leg() - self._pv_fixed_leg()
        return npv if self.payer else -npv

    def par_rate(self) -> float:
        """Taux de swap au pair (taux fixe qui annule le NPV)."""
        return self.curve.par_swap_rate(self.payment_dates.tolist())

    def greeks(self) -> dict:
        return {"dv01": self._dv01(), "par_rate": self.par_rate()}

    def _dv01(self) -> float:
        bump = 0.0001
        r0   = self.curve.zero_rate(self.T)
        up   = InterestRateSwap(self.nominal, self.fixed_rate, self.freq_per_year,
                                self.T, RateCurve.flat(r0 + bump), self.payer).price()
        dn   = InterestRateSwap(self.nominal, self.fixed_rate, self.freq_per_year,
                                self.T, RateCurve.flat(r0 - bump), self.payer).price()
        return (dn - up) / 2


class BasisSwap(BaseProduct):
    

    def __init__(self, nominal: float, freq1: float, freq2: float,
                 T: float, curve: RateCurve, spread: float = 0.0):
       
        self.nominal = nominal
        self.freq1   = freq1
        self.freq2   = freq2
        self.T       = T
        self.curve   = curve
        self.spread  = spread

    def _leg_pv(self, freq: float, extra_spread: float = 0.0) -> float:
        step  = 1.0 / freq
        dates = np.arange(step, self.T + 1e-9, step)
        pv    = 0.0
        prev  = 0.0
        for t in dates:
            fwd = self.curve.forward_rate(prev, t)
            tau = t - prev
            pv += self.nominal * (fwd + extra_spread) * tau * self.curve.discount_factor(t)
            prev = t
        return pv

    def price(self) -> float:
        """NPV = PV(jambe2) - PV(jambe1 + spread)."""
        return self._leg_pv(self.freq2) - self._leg_pv(self.freq1, self.spread)

    def greeks(self) -> dict:
        return {}


class EuropeanOption(BaseProduct):
    """
    Option européenne Call ou Put .
    """

    def __init__(self, S: float, K: float, T: float, r: float, q: float,
                 sigma: float, option_type: str = "call", quantity: float = 1.0):
    
        self.S    = S
        self.K    = K
        self.T    = T
        self.r    = r
        self.q    = q
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.quantity    = quantity

    def price(self) -> float:
        unit = BlackScholes.price(self.S, self.K, self.T, self.r, self.q,
                                   self.sigma, self.option_type)
        return unit * self.quantity

    def greeks(self) -> dict:
        g = BlackScholes.all_greeks(self.S, self.K, self.T, self.r, self.q,
                                     self.sigma, self.option_type)
        return {k: v * self.quantity for k, v in g.items()}


class BarrierOption(BaseProduct):
    """
    Option à barrière simple (KO ou KI) pricée par Monte Carlo sous Black-Scholes.

    barrier_type : "up-out", "up-in", "down-out", "down-in"
    """

    def __init__(self, S: float, K: float, T: float, r: float, q: float,
                 sigma: float, barrier: float, barrier_type: str,
                 option_type: str = "call", quantity: float = 1.0,
                 n_paths: int = 50_000, n_steps: int = 252):
        self.S            = S
        self.K            = K
        self.T            = T
        self.r            = r
        self.q            = q
        self.sigma        = sigma
        self.barrier      = barrier
        self.barrier_type = barrier_type.lower()
        self.option_type  = option_type.lower()
        self.quantity     = quantity
        self.n_paths      = n_paths
        self.n_steps      = n_steps

    def price(self) -> float:
        mc    = MonteCarlo(self.S, self.r, self.q, self.sigma, self.T,
                           self.n_paths, self.n_steps)
        paths = mc.simulate()
        S_T   = paths[:, -1]

        # Condition de barrière (vérifiée sur toute la trajectoire)
        if self.barrier_type == "down-out":
            alive = np.min(paths, axis=1) > self.barrier
        elif self.barrier_type == "down-in":
            alive = np.min(paths, axis=1) <= self.barrier
        elif self.barrier_type == "up-out":
            alive = np.max(paths, axis=1) < self.barrier
        elif self.barrier_type == "up-in":
            alive = np.max(paths, axis=1) >= self.barrier
        else:
            alive = np.ones(self.n_paths, dtype=bool)

        if self.option_type == "call":
            payoffs = np.maximum(S_T - self.K, 0) * alive
        else:
            payoffs = np.maximum(self.K - S_T, 0) * alive

        unit = float(np.exp(-self.r * self.T) * np.mean(payoffs))
        return unit * self.quantity

    def greeks(self) -> dict:
        """Greeks par différences finies (bump & reprice)."""
        p0    = self.price() / self.quantity
        bump  = self.S * 0.01

        self.S += bump
        p_up   = self.price() / self.quantity
        self.S -= 2 * bump
        p_dn   = self.price() / self.quantity
        self.S += bump

        delta = (p_up - p_dn) / (2 * bump)
        gamma = (p_up - 2 * p0 + p_dn) / (bump ** 2)

        sv = self.sigma * 0.01
        self.sigma += sv
        p_vup = self.price() / self.quantity
        self.sigma -= sv
        vega = (p_vup - p0) / sv / 100

        return {"delta": delta * self.quantity, "gamma": gamma * self.quantity,
                "vega": vega * self.quantity}




class CallSpread(BaseProduct):
    

    def __init__(self, S, K1, K2, T, r, q, sigma, quantity=1.0):
        self.long  = EuropeanOption(S, K1, T, r, q, sigma, "call",  quantity)
        self.short = EuropeanOption(S, K2, T, r, q, sigma, "call", -quantity)

    def price(self) -> float:
        return self.long.price() + self.short.price()

    def greeks(self) -> dict:
        g_l = self.long.greeks()
        g_s = self.short.greeks()
        return {k: g_l[k] + g_s[k] for k in g_l}


class PutSpread(BaseProduct):
   

    def __init__(self, S, K1, K2, T, r, q, sigma, quantity=1.0):
        self.long  = EuropeanOption(S, K2, T, r, q, sigma, "put",  quantity)
        self.short = EuropeanOption(S, K1, T, r, q, sigma, "put", -quantity)

    def price(self) -> float:
        return self.long.price() + self.short.price()

    def greeks(self) -> dict:
        g_l = self.long.greeks()
        g_s = self.short.greeks()
        return {k: g_l[k] + g_s[k] for k in g_l}


class Butterfly(BaseProduct):
    

    def __init__(self, S, K1, K2, K3, T, r, q, sigma, quantity=1.0):
        self.leg1 = EuropeanOption(S, K1, T, r, q, sigma, "call",  quantity)
        self.leg2 = EuropeanOption(S, K2, T, r, q, sigma, "call", -2 * quantity)
        self.leg3 = EuropeanOption(S, K3, T, r, q, sigma, "call",  quantity)

    def price(self) -> float:
        return self.leg1.price() + self.leg2.price() + self.leg3.price()

    def greeks(self) -> dict:
        keys = self.leg1.greeks().keys()
        return {k: self.leg1.greeks()[k] + self.leg2.greeks()[k] + self.leg3.greeks()[k]
                for k in keys}




class AutocallProduct(BaseProduct):
    """
    Produit Autocallable pricé par Monte Carlo.

    Mécanisme:
      - Observations bi-mensuelles (toutes les 2 périodes)
      - Niveau de rappel = 100%, baisse de 5% à chaque mois de février
      - Si rappelé à T_i : reçoit Nominal × (1 + coupon_rate × (T_i - T_ref) / 365)
      - Si jamais rappelé à la maturité finale : reçoit Nominal (sans coupon)

    """

    def __init__(self, S0: float, nominal: float,
                 obs_dates: list, recall_levels: list,
                 coupon_rate: float, ref_date_years: float,
                 r: float, q: float, sigma: float,
                 n_paths: int = 50_000, n_steps_per_year: int = 252):
        self.S0              = S0
        self.nominal         = nominal
        self.obs_dates       = sorted(obs_dates)
        self.recall_levels   = recall_levels
        self.coupon_rate     = coupon_rate
        self.ref_date_years  = ref_date_years
        self.r               = r
        self.q               = q
        self.sigma           = sigma
        self.n_paths         = n_paths
        self.T               = max(obs_dates)
        self.n_steps         = int(self.T * n_steps_per_year)

    def price(self) -> float:
        mc    = MonteCarlo(self.S0, self.r, self.q, self.sigma, self.T,
                           self.n_paths, self.n_steps)
        paths = mc.simulate()   # (n_paths, n_steps+1)

        payoffs = np.zeros(self.n_paths)
        recalled = np.zeros(self.n_paths, dtype=bool)

        for obs_t, recall_lvl in zip(self.obs_dates, self.recall_levels):
            # Index dans la grille temporelle
            idx = int(round(obs_t / self.T * self.n_steps))
            idx = min(idx, self.n_steps)
            S_obs = paths[:, idx]

            # Condition de rappel
            triggered = (~recalled) & (S_obs >= recall_lvl * self.S0)

            # Coupon = taux × (obs_t - ref_date_years) × nominal
            accrued = self.coupon_rate * max(obs_t - self.ref_date_years, 0) * self.nominal
            payoffs[triggered] = (self.nominal + accrued) * np.exp(-self.r * obs_t)
            recalled |= triggered

        # Paths jamais rappelés : remboursement du nominal à maturité
        payoffs[~recalled] = self.nominal * np.exp(-self.r * self.T)

        return float(np.mean(payoffs))

    def greeks(self) -> dict:
        p0   = self.price()
        bump = self.S0 * 0.01
        self.S0 += bump
        pu = self.price()
        self.S0 -= 2 * bump
        pd = self.price()
        self.S0 += bump
        delta = (pu - pd) / (2 * bump)

        sv = self.sigma * 0.01
        self.sigma += sv
        pv = self.price()
        self.sigma -= sv
        vega = (pv - p0) / sv / 100

        return {"delta": delta, "vega": vega}




class StructuredNote(BaseProduct):
    """
    Notes structurées mono-sous-jacent (classification SSPA simplifiée).

    Types supportés (code SSPA → structure) :
        1100  Tracker           : payoff = Nominal × S_T / S_0
        1130  Tracker KO        : idem mais KO (S_T ≥ barrier → rembourse Nominal)
        1220  Capped Tracker    : payoff = Nominal × min(S_T / S_0, cap / S_0)
        1320  Range Note        : Nominal + bonus si S_T ∈ [barrier1, barrier2]

    Tous pricés par Monte Carlo.
    """

    SSPA_LABELS = {
        1100: "Tracker",
        1130: "Tracker avec barrière KO",
        1220: "Tracker cappé",
        1320: "Range Note",
    }

    def __init__(self, sspa_code: int, nominal: float, quantity: float,
                 S0: float, T: float, r: float, q: float, sigma: float,
                 participation: float = 1.0,
                 cap: float = None, barrier1: float = None, barrier2: float = None,
                 bonus_rate: float = 0.10,
                 n_paths: int = 50_000, n_steps: int = 252):

        self.sspa_code    = sspa_code
        self.nominal      = nominal
        self.quantity     = quantity
        self.S0           = S0
        self.T            = T
        self.r            = r
        self.q            = q
        self.sigma        = sigma
        self.participation = participation
        self.cap          = cap
        self.barrier1     = barrier1
        self.barrier2     = barrier2
        self.bonus_rate   = bonus_rate
        self.n_paths      = n_paths
        self.n_steps      = n_steps

    @property
    def label(self):
        return self.SSPA_LABELS.get(self.sspa_code, f"SSPA {self.sspa_code}")

    def _simulate_terminal(self):
        mc = MonteCarlo(self.S0, self.r, self.q, self.sigma, self.T,
                        self.n_paths, self.n_steps)
        paths = mc.simulate()
        return paths, paths[:, -1]

    def _unit_price(self) -> float:
        paths, S_T = self._simulate_terminal()

        if self.sspa_code == 1100:
            # Tracker pur
            payoffs = self.nominal * self.participation * S_T / self.S0

        elif self.sspa_code == 1130:
            # Tracker avec KO Up (si S dépasse barrier1 → rembourse nominal)
            H = self.barrier1 if self.barrier1 else self.S0 * 1.3
            touched = np.max(paths, axis=1) >= H
            payoffs = np.where(
                touched,
                self.nominal,                                       # KO → remboursement
                self.nominal * self.participation * S_T / self.S0  # pas KO → tracker
            )

        elif self.sspa_code == 1220:
            # Tracker cappé
            cap_ratio = self.cap / self.S0 if self.cap else 1.3
            payoffs = self.nominal * np.minimum(
                self.participation * S_T / self.S0,
                cap_ratio
            )

        elif self.sspa_code == 1320:
            # Range Note : bonus si S_T ∈ [barrier1, barrier2]
            b1 = self.barrier1 if self.barrier1 else self.S0 * 0.85
            b2 = self.barrier2 if self.barrier2 else self.S0 * 1.05
            in_range = (S_T >= b1) & (S_T <= b2)
            payoffs = self.nominal * (1 + self.bonus_rate * in_range)

        else:
            # Fallback : tracker
            payoffs = self.nominal * S_T / self.S0

        return float(np.exp(-self.r * self.T) * np.mean(payoffs))

    def price(self) -> float:
        return self._unit_price() * self.quantity

    def greeks(self) -> dict:
        p0   = self._unit_price()
        bump = self.S0 * 0.01

        self.S0 += bump
        pu = self._unit_price()
        self.S0 -= 2 * bump
        pd = self._unit_price()
        self.S0 += bump

        delta = (pu - pd) / (2 * bump) * self.quantity
        gamma = (pu - 2 * p0 + pd) / (bump ** 2) * self.quantity

        sv = self.sigma * 0.01
        self.sigma += sv
        pv = self._unit_price()
        self.sigma -= sv
        vega = (pv - p0) / sv / 100 * self.quantity

        return {"delta": delta, "gamma": gamma, "vega": vega}
