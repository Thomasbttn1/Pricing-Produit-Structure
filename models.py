

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from scipy.interpolate import interp1d



class RateCurve:
    """
    Zero-coupon rate curve built by bootstrapping market rates  and interpolating linearly on zero rates.
    """

    def __init__(self, maturities: list, zero_rates: list):
       
        idx = np.argsort(maturities)
        self.maturities = np.array(maturities)[idx]
        self.zero_rates  = np.array(zero_rates)[idx]

        if len(self.maturities) > 1:
            self._interp = interp1d(
                self.maturities, self.zero_rates,
                kind="linear", fill_value="extrapolate"
            )
        else:
            r0 = self.zero_rates[0]
            self._interp = lambda t: r0



    @classmethod
    def from_dict(cls, rate_dict: dict):
        """Build a RateCurve from a {maturity: rate} dictionary."""
        mats = sorted(rate_dict.keys())
        rates = [rate_dict[m] for m in mats]
        return cls(mats, rates)

    @classmethod
    def flat(cls, rate: float = 0.04):
        
        return cls([0.001, 50.0], [rate, rate])


    @classmethod
    def nelson_siegel(cls, market_mats: list, market_rates: list):
        """
        Calibrate a Nelson-Siegel curve to market zero rates and return a
        RateCurve object built on a fine grid.

        """
        market_mats  = np.array(market_mats,  dtype=float)
        market_rates = np.array(market_rates, dtype=float)

        def ns_rate(T, b0, b1, b2, tau):
            T = np.maximum(T, 1e-6)
            x = T / tau
            fac = (1 - np.exp(-x)) / x
            return b0 + b1 * fac + b2 * (fac - np.exp(-x))

        def objective(params):
            b0, b1, b2, tau = params
            pred = ns_rate(market_mats, b0, b1, b2, tau)
            return np.sum((pred - market_rates) ** 2)

        # Initial guess
        x0 = [market_rates[-1], market_rates[0] - market_rates[-1], -0.01, 2.0]
        bounds = [(-0.1, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.1, 30.0)]
        res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        b0, b1, b2, tau = res.x

        # Build on fine grid
        fine_mats  = np.concatenate([
            np.linspace(0.01, 1.0, 20),
            np.linspace(1.0, 30.0, 60),
        ])
        fine_rates = ns_rate(fine_mats, b0, b1, b2, tau)
        return cls(fine_mats.tolist(), fine_rates.tolist())


    def zero_rate(self, T: float) -> float:
        """Continuously-compounded zero rate at maturity T (years)."""
        if T <= 0:
            return 0.0
        return float(self._interp(T))

    def discount_factor(self, T: float) -> float:
        """P(0, T) = exp(-r(T) * T)."""
        if T <= 0:
            return 1.0
        return float(np.exp(-self.zero_rate(T) * T))

    def forward_rate(self, T1: float, T2: float) -> float:
        """Instantaneous forward rate between T1 and T2."""
        if T2 <= T1:
            raise ValueError("T2 must be > T1")
        r1, r2 = self.zero_rate(T1), self.zero_rate(T2)
        return (r2 * T2 - r1 * T1) / (T2 - T1)

    def par_swap_rate(self, payment_dates_years: list) -> float:
        """
        Compute the par (fair) fixed rate for a swap with given annual payment dates.
        par_rate = (1 - DF(Tn)) / sum(τ_i * DF(T_i))
        """
        dfs   = [self.discount_factor(t) for t in payment_dates_years]
        taus  = [payment_dates_years[0]] + [payment_dates_years[i] - payment_dates_years[i-1]
                                             for i in range(1, len(payment_dates_years))]
        annuity = sum(tau * df for tau, df in zip(taus, dfs))
        return (1.0 - dfs[-1]) / annuity




class BlackScholes:
    """
    Analytical Black-Scholes pricing and Greeks for European vanilla options.
    """


    @staticmethod
    def d1(S, K, T, r, q, sigma):
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, q, sigma):
        return BlackScholes.d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)



    @staticmethod
    def price(S, K, T, r, q, sigma, option_type="call") -> float:
        """Price a European call or put."""
        if T <= 0:
            return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)
        if option_type == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


    @staticmethod
    def delta(S, K, T, r, q, sigma, option_type="call") -> float:
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        if option_type == "call":
            return float(np.exp(-q * T) * norm.cdf(d1))
        else:
            return float(np.exp(-q * T) * (norm.cdf(d1) - 1))

    @staticmethod
    def gamma(S, K, T, r, q, sigma) -> float:
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def vega(S, K, T, r, q, sigma) -> float:
        """Vega per 1 percentage-point move in volatility (i.e. divided by 100)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0)

    @staticmethod
    def theta(S, K, T, r, q, sigma, option_type="call") -> float:
        """Daily theta (divided by 365)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)
        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == "call":
            return float((term1
                          - r * K * np.exp(-r * T) * norm.cdf(d2)
                          + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365)
        else:
            return float((term1
                          + r * K * np.exp(-r * T) * norm.cdf(-d2)
                          - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365)

    @staticmethod
    def rho(S, K, T, r, q, sigma, option_type="call") -> float:
        """Rho per 1 percentage-point move in the risk-free rate."""
        if T <= 0:
            return 0.0
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)
        if option_type == "call":
            return float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0)
        else:
            return float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0)

    @staticmethod
    def all_greeks(S, K, T, r, q, sigma, option_type="call") -> dict:
        """Return all Greeks in a single dictionary."""
        return {
            "delta": BlackScholes.delta(S, K, T, r, q, sigma, option_type),
            "gamma": BlackScholes.gamma(S, K, T, r, q, sigma),
            "vega":  BlackScholes.vega(S, K, T, r, q, sigma),
            "theta": BlackScholes.theta(S, K, T, r, q, sigma, option_type),
            "rho":   BlackScholes.rho(S, K, T, r, q, sigma, option_type),
        }


    @staticmethod
    def implied_vol(S, K, T, r, q, market_price, option_type="call",
                    tol=1e-6, max_iter=200) -> float:
        """
        Compute implied volatility using the Newton-Raphson method.
        Returns NaN if convergence fails.
        """
        if T <= 0:
            return float("nan")
        # Lower bound for intrinsic value check
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option_type == "call" \
                    else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        if market_price < intrinsic - 1e-6:
            return float("nan")

        sigma = 0.20   # initial guess: 20%
        for _ in range(max_iter):
            price = BlackScholes.price(S, K, T, r, q, sigma, option_type)
            vega  = BlackScholes.vega(S, K, T, r, q, sigma) * 100   # un-normalize
            if abs(vega) < 1e-12:
                break
            diff  = price - market_price
            if abs(diff) < tol:
                return max(sigma, 1e-6)
            sigma -= diff / vega
            sigma  = max(sigma, 1e-6)
        return max(sigma, 1e-6)



class MonteCarlo:
    """
    Monte Carlo simulator under the Black-Scholes / GBM framework.
    """

    def __init__(self, S0: float, r: float, q: float, sigma: float, T: float,
                 n_paths: int = 50_000, n_steps: int = 252, seed: int = 42):
        self.S0      = S0
        self.r       = r
        self.q       = q
        self.sigma   = sigma
        self.T       = T
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed    = seed
        self._paths  = None   # cached after first simulate()

    def simulate(self) -> np.ndarray:
        """
        Simulate GBM paths using antithetic variates for variance reduction.
        """
        rng   = np.random.default_rng(self.seed)
        dt    = self.T / self.n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol   = self.sigma * np.sqrt(dt)

        half  = self.n_paths // 2
        Z     = rng.standard_normal((half, self.n_steps))
        Z     = np.vstack([Z, -Z])                         # antithetic variates

        log_S = np.cumsum(drift + vol * Z, axis=1)         # cumulative log-returns
        log_S = np.hstack([np.zeros((self.n_paths, 1)), log_S])
        self._paths = self.S0 * np.exp(log_S)
        return self._paths

    def price(self, payoff_fn, use_cached: bool = True) -> float:
        """
        Price an instrument given a payoff function.
        """
        paths = self._paths if (use_cached and self._paths is not None) else self.simulate()
        payoffs = payoff_fn(paths)
        return float(np.exp(-self.r * self.T) * np.mean(payoffs))
