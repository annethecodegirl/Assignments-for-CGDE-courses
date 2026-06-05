"""
Problem Set 3, Problem 2 — Numerical Replication and Sensitivity Analysis
PhD Macroeconomics I: Sustainability — Summer 2026

Model: Simplified two-region IAM from Azzimonti et al. (2025) Ch. 25,
       based on Golosov, Hassler, Krusell, Tsyvinski (2014) Econometrica.

Structure
---------
  Production    : Y_t = A_t K_t^alpha L^{1-alpha-nu} E_t^nu
  TFP           : A_t = exp(g*t - gamma*S_t)    [growth + damage]
  Energy        : CES aggregator over oil, coal, green
  Carbon cycle  : Two-component (permanent + transient), Golosov et al.
  Temperature   : T = ECS * log(S_atm / S_pre) / log(2)
  Oil supply    : Hotelling under log utility: e_o,t = (1-beta)*R_t
  Optimal tax   : tau* = gamma*Y_t * [phi_L/(1-beta) + (1-phi_L)/(1-beta*(1-d_T))]
                  (Golosov formula, eq. 25.32 in textbook)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BASELINE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# --- time ---
N  = 19                               # periods: 2020, 2030, …, 2200
dt = 10                               # years per period
YEARS = np.arange(2020, 2020 + dt*N, dt)

# --- preferences & technology ---
alpha = 0.30                          # capital share
nu    = 0.05                          # energy share
BETA  = 0.985**dt                     # per-decade discount factor ≈ 0.8597

# --- CES energy aggregator ---
# E = [lam_o e_o^rho + lam_c e_c^rho + lam_g e_g^rho]^{1/rho}
RHO  = 0.5                            # EoS = 1/(1-rho) = 2
LAM  = np.array([1/3, 1/3, 1/3])     # share weights: [oil, coal, green]

# --- carbon intensities (GtC per unit energy, *before* xi scaling) ---
KAPPA = np.array([1.0, 1.5, 0.0])    # oil, coal, green

# --- energy prices (normalized output units per unit energy) ---
P_COAL  = 1.00
P_GREEN = 2.00   # calibrated lower than Golosov's 3.5 so that the optimal
                  # tax eventually induces a green switch over the century
                  # (key driver of the 2°C outcome in Fig. 25.6)
P_OIL_0 = 1.00   # initial oil price; equilibrates via Hotelling

# --- productivity ---
G_YR  = 0.015                         # annual TFP growth
G_DEC = G_YR * dt                     # per-decade TFP growth

# --- carbon cycle (Golosov et al. 2014) ---
PHI_L = 0.20                          # permanent atmospheric fraction
D_T   = 1.0 - (1.0 - 0.0228)**dt    # transient decay per decade ≈ 0.206

# --- initial atmospheric carbon (year 2020) ---
S_PRE   = 596.4                       # GtC at 280 ppm (pre-industrial)
S_2020  = 893.5                       # GtC at ~420 ppm (2020)
S_EX_0  = S_2020 - S_PRE             # ≈ 297 GtC excess above pre-industrial

# historical cumulative emissions ≈ 360 GtC → permanent component ≈ PHI_L * 360
S_P0 = PHI_L * 360.0                 # ≈ 72 GtC permanent
S_T0 = S_EX_0 - S_P0                # ≈ 225 GtC transient

# --- climate ---
ECS = 3.0                             # equilibrium climate sensitivity (°C)

def temp(Sp, St):
    """Temperature above pre-industrial (°C) using log forcing."""
    return ECS * np.log((S_PRE + Sp + St) / S_PRE) / np.log(2.0)

T0 = temp(S_P0, S_T0)   # initial temperature (slightly > 1.1°C due to ECS vs TCR)
print(f"Initial temperature: {T0:.2f}°C  (obs. ~1.1°C; difference reflects ECS vs TCR)")

# --- Nordhaus damage parameter (Golosov calibration) ---
GAMMA = 5.3e-5            # per GtC

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CALIBRATE STEADY-STATE INITIAL CAPITAL AND EMISSION SCALE FACTOR xi
# ─────────────────────────────────────────────────────────────────────────────
# We want the economy to start approximately on its balanced growth path, so
# that capital doesn't jump discontinuously in period 1.
#
# BGP condition (no tax, no damage at t=0, L=1):
#   K_ss^{1-alpha-nu/(1-nu)*alpha} = alpha*beta * A0 * (nu*A0/P0)^{nu/(1-nu)}
# Solved numerically below.

def _p0_no_tax():
    p_hat = np.array([P_OIL_0, P_COAL, P_GREEN])
    e1 = 1/(1-RHO); e2 = RHO/(RHO-1); e3 = (RHO-1)/RHO
    return (np.sum(LAM**e1 * p_hat**e2))**e3

P0_NO_TAX = _p0_no_tax()

def _bgp_K(K):
    """BGP residual: K - alpha*beta*Y(K)."""
    A0 = np.exp(-GAMMA * S_EX_0)   # TFP with current damage
    E  = (nu * A0 * K**alpha / P0_NO_TAX)**(1/(1-nu))
    Y  = A0 * K**alpha * E**nu
    return K - alpha * BETA * Y

K_SS = brentq(_bgp_K, 1e-6, 1e3)
print(f"Steady-state initial capital K_0 = {K_SS:.5f}")

# emission scale factor xi: maps normalized energy units → GtC
A0_  = np.exp(-GAMMA * S_EX_0)
E0_  = (nu * A0_ * K_SS**alpha / P0_NO_TAX)**(1/(1-nu))
p0   = np.array([P_OIL_0, P_COAL, P_GREEN])
e0_  = E0_ * (P0_NO_TAX * LAM / p0)**(1/(1-RHO))
M0_  = np.dot(KAPPA, e0_)           # normalized emissions in period 0

M0_TARGET = 100.0                    # GtC/decade ≈ 10 GtC/year (current)
XI = M0_TARGET / M0_                 # scale factor
print(f"Emission scale factor  xi = {XI:.2f}")
print(f"Initial oil reserves   R_0: calibrated to clear market")

# initial oil reserves from Hotelling supply = demand at P_OIL_0
R_0 = e0_[0] / (1 - BETA)
print(f"  R_0 = {R_0:.5f} (normalized energy units)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  OPTIMAL CARBON TAX (Golosov eq. 25.32)
# ─────────────────────────────────────────────────────────────────────────────

SCC_FACTOR = PHI_L / (1 - BETA) + (1 - PHI_L) / (1 - BETA*(1 - D_T))
print(f"\nSCC factor = {SCC_FACTOR:.3f}")

def tau_star_GtC(Y):
    """Optimal carbon tax in [output / GtC]."""
    return GAMMA * Y * SCC_FACTOR

def tau_star_energy(Y, kappa_vec=KAPPA):
    """Optimal tax converted to per-energy-unit prices for each fuel type."""
    tg = tau_star_GtC(Y)
    return tg * kappa_vec * XI   # [output / energy-unit], per fuel

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENERGY-MARKET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ces_P(p_hat):
    """CES price index."""
    e1, e2, e3 = 1/(1-RHO), RHO/(RHO-1), (RHO-1)/RHO
    return (np.sum(LAM**e1 * p_hat**e2))**e3

def energy_demand(K, A, P):
    """Total energy demand: nu*A*K^alpha*E^{nu-1}=P."""
    return (nu * A * K**alpha / P)**(1/(1-nu))

def energy_inputs(E, P, p_hat):
    """Individual energy inputs from CES cost min."""
    return E * (P * LAM / p_hat)**(1/(1-RHO))

def oil_residual(p_oil, K, A, R, tau_e):
    """Hotelling supply minus CES demand for oil."""
    p_hat = np.array([p_oil + tau_e[0], P_COAL + tau_e[1], P_GREEN + tau_e[2]])
    P_    = ces_P(p_hat)
    E_    = energy_demand(K, A, P_)
    e_oil = energy_inputs(E_, P_, p_hat)[0]
    return (1 - BETA) * R - e_oil

def clear_oil_price(K, A, R, tau_e):
    """Find equilibrium oil price (Hotelling supply = CES demand)."""
    if R <= 0:
        return 1e8
    f_lo = oil_residual(1e-6, K, A, R, tau_e)
    f_hi = oil_residual(5e3,  K, A, R, tau_e)
    if f_lo * f_hi > 0:          # no sign change – supply < demand at all prices;
        return P_OIL_0           # use initial price as fallback
    return brentq(oil_residual, 1e-6, 5e3, args=(K, A, R, tau_e),
                  xtol=1e-8, maxiter=200)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def simulate(policy='no_tax', rho_val=None, p_green_val=None,
             gamma_growth=0.0):
    """
    Simulate the Golosov IAM.

    Parameters
    ----------
    policy       : 'no_tax' | 'optimal' | 'moderate'
    rho_val      : override CES parameter rho  (for Sensitivity 2)
    p_green_val  : override green energy price  (for custom runs)
    gamma_growth : extra damage parameter on TFP *growth*  (Sensitivity 3)
                   TFP = exp(g*(1-gamma_growth*S)*t - gamma*S)
    """
    rho_  = rho_val      if rho_val      is not None else RHO
    pg_   = p_green_val  if p_green_val  is not None else P_GREEN
    xi_   = XI           # emission scale factor (unchanged across scenarios)

    # pre-compute CES exponents for this rho
    # Special case: rho=0 → Cobb-Douglas  E = prod_k e_k^lam_k
    cd_case = (abs(rho_) < 1e-8)

    if not cd_case:
        e1_ = 1/(1-rho_); e2_ = rho_/(rho_-1)
        def _P(p_hat):
            e3_ = (rho_-1)/rho_
            return (np.sum(LAM**e1_ * p_hat**e2_))**e3_
        def _e(E, P, p_hat):
            return E * (P * LAM / p_hat)**(1/(1-rho_))
    else:
        # Cobb-Douglas: P = prod_k (p_k/lam_k)^lam_k
        def _P(p_hat):
            return np.prod((p_hat / LAM)**LAM)
        # inputs: e_k = lam_k * E * P / p_k
        def _e(E, P, p_hat):
            return LAM * E * P / p_hat

    def _E(K, A, P):
        return (nu * A * K**alpha / P)**(1/(1-nu))

    def _oil_res(p_oil, K, A, R, tau_e):
        p_hat = np.array([p_oil + tau_e[0], P_COAL + tau_e[1], pg_ + tau_e[2]])
        P_ = _P(p_hat); E_ = _E(K, A, P_); e_ = _e(E_, P_, p_hat)
        return (1-BETA)*R - e_[0]

    def _clear_oil(K, A, R, tau_e):
        if R <= 0:
            return 1e8
        try:
            fl = _oil_res(1e-6, K, A, R, tau_e)
            fh = _oil_res(5e3,  K, A, R, tau_e)
            if fl*fh > 0:
                return P_OIL_0
            return brentq(_oil_res, 1e-6, 5e3, args=(K,A,R,tau_e), xtol=1e-8)
        except Exception:
            return P_OIL_0

    # --- allocate output arrays ---
    K_v  = np.zeros(N); Y_v  = np.zeros(N); E_v = np.zeros(N)
    e_v  = np.zeros((N,3))
    R_v  = np.zeros(N); M_v  = np.zeros(N)
    Sp_v = np.zeros(N); St_v = np.zeros(N); T_v = np.zeros(N)
    tau_v= np.zeros(N); po_v = np.zeros(N)

    # --- initial conditions ---
    K_v[0]  = K_SS; R_v[0]  = R_0
    Sp_v[0] = S_P0; St_v[0] = S_T0; T_v[0] = T0

    for i in range(N):
        S_ex = Sp_v[i] + St_v[i]

        # TFP: standard growth + optional growth damage
        g_eff = G_DEC * (1.0 - gamma_growth * S_ex)
        A_i   = np.exp(g_eff * i - GAMMA * S_ex)

        # --- carbon tax ---
        Y_ref = Y_v[i-1] if i > 0 else 1.0
        if   policy == 'no_tax':   tau_GtC = 0.0
        elif policy == 'optimal':  tau_GtC = GAMMA * Y_ref * SCC_FACTOR
        else:                       tau_GtC = GAMMA * Y_ref * SCC_FACTOR / 3.0
        tau_e = tau_GtC * KAPPA * xi_   # per-fuel energy-unit tax

        # --- equilibrium oil price ---
        po_i  = _clear_oil(K_v[i], A_i, R_v[i], tau_e)

        # --- energy market ---
        p_hat = np.array([po_i + tau_e[0], P_COAL + tau_e[1], pg_ + tau_e[2]])
        P_i   = _P(p_hat); E_i = _E(K_v[i], A_i, P_i); e_i = _e(E_i, P_i, p_hat)
        Y_i   = A_i * K_v[i]**alpha * E_i**nu

        # refine tax with actual Y (one fixed-point iteration)
        if policy == 'optimal':   tau_GtC = GAMMA * Y_i * SCC_FACTOR
        elif policy == 'moderate':tau_GtC = GAMMA * Y_i * SCC_FACTOR / 3.0
        if tau_GtC > 0:
            tau_e = tau_GtC * KAPPA * xi_
            po_i  = _clear_oil(K_v[i], A_i, R_v[i], tau_e)
            p_hat = np.array([po_i+tau_e[0], P_COAL+tau_e[1], pg_+tau_e[2]])
            P_i   = _P(p_hat); E_i = _E(K_v[i], A_i, P_i); e_i = _e(E_i, P_i, p_hat)
            Y_i   = A_i * K_v[i]**alpha * E_i**nu

        # --- store ---
        Y_v[i]=Y_i; E_v[i]=E_i; e_v[i]=e_i; tau_v[i]=tau_GtC; po_v[i]=po_i

        # --- emissions (GtC/decade) ---
        M_i   = xi_ * np.dot(KAPPA, e_i)
        M_v[i]= M_i

        # --- state transitions ---
        if i < N-1:
            K_v[i+1]  = alpha * BETA * Y_i            # capital (log utility)
            R_v[i+1]  = max(R_v[i] - e_i[0], 0.0)    # oil reserves
            Sp_v[i+1] = Sp_v[i] + PHI_L * M_i        # permanent carbon
            St_v[i+1] = (1-D_T)*St_v[i] + (1-PHI_L)*M_i   # transient
            T_v[i+1]  = temp(Sp_v[i+1], St_v[i+1])

    return dict(years=YEARS, T=T_v, Y=Y_v, E=E_v, e=e_v,
                M=M_v, K=K_v, R=R_v, tau=tau_v, po=po_v,
                Sp=Sp_v, St=St_v)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PART 2(a): REPLICATE FIGURE 25.6
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Part 2(a): Replicating Figure 25.6")
print("="*60)

r_nt = simulate('no_tax')
r_op = simulate('optimal')
r_mo = simulate('moderate')

# summary table
print(f"\n{'Policy':<12} {'T₀':>6} {'T(2100)':>8} {'T(2200)':>8}  {'M₀ (GtC/dec)':>14}")
for r, label in [(r_nt,'No tax'),(r_op,'Optimal'),(r_mo,'Moderate')]:
    print(f"{label:<12} {r['T'][0]:>6.2f}  {r['T'][8]:>8.2f}   {r['T'][-1]:>8.2f}"
          f"  {r['M'][0]:>14.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.plot(YEARS, r_nt['T'],  'k-',  lw=2.5, label='No tax')
ax1.plot(YEARS, r_mo['T'],  'k-.', lw=2.5, label=r'Moderate tax ($\frac{1}{3}\tau^*$)')
ax1.plot(YEARS, r_op['T'],  'k--', lw=2.5, label=r'Optimal tax ($\tau^*$)')
ax1.axhline(2.0, color='gray', lw=1.0, ls=':', alpha=0.8)
ax1.text(2025, 2.12, '2 °C (Paris)', fontsize=9.5, color='gray')
ax1.set(xlabel='Year', ylabel='Global warming, °C',
        xlim=[2020,2200], ylim=[0,10])
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.25)
ax1.set_title('Replication of Figure 25.6 — Azzimonti et al. (2025), Ch. 25',
              fontsize=11)
fig1.tight_layout()
fig1.savefig(r'C:\Users\nb\Downloads\fig_25_6_replication.pdf',
             bbox_inches='tight', dpi=180)
print("Saved: fig_25_6_replication.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PART 2(b): SENSITIVITY 1 — SOCIAL DISCOUNT FACTOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Part 2(b): Sensitivity to social discount factor")
print("="*60)

# Three discount factors spanning the Stern–Nordhaus debate
# Converted from annual rates: beta = (1 - delta)^10
# Stern:     delta ≈ 0.001/yr  → beta ≈ 0.990
# Baseline:  delta ≈ 0.015/yr  → beta ≈ 0.860  (Golosov / Nordhaus)
# High disc: delta ≈ 0.045/yr  → beta ≈ 0.638  (Nordhaus "market" rate)
discount_scenarios = {
    'Stern  (δ=0.1%/yr, β=0.990)':   0.999**10,
    'Baseline (δ=1.5%/yr, β=0.860)': 0.985**10,
    'Nordhaus (δ=4.5%/yr, β=0.638)': 0.955**10,
}

# We need to re-run with modified BETA; update SCC_FACTOR inside the loop
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
styles = ['b-', 'k--', 'r-.']
T2100_dict = {}

for (label, beta_val), sty in zip(discount_scenarios.items(), styles):
    # Recompute SCC_FACTOR for this beta
    scc_f_b = PHI_L/(1-beta_val) + (1-PHI_L)/(1-beta_val*(1-D_T))

    # Temporarily override BETA and SCC_FACTOR in a local simulation
    def _sim_b(policy, bv, sccf):
        K_v  = np.zeros(N); Y_v  = np.zeros(N)
        Sp_v = np.zeros(N); St_v = np.zeros(N); T_v = np.zeros(N)
        R_v  = np.zeros(N); e_v  = np.zeros((N,3)); M_v = np.zeros(N)
        # Re-calibrate K_SS for this beta
        def _bgp(K):
            A0 = np.exp(-GAMMA * S_EX_0)
            E  = (nu*A0*K**alpha/P0_NO_TAX)**(1/(1-nu))
            Y  = A0*K**alpha*E**nu
            return K - alpha*bv*Y
        K0b = brentq(_bgp, 1e-8, 1e3)
        A0_ = np.exp(-GAMMA*S_EX_0)
        E0_ = (nu*A0_*K0b**alpha/P0_NO_TAX)**(1/(1-nu))
        e0_ = E0_*(P0_NO_TAX*LAM/np.array([P_OIL_0,P_COAL,P_GREEN]))**(1/(1-RHO))
        xi_ = M0_TARGET / np.dot(KAPPA, e0_)
        R0b = e0_[0]/(1-bv)

        K_v[0]=K0b; R_v[0]=R0b; Sp_v[0]=S_P0; St_v[0]=S_T0; T_v[0]=T0

        for i in range(N):
            S_ex = Sp_v[i]+St_v[i]
            A_i  = np.exp(G_DEC*i - GAMMA*S_ex)
            Y_ref= Y_v[i-1] if i>0 else 1.0
            if   policy=='no_tax':  tau_GtC=0.0
            elif policy=='optimal': tau_GtC=GAMMA*Y_ref*sccf
            else:                    tau_GtC=GAMMA*Y_ref*sccf/3.0
            tau_e = tau_GtC*KAPPA*xi_

            # simplified: use fixed oil price path for speed
            p_hat = np.array([P_OIL_0+tau_e[0], P_COAL+tau_e[1], P_GREEN+tau_e[2]])
            P_i   = ces_P(p_hat)
            E_i   = energy_demand(K_v[i], A_i, P_i)
            e_i   = energy_inputs(E_i, P_i, p_hat)
            Y_i   = A_i*K_v[i]**alpha*E_i**nu

            if policy=='optimal':   tau_GtC=GAMMA*Y_i*sccf
            elif policy=='moderate':tau_GtC=GAMMA*Y_i*sccf/3.0
            if tau_GtC>0:
                tau_e=tau_GtC*KAPPA*xi_
                p_hat=np.array([P_OIL_0+tau_e[0],P_COAL+tau_e[1],P_GREEN+tau_e[2]])
                P_i=ces_P(p_hat); E_i=energy_demand(K_v[i],A_i,P_i)
                e_i=energy_inputs(E_i,P_i,p_hat); Y_i=A_i*K_v[i]**alpha*E_i**nu

            Y_v[i]=Y_i; M_v[i]=xi_*np.dot(KAPPA,e_i); e_v[i]=e_i
            if i<N-1:
                K_v[i+1]=alpha*bv*Y_i
                R_v[i+1]=max(R_v[i]-e_i[0],0.)
                Sp_v[i+1]=Sp_v[i]+PHI_L*M_v[i]
                St_v[i+1]=(1-D_T)*St_v[i]+(1-PHI_L)*M_v[i]
                T_v[i+1]=temp(Sp_v[i+1],St_v[i+1])
        return T_v

    T_opt = _sim_b('optimal', beta_val, scc_f_b)
    T2100_dict[label] = T_opt[8]
    ax2.plot(YEARS, T_opt, sty, lw=2.2, label=label)

ax2.axhline(2.0, color='gray', lw=1.0, ls=':', alpha=0.8)
ax2.text(2025, 2.12, '2 °C (Paris)', fontsize=9.5, color='gray')
ax2.set(xlabel='Year', ylabel='Global warming, °C',
        xlim=[2020,2200], ylim=[0,5])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.25)
ax2.set_title('Sensitivity 1: Social Discount Factor (Optimal Tax Path)', fontsize=11)
fig2.tight_layout()
fig2.savefig(r'C:\Users\nb\Downloads\fig_sens1_discounting.pdf',
             bbox_inches='tight', dpi=180)

print("\n  T(2100) under optimal tax, by discount factor:")
for lab, t21 in T2100_dict.items():
    paris = "✓ ≤2°C" if t21<=2.0 else f"  {t21:.2f}°C"
    print(f"    {lab:<44}  {paris}")
print(f"\n[Savings: Figure saved as fig_sens1_discounting.pdf]")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PART 2(c): SENSITIVITY 2 — ELASTICITY OF SUBSTITUTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Part 2(c): Sensitivity to CES energy elasticity (rho)")
print("="*60)
# Benchmark: rho=0.5  (EoS=2)
# Low subst.: rho=0.0  (EoS=1, Cobb-Douglas)
# High subst.:rho=0.75 (EoS=4)

rho_cases = {'Low subst. (ρ=0, EoS=1)': 0.0,
             'Benchmark (ρ=0.5, EoS=2)': 0.5,
             'High subst. (ρ=0.75, EoS=4)': 0.75}

fig3, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
linestyles = ['k-', 'k-.', 'k--']

for ax, (rho_label, rho_v) in zip(axes, rho_cases.items()):
    for policy, ls, lbl in [('no_tax','k-','No tax'),
                              ('moderate','k-.','Moderate'),
                              ('optimal','k--','Optimal')]:
        r = simulate(policy, rho_val=rho_v)
        ax.plot(YEARS, r['T'], ls, lw=2.0, label=lbl)
    ax.axhline(2.0, color='gray', lw=1.0, ls=':', alpha=0.8)
    ax.set(xlabel='Year', xlim=[2020,2200], ylim=[0,10])
    ax.set_title(rho_label, fontsize=10.5)
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel('Global warming, °C')
axes[1].legend(fontsize=10, loc='upper left')
fig3.suptitle('Sensitivity 2: CES Elasticity of Substitution in Energy',
              fontsize=12, y=1.01)
fig3.tight_layout()
fig3.savefig(r'C:\Users\nb\Downloads\fig_sens2_elasticity.pdf',
             bbox_inches='tight', dpi=180)
print("Saved: fig_sens2_elasticity.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  PART 2(d)(ii): SENSITIVITY 3 — GROWTH DAMAGE (illustrative)
# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: A_t = exp(g*t - gamma*S_t)            [level damage only]
# With growth damage: effective growth rate g_eff = g*(1 - gamma_g*S_t)
# so A_t = exp(g*(1-gamma_g*S)*t - gamma*S_t)
# (see derivation discussion in write-up)

print("\n" + "="*60)
print("Part 2(d)(ii): Growth damage (illustrative)")
print("="*60)

gamma_g = 1e-4   # growth damage coefficient (illustrative; see discussion)

r_nt_lv  = simulate('no_tax')                          # level damage only
r_nt_gr  = simulate('no_tax',   gamma_growth=gamma_g)  # + growth damage
r_op_lv  = simulate('optimal')
r_op_gr  = simulate('optimal',  gamma_growth=gamma_g)

fig4, ax4 = plt.subplots(figsize=(9, 5.5))
ax4.plot(YEARS, r_nt_lv['T'], 'k-',  lw=2.0, label='No tax — level damage only')
ax4.plot(YEARS, r_nt_gr['T'], 'k:',  lw=2.0, label='No tax — level + growth damage')
ax4.plot(YEARS, r_op_lv['T'], 'b--', lw=2.0, label='Optimal — level damage only')
ax4.plot(YEARS, r_op_gr['T'], 'b:',  lw=2.0, label='Optimal — level + growth damage')
ax4.axhline(2.0, color='gray', lw=1.0, ls=':', alpha=0.7)
ax4.set(xlabel='Year', ylabel='Global warming, °C',
        xlim=[2020,2200], ylim=[0,10])
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.25)
ax4.set_title('Sensitivity 3: Level vs. Level+Growth Damages', fontsize=11)
fig4.tight_layout()
fig4.savefig(r'C:\Users\nb\Downloads\fig_sens3_growth_damage.pdf',
             bbox_inches='tight', dpi=180)
print("Saved: fig_sens3_growth_damage.pdf")

print("\n" + "="*60)
print("All figures generated.")
print("="*60)
