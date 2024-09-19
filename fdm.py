import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from math import sinh, asinh

def fdm(time_to_maturity, risk_free_rate, reversion_speed,
        long_average_variance, vol_of_vol, correlation):

    # Strike price K (1 as specified)
    K = 1.0

    # Grid parameters to generate S_0 and V_0
    S_max = 8 * K * (1 + 0.5 * vol_of_vol)
    V_max = 5.0 * (1 + 0.5 * long_average_variance)
    S_min = K * (1 - 0.5 * vol_of_vol)
    V_min = max(0.0, long_average_variance - 0.5 * vol_of_vol)

    T = time_to_maturity
    r_d = risk_free_rate  # Domestic interest rate
    rho = correlation
    sigma = vol_of_vol
    kappa = reversion_speed
    eta = long_average_variance

    # Number of price and variance steps
    ns = 50  # Number of price steps
    nv = 25  # Number of variance steps
    c = (S_max - S_min) / 10
    d = (V_max - V_min) / 10

    # Time discretization
    N = 20
    delta_t = T / N

    # Generate the non-uniform grid, which will give us S_0 and V_0
    Vec_s, Delta_s, Vec_v, Delta_v, X, Y = make_grid(ns, S_min, S_max, c, nv, V_min, V_max, d)

    # Continue with rest of the FDM setup
    A = make_matrices(ns, nv, rho, sigma, r_d, kappa, eta, Vec_s, Vec_v, Delta_s, Delta_v)
    B = make_boundaries(ns, nv, r_d, Vec_s)

    # Initial condition
    UU_0 = np.array([[max(Vec_s[i] - K, 0) for i in range(ns + 1)] for _ in range(nv + 1)])
    U_0 = UU_0.flatten()

    # Solve the PDE using Crank-Nicolson scheme
    n = (ns + 1) * (nv + 1)
    price_cn = cn_scheme(n=n, n_tm=N, u_0=U_0, delta_t=delta_t, l=A, b=B)
    price_cn = np.reshape(price_cn, (nv + 1, ns + 1))

    # Compute moneyness and V0_values
    moneyness = X / K  # X is the meshgrid of Vec_s
    V0_values = Y  # Y is the meshgrid of Vec_v

    # Flatten the arrays
    moneyness = moneyness.flatten()
    V0_values = V0_values.flatten()
    option_prices = price_cn.flatten()

    return option_prices, moneyness, V0_values  # Return arrays

def make_grid(ns, s_min, s_max, c, nv, v_min, v_max, d):
    delta_zeta_i = (1.0 / ns) * (asinh((s_max - s_min) / c))
    zeta_s = [asinh((s_min - s_min) / c) + i * delta_zeta_i for i in range(ns + 1)]
    vec_s = [g(zeta_s[i], s_min, c) for i in range(ns + 1)]

    delta_eta = (1.0 / nv) * (asinh((v_max - v_min) / d))
    eta_v = [asinh((v_min - v_min) / d) + i * delta_eta for i in range(nv + 1)]
    vec_v = [h(eta_v[i], v_min, d) for i in range(nv + 1)]

    x, y = np.meshgrid(vec_s, vec_v)

    # Compute deltas
    delta_s = [vec_s[i + 1] - vec_s[i] for i in range(len(vec_s) - 1)]
    delta_v = [vec_v[i + 1] - vec_v[i] for i in range(len(vec_v) - 1)]

    return np.array(vec_s), delta_s, np.array(vec_v), delta_v, x, y

def g(xi, s_min, c):
    return s_min + c * sinh(xi)

def h(xi, v_min, d):
    return v_min + d * sinh(xi)

# Central approximation of derivatives, inner points
def central_coefficients1(i, index, delta):
    if index == -1:
        return -delta[i + 1] / (delta[i] * (delta[i] + delta[i + 1]))
    elif index == 0:
        return (delta[i + 1] - delta[i]) / (delta[i] * delta[i + 1])
    elif index == 1:
        return delta[i] / (delta[i + 1] * (delta[i] + delta[i + 1]))

def central_coefficients2(i, index, delta):
    if index == -1:
        return 2 / (delta[i] * (delta[i] + delta[i + 1]))
    elif index == 0:
        return -2 / (delta[i] * delta[i + 1])
    elif index == 1:
        return 2 / (delta[i + 1] * (delta[i] + delta[i + 1]))

# Backward approximation of derivatives, right border
def backward_coefficients(i, index, delta):
    if index == -2:
        return delta[i] / (delta[i - 1] * (delta[i] + delta[i - 1]))
    elif index == -1:
        return (-delta[i] - delta[i - 1]) / (delta[i] * delta[i - 1])
    elif index == 0:
        return (2 * delta[i] + delta[i - 1]) / (delta[i] * (delta[i] + delta[i - 1]))

# Forward approximation of derivatives, left border
def forward_coefficients(i, index, delta):
    if index == 0:
        return (-2 * delta[i + 1] - delta[i + 2]) / (delta[i + 1] * (delta[i + 1] + delta[i + 2]))
    elif index == 1:
        return (delta[i + 1] + delta[i + 2]) / (delta[i + 1] * delta[i + 2])
    elif index == 2:
        return -delta[i + 1] / (delta[i + 2] * (delta[i + 1] + delta[i + 2]))

def make_matrices(ns, nv, rho, sigma, r_d, kappa, eta, vec_s, vec_v, delta_s, delta_v):
    n = (ns + 1) * (nv + 1)
    a_0 = csc_matrix((n, n))
    a_1 = csc_matrix((n, n))
    a_2 = csc_matrix((n, n))

    # Definition of a_0 (cross derivative terms)
    for j in range(1, nv):
        for i in range(1, ns):
            c = rho * sigma * vec_s[i] * vec_v[j]
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    idx_i_j = i + j * (ns + 1)
                    idx_k_l = (i + k) + (j + l) * (ns + 1)
                    if 0 <= i + k <= ns and 0 <= j + l <= nv:
                        coeff = c * central_coefficients1(i - 1, k, delta_s) * central_coefficients1(j - 1, l, delta_v)
                        a_0[idx_i_j, idx_k_l] += coeff

    # Definition of a_1 (price derivative terms)
    for j in range(nv + 1):
        for i in range(1, ns):
            b = r_d * vec_s[i]
            c = 0.5 * vec_s[i] ** 2 * vec_v[j]
            idx_i_j = i + j * (ns + 1)
            for k in [-1, 0, 1]:
                if 0 <= i + k <= ns:
                    idx_k_j = (i + k) + j * (ns + 1)
                    coeff = c * central_coefficients2(i - 1, k, delta_s) + b * central_coefficients1(i - 1, k, delta_s)
                    a_1[idx_i_j, idx_k_j] += coeff
            a_1[idx_i_j, idx_i_j] += -0.5 * r_d

    # Definition of a_2 (variance derivative terms)
    for j in range(1, nv):
        for i in range(ns + 1):
            d = kappa * (eta - vec_v[j])
            e = 0.5 * sigma ** 2 * vec_v[j]
            idx_i_j = i + j * (ns + 1)
            for k in [-1, 0, 1]:
                if 0 <= j + k <= nv:
                    idx_i_jk = i + (j + k) * (ns + 1)
                    coeff = d * central_coefficients1(j - 1, k, delta_v) + e * central_coefficients2(j - 1, k, delta_v)
                    a_2[idx_i_j, idx_i_jk] += coeff
            a_2[idx_i_j, idx_i_j] += -0.5 * r_d

    a = a_0 + a_1 + a_2

    return a

def make_boundaries(ns, nv, r_d, vec_s):
    n = (ns + 1) * (nv + 1)
    b = np.zeros(n)

    # Boundary when s = s_max
    for j in range(nv + 1):
        idx = ns + j * (ns + 1)
        b[idx] = r_d * vec_s[-1]

    # Boundary when v = v_max
    for i in range(ns + 1):
        idx = i + nv * (ns + 1)
        b[idx] += -0.5 * r_d * vec_s[i]

    return b

def cn_scheme(n, n_tm, u_0, delta_t, l, b):
    u = u_0
    identity = csc_matrix(np.identity(n))
    lhs = identity - 0.5 * delta_t * l
    rhs_constant = 0.5 * delta_t * b

    for _ in range(n_tm):
        rhs = (identity + 0.5 * delta_t * l).dot(u) + rhs_constant
        u = spsolve(lhs, rhs)
        u = np.maximum(u, 0)  # Enforce non-negativity

    return u


