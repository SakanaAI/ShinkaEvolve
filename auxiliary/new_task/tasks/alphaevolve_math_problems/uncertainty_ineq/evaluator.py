# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the uncertainty inequality problem.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import sys, os, time, numpy as np
import sympy as sp

BENCHMARK = 0.3215872333529007

x = sp.symbols("x")


def _hermite_4k_polys(m: int):
    degrees = [4 * k for k in range(m)]
    Hs = [sp.polys.orthopolys.hermite_poly(n=d, x=x, polys=False) for d in degrees]
    return Hs, degrees


def _construct_P_with_forced_zero(coeffs: np.ndarray) -> sp.Expr:
    """
    Given m input coeffs (c0..c_{m-1}), build the Hermite combo
    c0*H0 + ... + c_{m-1}*H_{4(m-1)} + c_last*H_{4m}
    where c_last is chosen so that P(0) = 0.
    Also flip sign if limit at +inf is negative.
    """
    m = len(coeffs)
    Hs, _ = _hermite_4k_polys(m + 1)  # include the (m)-th term for solving P(0)=0
    rc = [sp.Rational(c) for c in coeffs]

    partial = sum(rc[i] * Hs[i] for i in range(m))
    a = Hs[m].subs(x, 0)
    b = -partial.subs(x, 0)
    c_last = sp.Rational(b) / sp.Rational(a)

    P = partial + c_last * Hs[m]

    # Ensure positivity at +inf (all degrees are multiples of 4, so sign is well-defined)
    if sp.limit(P, x, sp.oo) < 0:
        P = -P

    return sp.simplify(P)


def _largest_positive_root_of_P_over_x2(P: sp.Expr) -> float:
    # P is even and has P(0)=0; divide by x^2 and find the largest positive real root.
    gq = sp.exquo(P, x**2)  # exact division (should divide cleanly if multiplicity >= 2)
    roots = sp.real_roots(gq, x)
    if not roots:
        raise ValueError("No real roots for P(x)/x^2.")

    # Validate sign change around each candidate
    best = None
    for r in roots:
        r_approx = r.eval_rational(n=200)
        eps = sp.Rational(1, 10**198)
        left = gq.subs(x, r_approx - eps)
        right = gq.subs(x, r_approx + eps)
        if (left > 0 and right < 0) or (left < 0 and right > 0):
            if best is None or r_approx > best:
                best = r_approx

    if best is None:
        raise ValueError("No root with a verified sign change for P(x)/x^2.")
    return float(best)


def compute_c4_and_rmax(input_coeffs: np.ndarray):
    P = _construct_P_with_forced_zero(input_coeffs)
    # Quick sanity checks
    assert P.subs(x, 0) == 0, "P(0) != 0 after forcing."
    assert sp.limit(P, x, sp.oo) > 0, "Limit at +inf is not positive."

    rmax = _largest_positive_root_of_P_over_x2(P)
    c4 = (rmax**2) / (2.0 * np.pi)
    return c4, rmax


def verify_c4_solution_strict(
    user_coeffs: np.ndarray, reported_c4: float, reported_rmax: float, atol=1e-9, rtol=1e-9
):
    c4, rmax = compute_c4_and_rmax(np.asarray(user_coeffs, dtype=float))

    if not np.isclose(c4, reported_c4, rtol=rtol, atol=atol):
        raise ValueError(f"C4 mismatch: reported {reported_c4:.12f}, recomputed {c4:.12f}")

    if not np.isclose(rmax, reported_rmax, rtol=rtol, atol=atol):
        raise ValueError(f"r_max mismatch: reported {reported_rmax:.12f}, recomputed {rmax:.12f}")

    return c4, rmax


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)
            t0 = time.time()
            coeffs, c4_bound, r_max = program.run()
            t1 = time.time()
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        coeffs = np.asarray(coeffs, dtype=float)

        c4, rmax = verify_c4_solution_strict(coeffs, float(c4_bound), float(r_max))

        return {
            "c4_bound": float(c4),
            "combined_score": float(BENCHMARK / c4),
            "r_max": float(rmax),
            "coeffs": coeffs.tolist(),
            "eval_time": float(t1 - t0),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
