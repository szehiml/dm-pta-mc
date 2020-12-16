"""
    Functions computing the signal shapes
"""

import numpy as np

import src.constants as const


def subtract_signal(t, signal, fit_params=3):
    """

    Returns the subtracted signal

    """

    # fit dphi(t) to polynomials and subtract the contribution from n=0, 1 and 2
    coef = np.polynomial.polynomial.polyfit(t, signal, fit_params - 1)  # (3)
    delta_signal = np.einsum(
        "n,nj->j", coef, np.asarray([np.power(t, n) for n in range(fit_params)])
    )  # (Nt)

    # compute the subtracted signal
    ht = signal - delta_signal  # (Nt), unit = s

    return ht


def dphi_dop_chunked(
    t,
    mass,
    r0_vec,
    v_vec,
    d_hat,
    use_form=False,
    conc=0,
    use_chunk=False,
    chunk_size=10000,
    verbose=False,
):
    """

    Compute dphi but in chunks over the subhalos, use when Nt x N is too large an array to
    store in memory

    """

    num_objects = len(mass)

    dphi = np.zeros(len(t))

    if use_chunk == True:

        if verbose:
            print("   Chunking data ... ")
            print()

        if num_objects % chunk_size == 0:
            num_chunks = num_objects // chunk_size
        else:
            num_chunks = num_objects // chunk_size + 1

        for i in range(num_chunks):

            mass_c = mass[i * chunk_size : (i + 1) * chunk_size]
            r0_c = r0_vec[i * chunk_size : (i + 1) * chunk_size]
            v_c = v_vec[i * chunk_size : (i + 1) * chunk_size]
            conc_c = conc[i * chunk_size : (i + 1) * chunk_size]

            dphi += dphi_dop(
                t, mass_c, r0_c, v_c, d_hat, use_form=use_form, conc=conc_c
            )
    else:

        dphi += dphi_dop(t, mass, r0_vec, v_vec, d_hat, use_form=use_form, conc=conc)

    return dphi


def dphi_dop_chunked_vec(
    t,
    mass,
    r0_vec,
    v_vec,
    use_form=False,
    conc=0,
    use_chunk=False,
    chunk_size=10000,
    verbose=False,
):
    """

    Compute dphi but in chunks over the subhalos, use when Nt x N is too large an array to
    store in memory

    """

    num_objects = len(mass)

    dphi_vec = np.zeros((len(t), 3))

    if use_chunk == True:

        if verbose:
            print("   Chunking data ... ")
            print()

        if num_objects % chunk_size == 0:
            num_chunks = num_objects // chunk_size
        else:
            num_chunks = num_objects // chunk_size + 1

        for i in range(num_chunks):

            mass_c = mass[i * chunk_size : (i + 1) * chunk_size]
            r0_c = r0_vec[i * chunk_size : (i + 1) * chunk_size]
            v_c = v_vec[i * chunk_size : (i + 1) * chunk_size]
            conc_c = conc[i * chunk_size : (i + 1) * chunk_size]

            dphi_vec += dphi_dop_vec(
                t, mass_c, r0_c, v_c, use_form=use_form, conc=conc_c
            )
    else:

        dphi_vec += dphi_dop_vec(t, mass, r0_vec, v_vec, use_form=use_form, conc=conc)

    return dphi_vec


def dphi_dop_vec(t, mass, r0_vec, v_vec, use_form=False, conc=0):
    """

    Returns the vector phase shift due to the Doppler delay for subhalos of mass, mass.
    Dot with d_hat to get dphi_I

    TODO: add use_closest option

    """

    v_mag = np.linalg.norm(v_vec, axis=1)

    r0_v = np.einsum("ij, ij -> i", r0_vec, v_vec)
    t0 = -r0_v / np.square(v_mag)  # year

    b_vec = r0_vec + v_vec * t0[:, np.newaxis]  # (N, 3)
    b_mag = np.linalg.norm(b_vec, axis=1)  # (N)
    tau = b_mag / v_mag

    b_hat = b_vec / b_mag[:, np.newaxis]  # (N, 3)
    v_hat = v_vec / v_mag[:, np.newaxis]

    x = np.subtract.outer(t, t0) / tau
    x0 = -t0 / tau

    bd_term = (np.sqrt(1 + x ** 2) + x) - (np.sqrt(1 + x0 ** 2) + x0)  # (Nt, N)
    vd_term = np.arcsinh(x) - np.arcsinh(x0)

    prefactor = (
        const.yr_to_s
        * const.GN
        * mass
        / (const.km_s_to_kpc_yr * const.c_light * np.square(v_mag))
    )

    if use_form:

        t_cl = np.maximum(np.minimum(t0, t[-1]), 0)
        x_cl = (t_cl - t0) / tau
        r_cl = tau * v_mag * np.sqrt(1 + x_cl ** 2)

        rv = ((3 * mass / (4 * np.pi)) * (1 / 200) * (1 / const.rho_crit)) ** (1 / 3)

        form_func = form(r_cl / rv, conc) * np.heaviside(rv - r_cl, 0) + np.heaviside(
            r_cl - rv, 0
        )  # (N)

        bd_term = prefactor * form_func * bd_term
        vd_term = prefactor * form_func * vd_term

    else:

        bd_term = prefactor * bd_term
        vd_term = prefactor * vd_term

    # sum the signal over all the events
    sig = np.einsum("to, oi -> ti", bd_term, b_hat) - np.einsum(
        "to, oi -> ti", vd_term, v_hat
    )

    return sig


def dphi_dop(t, mass, r0_vec, v_vec, d_hat, use_form=False, conc=0):
    """

    Returns the phase shift due to the Doppler delay for subhalos of mass, mass

    TODO: add use_closest option

    """

    v_mag = np.linalg.norm(v_vec, axis=1)

    r0_v = np.einsum("ij, ij -> i", r0_vec, v_vec)  # kpc^2/yr
    t0 = -r0_v / np.square(v_mag)  # year

    b_vec = r0_vec + v_vec * t0[:, np.newaxis]  # (N, 3), kpc
    b_mag = np.linalg.norm(b_vec, axis=1)  # (N)
    tau = b_mag / v_mag  # year

    b_hat = b_vec / b_mag[:, np.newaxis]
    v_hat = v_vec / v_mag[:, np.newaxis]

    b_d = np.dot(b_hat, d_hat)
    v_d = np.dot(v_hat, d_hat)

    x = np.subtract.outer(t, t0) / tau
    x0 = -t0 / tau

    bd_term = (np.sqrt(1 + x ** 2) + x) - (np.sqrt(1 + x0 ** 2) + x0)
    vd_term = np.arcsinh(x) - np.arcsinh(x0)

    sig = bd_term * b_d - vd_term * v_d

    prefactor = (
        const.yr_to_s
        * const.GN
        * mass
        / (const.km_s_to_kpc_yr * const.c_light * np.square(v_mag))
    )

    sig = prefactor * sig

    if use_form:

        t_cl = np.maximum(np.minimum(t0, t[-1]), 0)
        x_cl = (t_cl - t0) / tau
        r_cl = tau * v_mag * np.sqrt(1 + x_cl ** 2)

        rv = ((3 * mass / (4 * np.pi)) * (1 / 200) * (1 / const.rho_crit)) ** (1 / 3)

        form_func = form(r_cl / rv, conc) * np.heaviside(rv - r_cl, 0) + np.heaviside(
            r_cl - rv, 0
        )

        sig = form_func * sig

    # sum the signal over all the events
    return np.sum(sig, axis=-1)


def form(s, c):

    return (np.log(1 + c * s) - c * s / (1 + c * s)) / (np.log(1 + c) - c / (1 + c))
