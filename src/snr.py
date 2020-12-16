"""
    Contains functions which compute different SNRs
"""

import numpy as np

import src.constants as const


def opt_pulsar_snr(ht, trms_ns, cad_wk):
    """

    Computes the optimal pulsar SNR

    """

    trms = const.ns_to_s * trms_ns
    cad = const.week_to_s * cad_wk

    noise = np.square(trms) * cad

    snr_sq = (1 / noise) * cad * np.sum(np.square(ht))

    return np.sqrt(snr_sq)


def opt_earth_snr(ht_list, trms_ns, cad_wk):
    """

    Computes the optimal earth SNR

    """

    trms = const.ns_to_s * trms_ns
    cad = const.week_to_s * cad_wk

    noise = np.sqrt(2) * np.square(trms) * cad

    prefactor = cad / noise

    snr = prefactor * np.sqrt(
        (
            np.square(np.sum(np.square(ht_list)))
            - np.sum(np.square(np.sum(np.square(ht_list), axis=1)))
        )
    )

    return snr
