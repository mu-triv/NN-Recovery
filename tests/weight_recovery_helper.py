import sys

sys.path.append("../")

from batina.weight_recovery import *
from batina.utils import *
import numpy as np


def batina_recover_weight(secret_number, guess_range, mantissa_nbits=10, max_number_of_best_candidates=10, noise=None):
    """
    recover the weight value (secret_number)
    :param secret_number:
    :param guess_range:
    :param mantissa_nbits:
    :param max_number_of_best_candidates:
    :param noise: a tuple of (add_noise_function, signal_to_noise, frequency). The prototype of add_noise_function is,  func(signal, signal_to_noise, frequency)
    :return: the 10 values which have highest HW correlations
    """
    if noise is not None:
        add_noise_function, signal_to_noise, frequency = noise
    # step 1: build signal for mantissa
    mantissa_known_inputs = build_input_values(mantissa_nbits=mantissa_nbits, component='mantissa')
    mantissa_weight_hw = np.vectorize(hamming_weight)(mantissa_known_inputs * secret_number)
    if noise is not None:
        mantissa_weight_hw = add_noise_function(mantissa_weight_hw, signal_to_noise, frequency)

    # step 2: build signal for exponent 8 bits
    exponent_known_inputs = build_input_values(component='exponent')
    exponent_weight_hw = np.vectorize(hamming_weight)(exponent_known_inputs * secret_number)
    if noise is not None:
        exponent_weight_hw = add_noise_function(exponent_weight_hw, signal_to_noise, frequency)
    # step 3: build signal for sign 1 bit
    sign_known_inputs = build_input_values(component='sign')
    sign_weight_hw = np.vectorize(hamming_weight)(sign_known_inputs * secret_number)
    if noise is not None:
        sign_weight_hw = add_noise_function(sign_weight_hw, signal_to_noise, frequency)

    return recover_weight(secret_hamming_weight_set=(mantissa_weight_hw, exponent_weight_hw, sign_weight_hw),
                          input_value_set=(mantissa_known_inputs, exponent_known_inputs, sign_known_inputs),
                          guess_range=guess_range,
                          mantissa_nbits=mantissa_nbits,
                          max_number_of_best_candidates=max_number_of_best_candidates)
