import numpy as np
import pandas as pd
import struct
from pybatina.utils import *

import numpy as np
import pandas as pd

max_mantissa_nbits = 23


def build_guess_values(component='mantissa', numbers=None, mantissa_nbits=10, guess_range=None):
    """
    Build the list of guess values which is used to evaluate the weight value.
    This function should be used internally.
    :param component: IEEE 754 component name, it must be "mentissa", "exponent", "sign"
    :param numbers: the list of numbers of the previous state
    :param mantissa_nbits: number of mantissa bits to be recovered
    :param guess_range: the guess range
    :return: the list of guess numbers
    """
    if component == 'mantissa':
        # set the exponent 1
        e = (0x7f << max_mantissa_nbits)
        guess_numbers = np.vectorize(int_to_float)(
            np.left_shift(np.arange(0, 1 << mantissa_nbits), max_mantissa_nbits - mantissa_nbits) | e)
    elif component == 'exponent':
        # remove the exponent bits in mantissa
        m = np.vectorize(lambda x: x & ~(0xff << max_mantissa_nbits))(np.vectorize(float_to_int)(numbers))
        # set all possible exponent bits
        e = np.left_shift(np.arange(0, 1 << 8), max_mantissa_nbits)
        y = np.vectorize(int_to_float)(m | e[:, np.newaxis]).reshape(-1)
        # because we do not take in account the sign, the value is in positive
        # so we need to change the low and high of the guess range
        if None is guess_range:
            guess_numbers = y
        else:
            hi_range = max(np.abs(guess_range))
            lo_range = max(np.min(guess_range), 0.0)
            guess_numbers = y[(lo_range <= y) & (y <= hi_range)]
    elif component == 'sign':
        y = np.concatenate((np.asarray(numbers), -np.asarray(numbers)))
        if None is guess_range:
            guess_numbers = y
        else:
            guess_numbers = y[(guess_range[0] <= y) & (y <= guess_range[1])]
    else:
        raise ValueError('the component is not supported')
    return guess_numbers


def build_input_values(component='mantissa',  mantissa_nbits=10):
    """
    build the list of input values which is used to evaluate the weight value. The user needs
    to get this input values for attacking the targeted neuron weight. The weight attack needs
    three steps: recover mantissa, recover exponent and then recover sign.
    :param component: IEEE 754 component name, it must be "mentissa", "exponent", "sign"
    :param mantissa_nbits: number of mantissa bits to be recovered
    :return: the list of input numbers
    """
    if component == 'mantissa':
        retval = np.vectorize(int_to_float)(
            np.left_shift(np.arange(0, 1 << mantissa_nbits), max_mantissa_nbits - mantissa_nbits))
    elif component == 'exponent':
        retval = np.vectorize(int_to_float)(np.left_shift(np.arange(0, 1 << 8), max_mantissa_nbits))
    elif component == 'sign':
        pos_nums = np.vectorize(int_to_float)(np.left_shift(np.arange(0, 1 << 8), max_mantissa_nbits))
        retval = np.concatenate((pos_nums, -pos_nums))
    else:
        raise ValueError('the component is not supported')
    return retval


def compute_corr_numbers(weight_hw, known_inputs, guess_numbers):
    """
    compute the HW correlations of the weight_hw and the HW of the results of the multiplication
    between known_inputs and guess_numbers.
    This function should be used internally.
    :param weight_hw: hamming weight of the known_inputs with the secret value
    :param known_inputs: known input values
    :param guess_numbers: guess numbers
    :return: Pearson correlation of the hamming weights
    """
    hw = pd.DataFrame(columns=guess_numbers,
                      data=np.vectorize(hamming_weight)(known_inputs.reshape(-1, 1) * guess_numbers))
    return hw.corrwith(pd.Series(weight_hw), method='pearson')


def recover_weight(secret_hamming_weight_set,
                   input_value_set,
                   guess_range=None,
                   mantissa_nbits=10,
                   max_number_of_best_candidates=10):
    """
    recover the weight value.
    :param secret_hamming_weight_set: a tuple of 3 hamming weights: hamming weights of mantissa, hamming weights of exponent and hamming weights of sign
    :param input_value_set: a tuple of 3 input values: input values of mantissa, input values of exponent and input values of sign
    :param guess_range: the guess range which is used to reduce the search space
    :param mantissa_nbits: the number of interested bits of mantissa, so called precision
    :param max_number_of_best_candidates: the maximum number output candidates
    :return: the output candidates of the neural network weight
    """
    if len(input_value_set) != 3:
        raise TypeError('invalid input_value_set, length must be 3')
    if len(secret_hamming_weight_set) != 3:
        raise TypeError('invalid secret_hamming_weight_set, length must be 3')
    if (guess_range is not None) and (guess_range[0] >= guess_range[1]):
        raise ValueError('invalid guess_range, lower must be smaller than higher %s' % str(guess_range))

    # step 1: guess the mantissa 10 bits
    hw_idx = 0
    known_inputs = input_value_set[hw_idx]
    weight_hw = secret_hamming_weight_set[hw_idx]
    guess_numbers = build_guess_values(component='mantissa', mantissa_nbits=mantissa_nbits, guess_range=guess_range)
    mantisa_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)

    # step 2: guess the exponent 8 bits
    hw_idx = 1
    known_inputs = input_value_set[hw_idx]
    weight_hw = secret_hamming_weight_set[hw_idx]
    guess_numbers = build_guess_values(component='exponent',
                                       numbers=mantisa_corr.sort_values(ascending=False).index[:max_number_of_best_candidates],
                                       guess_range=guess_range)
    mantisa_exp_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)

    # step 3: guess the sign 1 bit
    hw_idx = 2
    known_inputs = input_value_set[hw_idx]
    weight_hw = secret_hamming_weight_set[hw_idx]
    guess_numbers = build_guess_values(component='sign',
                                       numbers=mantisa_exp_corr.sort_values(ascending=False).index[:max_number_of_best_candidates],
                                       guess_range=guess_range)
    full_number_corr = compute_corr_numbers(weight_hw, known_inputs, guess_numbers)
    return full_number_corr.sort_values(ascending=False).iloc[:max_number_of_best_candidates]

