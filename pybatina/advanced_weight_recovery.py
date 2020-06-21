from pybatina.utils import *

import numpy as np
import pandas as pd


class AdvancedWeightRecovery:
    MANTISSA_THREE_BYTES = [11, 8, 4]
    MAX_MANTISSA_NBITS = np.sum(MANTISSA_THREE_BYTES)
    NUMBER_OF_BEST_CANDIDATES = 10

    def __init__(self, guess_range, number_of_best_candidates=NUMBER_OF_BEST_CANDIDATES):
        self.guess_range = guess_range
        self.number_of_best_candidates = number_of_best_candidates
        self.input_value_set = AdvancedWeightRecovery.build_input_value_set()

    @property
    def guess_range(self):
        return self.__guess_range

    @guess_range.setter
    def guess_range(self, value):
        self.__guess_range = value

    @property
    def number_of_best_candidates(self):
        return self.__number_of_best_candidates

    @number_of_best_candidates.setter
    def number_of_best_candidates(self, value):
        self.__number_of_best_candidates = value

    @property
    def input_value_set(self):
        return self.__input_value_set

    @input_value_set.setter
    def input_value_set(self, value):
        self.__input_value_set = value

    @staticmethod
    def build_input_values(component, mantissa_byte_index=0):
        if component == 'mantissa':
            # this defines the number of mantissa bits of input values which are for generating HW
            n_msbbits = AdvancedWeightRecovery.MANTISSA_THREE_BYTES[mantissa_byte_index]
            m = np.left_shift(np.arange(0, 1 << n_msbbits), AdvancedWeightRecovery.MAX_MANTISSA_NBITS - n_msbbits)
            e = 127 << AdvancedWeightRecovery.MAX_MANTISSA_NBITS
            ivals = m | e
        elif component == 'exponent':
            ivals = np.left_shift(np.arange(0, 1 << 8), AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        else:
            raise ValueError('the component is not supported')

        fvals = np.vectorize(int_to_float)(ivals).astype(np.float32)
        return np.concatenate((fvals, -fvals))

    @staticmethod
    def build_values(component, mantissa_byte_index):
        if component == 'mantissa':
            assert (mantissa_byte_index < len(AdvancedWeightRecovery.MANTISSA_THREE_BYTES))
            sum_nbits = np.sum(AdvancedWeightRecovery.MANTISSA_THREE_BYTES[:mantissa_byte_index + 1])
            m = np.left_shift(np.arange(0, 1 << AdvancedWeightRecovery.MANTISSA_THREE_BYTES[mantissa_byte_index]), AdvancedWeightRecovery.MAX_MANTISSA_NBITS - sum_nbits)
            e = 127 << AdvancedWeightRecovery.MAX_MANTISSA_NBITS
            s = 1 << 31
            retval = np.concatenate((m | e | s, m | e))
        elif component == 'exponent':
            retval = np.left_shift(np.arange(0, 1 << 8), AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        else:
            raise ValueError('the component is not supported')
        return retval

    @staticmethod
    def build_input_value_set():
        input_value_set = [AdvancedWeightRecovery.build_input_values('mantissa', i) for i in range(len(AdvancedWeightRecovery.MANTISSA_THREE_BYTES))]
        input_value_set.append(AdvancedWeightRecovery.build_input_values('exponent'))
        return input_value_set

    @staticmethod
    def build_guess_values(component, mantissa_byte_index=None, numbers=None):
        values = AdvancedWeightRecovery.build_values(component, mantissa_byte_index)
        if numbers is not None:
            if component == 'mantissa':
                int_numbers = np.vectorize(float_to_int)(numbers)
            elif component == 'exponent':
                mask = ~(0xff << AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
                int_numbers = np.vectorize(lambda x: float_to_int(x) & mask)(numbers)
            else:
                raise ValueError('the component is not supported')
            values = np.unique((values | int_numbers[:, np.newaxis]).reshape(-1))
        return np.vectorize(int_to_float)(values)

    @staticmethod
    def compute_corr_numbers(secret_hw, known_inputs, guess_numbers):
        """
        compute the HW correlations of the weight_hw and the HW of the results of the multiplication
        between known_inputs and guess_numbers.
        This function should be used internally.
        :param secret_hw: hamming weight of the known_inputs with the secret value
        :param known_inputs: known input values
        :param guess_numbers: guess numbers
        :return: Pearson correlation of the hamming weights
        """
        hw = pd.DataFrame(columns=guess_numbers,
                          data=np.vectorize(hamming_weight)(known_inputs.reshape(-1, 1) * guess_numbers))
        return hw.corrwith(pd.Series(secret_hw), method='pearson')

    def recover_weight(self, secret_hamming_weight_set):
        if len(secret_hamming_weight_set) != len(self.input_value_set):
            raise ValueError('size of secret_hamming_weight_set does not match size of input_value_set (%d, %d)' % (len(secret_hamming_weight_set), len(self.input_value_set)))

        set_idx = 0

        # step 1: recover mantissa and sign
        numbers = None
        for mantissa_byte_index in range(len(AdvancedWeightRecovery.MANTISSA_THREE_BYTES)):
            guess_numbers = AdvancedWeightRecovery.build_guess_values(component='mantissa',
                                                                      mantissa_byte_index=mantissa_byte_index,
                                                                      numbers=numbers)
            known_inputs = self.input_value_set[set_idx]
            secret_hw = secret_hamming_weight_set[set_idx]
            if len(secret_hw) != len(known_inputs):
                raise ValueError('#%d: size of secret_hw does not match size of known_inputs (%d, %d)' % (set_idx, len(secret_hw), len(known_inputs)))
            mantissa_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs,guess_numbers).sort_values(ascending=False).iloc[:self.number_of_best_candidates]
            numbers = mantissa_corr.index
            set_idx = set_idx + 1

        # step 2: recover exponent
        guess_numbers = AdvancedWeightRecovery.build_guess_values(component='exponent', numbers=numbers)
        known_inputs = self.input_value_set[set_idx]
        secret_hw = secret_hamming_weight_set[set_idx]
        if len(secret_hw) != len(known_inputs):
            raise ValueError('#%d: size of secret_hw does not match size of known_inputs (%d, %d)' % (set_idx, len(secret_hw), len(known_inputs)))
        exponent_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs, guess_numbers).sort_values(ascending=False).iloc[:self.number_of_best_candidates]
        return exponent_corr
