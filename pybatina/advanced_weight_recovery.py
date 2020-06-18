from pybatina.utils import *

import numpy as np
import pandas as pd


class AdvancedWeightRecovery:
    MANTISSA_THREE_BYTES = [7, 8, 8]
    MAX_MANTISSA_NBITS = np.sum(MANTISSA_THREE_BYTES)
    NUMBER_OF_BEST_CANDIDATES = 10

    def __init__(self, guess_range):
        self.guess_range = guess_range

    @property
    def guess_range(self):
        return self.__guess_range

    @guess_range.setter
    def guess_range(self, value):
        self.__guess_range = value
        self.__input_value_set = self.build_input_value_set()

    @property
    def input_value_set(self):
        return self.__input_value_set

    @staticmethod
    def build_values(component, mantissa_byte_index):
        if component == 'mantissa':
            assert (mantissa_byte_index < len(AdvancedWeightRecovery.MANTISSA_THREE_BYTES))
            sum_nbits = np.sum(AdvancedWeightRecovery.MANTISSA_THREE_BYTES[:mantissa_byte_index + 1])
            retval = np.left_shift(np.arange(0, 1 << AdvancedWeightRecovery.MANTISSA_THREE_BYTES[mantissa_byte_index]),
                                   AdvancedWeightRecovery.MAX_MANTISSA_NBITS - sum_nbits) | np.left_shift(127,
                                                                                                          AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        elif component == 'exponent':
            retval = np.left_shift(np.arange(0, 1 << 8), AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        else:
            raise ValueError('the component is not supported')
        return retval

    @staticmethod
    def build_input_values(component, mantissa_byte_index=None):
        return np.vectorize(int_to_float)(AdvancedWeightRecovery.build_values(component, mantissa_byte_index))

    def build_input_values_inrange(self):
        return np.random.uniform(self.guess_range[0], self.guess_range[1], 1000)

    def build_input_value_set(self):
        input_value_set = [AdvancedWeightRecovery.build_input_values('mantissa', mantissa_byte_index=idx) for idx in range(len(AdvancedWeightRecovery.MANTISSA_THREE_BYTES))]
        input_value_set.append(AdvancedWeightRecovery.build_input_values('exponent'))
        input_value_set.append(self.build_input_values_inrange())
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

        # step 1: recover mantissa
        numbers = None
        for mantissa_byte_index in [0, 1, 2]:
            guess_numbers = AdvancedWeightRecovery.build_guess_values(component='mantissa', mantissa_byte_index=mantissa_byte_index, numbers=numbers)
            known_inputs = self.input_value_set[set_idx]
            secret_hw = secret_hamming_weight_set[set_idx]
            if len(secret_hw) != len(known_inputs):
                raise ValueError('#%d: size of secret_hw does not match size of known_inputs (%d, %d)' % (set_idx, len(secret_hw), len(known_inputs)))
            mantissa_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs, guess_numbers).sort_values(ascending=False)
            numbers = mantissa_corr.index[:AdvancedWeightRecovery.NUMBER_OF_BEST_CANDIDATES]
            set_idx = set_idx + 1

        # step 2: recover exponent
        guess_numbers = AdvancedWeightRecovery.build_guess_values(component='exponent')
        known_inputs = self.input_value_set[set_idx]
        secret_hw = secret_hamming_weight_set[set_idx]
        if len(secret_hw) != len(known_inputs):
            raise ValueError('#%d: size of secret_hw does not match size of known_inputs (%d, %d)' % (set_idx, len(secret_hw), len(known_inputs)))
        exponent_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs, guess_numbers).sort_values(ascending=False)
        set_idx = set_idx + 1

        # step 3: combine exponent and mantissa
        int_mantissa = np.vectorize(lambda x: float_to_int(x) & (~(0xff << AdvancedWeightRecovery.MAX_MANTISSA_NBITS)))(mantissa_corr.index[:AdvancedWeightRecovery.NUMBER_OF_BEST_CANDIDATES])
        int_exponent = np.vectorize(float_to_int)(exponent_corr.index[:AdvancedWeightRecovery.NUMBER_OF_BEST_CANDIDATES])
        guess_numbers = np.vectorize(int_to_float)((int_mantissa | int_exponent[:, np.newaxis]).reshape(-1))
        # now we work also with negative numbers
        guess_numbers = np.concatenate((guess_numbers, -guess_numbers))
        # eliminate the numbers that are out-range
        guess_numbers = guess_numbers[np.where(np.logical_and(guess_numbers >= self.guess_range[0], guess_numbers <= self.guess_range[1]))]

        # step 4: run the last check
        known_inputs = self.input_value_set[set_idx]
        secret_hw = secret_hamming_weight_set[set_idx]
        if len(secret_hw) != len(known_inputs):
            raise ValueError('#%d: size of secret_hw does not match size of known_inputs (%d, %d)' % (set_idx, len(secret_hw), len(known_inputs)))
        full_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs, guess_numbers).sort_values(ascending=False)

        return full_corr.iloc[:AdvancedWeightRecovery.NUMBER_OF_BEST_CANDIDATES]
