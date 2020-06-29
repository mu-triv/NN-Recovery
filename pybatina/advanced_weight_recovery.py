from pybatina.utils import *

import numpy as np
import pandas as pd


class AdvancedWeightRecovery:
    MANTISSA_THREE_BYTES = [7, 8, 8]
    MAX_MANTISSA_NBITS = sum(MANTISSA_THREE_BYTES)
    NUMBER_OF_BEST_CANDIDATES = 30
    CORRELATION_STR = 'correlation'
    GUESS_VALUE_STR = 'guess_value'

    def __init__(self, guess_range, number_of_best_candidates=NUMBER_OF_BEST_CANDIDATES):
        self.guess_range = guess_range
        self.number_of_best_candidates = number_of_best_candidates
        self.input_value_set = AdvancedWeightRecovery.build_input_value_set()

    @property
    def guess_range(self):
        """
        retrieve the guess range
        :return: the guess range
        """
        return self.__guess_range

    @guess_range.setter
    def guess_range(self, value):
        """
        set a new guess range
        :param value: new guess range
        :return: None
        """
        self.__guess_range = value

    @property
    def number_of_best_candidates(self):
        """
        retrieve the number of best candidates
        :return: the number of best candidates
        """
        return self.__number_of_best_candidates

    @number_of_best_candidates.setter
    def number_of_best_candidates(self, value):
        """
        set a new number of best candidates
        :param value: new number of best candidates
        :return: None
        """
        self.__number_of_best_candidates = value

    @property
    def input_value_set(self):
        """
        retrieve input value set which must be used to get the hamming weight trace of the secret number
        :return: input value set
        """
        return self.__input_value_set

    @input_value_set.setter
    def input_value_set(self, value):
        """
        set a new input value set.
        :param value: a new input value set
        :return: None
        """
        self.__input_value_set = value

    @staticmethod
    def build_input_values(component):
        """
        build the input values for recovering the IEEE-754 floating point components
        :param component: 'mantissa' or 'exponent'
        :return: an array of input values
        """
        if component == 'mantissa':
            # this defines the number of mantissa bits of input values which are for generating HW
            n_msbbits = 10
            m = np.left_shift(np.arange(0, 1 << n_msbbits), AdvancedWeightRecovery.MAX_MANTISSA_NBITS - n_msbbits) | np.arange(0, 1 << n_msbbits)[::-1]
            e = 127 << AdvancedWeightRecovery.MAX_MANTISSA_NBITS
            ivals = m | e
        elif component == 'exponent':
            ivals = np.left_shift(np.arange(0, 1 << 8), AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        else:
            raise ValueError('the component is not supported')

        fvals = np.vectorize(int_to_float)(ivals).astype(np.float32)
        return np.concatenate((fvals, -fvals))

    @staticmethod
    def build_input_value_set():
        """
        build a set of input values that are used for recovering the secret number. The input set will be multiplied
        with the secret number to collect the hamming weight trace.
        :return: a set of input values
        """
        return [AdvancedWeightRecovery.build_input_values('mantissa'), AdvancedWeightRecovery.build_input_values('exponent')]

    @staticmethod
    def get_mask(component, mantissa_byte_index=None):
        if component == 'mantissa':
            try:
                retval = 0
                for idx in iter(mantissa_byte_index):
                    retval = retval | AdvancedWeightRecovery.get_mask(component, idx)
            except TypeError:
                nbits = sum(AdvancedWeightRecovery.MANTISSA_THREE_BYTES[:mantissa_byte_index + 1])
                retval = ((1 << AdvancedWeightRecovery.MANTISSA_THREE_BYTES[mantissa_byte_index]) - 1) << (
                            AdvancedWeightRecovery.MAX_MANTISSA_NBITS - nbits)
        elif component == 'exponent':
            retval = (0xff << AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        elif component == 'sign':
            retval = (1 << 31)
        return retval

    @staticmethod
    def group_candidates(correlations, component, mantissa_byte_index=0):
        df = correlations.reset_index().rename(
            columns={'index': AdvancedWeightRecovery.GUESS_VALUE_STR, 0: AdvancedWeightRecovery.CORRELATION_STR})

        if component == 'mantissa':
            if mantissa_byte_index == 0:
                nbits = 5
            else:
                nbits = sum(AdvancedWeightRecovery.MANTISSA_THREE_BYTES[:mantissa_byte_index])
            mask = ((1 << nbits) - 1) << (AdvancedWeightRecovery.MAX_MANTISSA_NBITS - nbits)
        elif component == 'exponent':
            mask = (0xff << AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        else:
            raise ValueError('the component is not supported')

        group_str = 'group'
        df[group_str] = np.vectorize(float_to_int)(df[AdvancedWeightRecovery.GUESS_VALUE_STR]) & mask
        return df.groupby(group_str)

    @staticmethod
    def build_values(component, mantissa_byte_index):
        """
        build the guess values for recovering the IEEE-754 floating point components
        :param component: 'mantissa' or 'exponent'
        :param mantissa_byte_index: the byte index of the mantissa component
        :return: an array of guess values
        """
        if component == 'mantissa':
            assert (mantissa_byte_index < len(AdvancedWeightRecovery.MANTISSA_THREE_BYTES))
            sum_nbits = np.sum(AdvancedWeightRecovery.MANTISSA_THREE_BYTES[:mantissa_byte_index + 1])
            m = np.left_shift(np.arange(0, 1 << AdvancedWeightRecovery.MANTISSA_THREE_BYTES[mantissa_byte_index]), AdvancedWeightRecovery.MAX_MANTISSA_NBITS - sum_nbits)
            e = 127 << AdvancedWeightRecovery.MAX_MANTISSA_NBITS
            retval = m | e
        elif component == 'exponent':
            retval = np.left_shift(np.arange(0, 1 << 8), AdvancedWeightRecovery.MAX_MANTISSA_NBITS)
        else:
            raise ValueError('the component is not supported')
        return retval

    @staticmethod
    def build_guess_values(component, mantissa_byte_index=None, numbers=None):
        """
        build the guess values for recovering the IEEE-754 floating point components
        :param component: 'mantissa' or 'exponent'
        :param mantissa_byte_index: the byte index of the mantissa component
        :param numbers: the numbers which will be combined to build a new guess values.
        :return: an array of guess values
        """
        values = AdvancedWeightRecovery.build_values(component, mantissa_byte_index)
        if numbers is not None:
            if component == 'mantissa':
                int_numbers = np.vectorize(float_to_int)(numbers)
            elif component == 'exponent':
                mask = AdvancedWeightRecovery.get_mask(component) ^ 0xffffffff
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
        number_of_subcorr = 3
        # sanity check
        if len(secret_hamming_weight_set) != len(self.input_value_set):
            raise ValueError('size of secret_hamming_weight_set does not match size of input_value_set (%d, %d)' % (len(secret_hamming_weight_set), len(self.input_value_set)))

        set_idx = 0
        # step 1: recover mantissa
        numbers = None
        known_inputs = self.input_value_set[set_idx]
        secret_hw = secret_hamming_weight_set[set_idx]
        # sanity check
        if len(secret_hw) != len(known_inputs):
            raise ValueError('#%d: size of secret_hw does not match size of known_inputs (%d, %d)' % (set_idx, len(secret_hw), len(known_inputs)))

        for mantissa_byte_index in range(len(AdvancedWeightRecovery.MANTISSA_THREE_BYTES)):
            if numbers is None:
                guess_numbers = AdvancedWeightRecovery.build_guess_values(component='mantissa',
                                                                          mantissa_byte_index=mantissa_byte_index,
                                                                          numbers=numbers)
                mantissa_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs,guess_numbers)
            else:
                mantissa_corr = pd.Series()
                for number in numbers:
                    guess_numbers = AdvancedWeightRecovery.build_guess_values(component='mantissa',
                                                                              mantissa_byte_index=mantissa_byte_index,
                                                                              numbers=np.asarray([number]))
                    # compute correlations of the guess values which are generated from the number
                    corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs, guess_numbers).sort_values(ascending=False)
                    # only keep the best values (we can keep more, but is this needed)
                    mantissa_corr = pd.concat([mantissa_corr, corr.iloc[:number_of_subcorr]])

            # get the best values
            numbers = np.asarray(mantissa_corr.sort_values(ascending=False).index[:self.number_of_best_candidates])
            #
            # Note: discovery of mantissa bits only uses one hamming weight trace
        # step 2: determine sign
        positive_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw=secret_hw, known_inputs=known_inputs, guess_numbers=numbers)
        negative_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw=secret_hw, known_inputs=known_inputs, guess_numbers=-numbers)
        if positive_corr.sum() < negative_corr.sum():
            numbers = -numbers

        set_idx = set_idx + 1

        # step 3: recover exponent
        known_inputs = self.input_value_set[set_idx]
        secret_hw = secret_hamming_weight_set[set_idx]
        # sanity check
        if len(secret_hw) != len(known_inputs):
            raise ValueError('#%d: size of secret_hw does not match size of known_inputs (%d, %d)' % (set_idx, len(secret_hw), len(known_inputs)))
        exponent_corr = pd.Series()
        for number in numbers:
            guess_numbers = AdvancedWeightRecovery.build_guess_values(component='exponent', numbers=np.asarray([number]))
            guess_numbers = guess_numbers[np.where(np.logical_and(guess_numbers >= self.guess_range[0], guess_numbers <= self.guess_range[1]))]
            corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hw, known_inputs, guess_numbers).sort_values(ascending=False).iloc[:self.number_of_best_candidates]
            exponent_corr = pd.concat([exponent_corr, corr.iloc[:number_of_subcorr]])
        # get the best values
        numbers = np.asarray(exponent_corr.sort_values(ascending=False).index[:self.number_of_best_candidates])

        # step 4: last sorting
        full_corr = AdvancedWeightRecovery.compute_corr_numbers(np.concatenate(secret_hamming_weight_set), np.concatenate(self.input_value_set), numbers)
        idx0_corr = AdvancedWeightRecovery.compute_corr_numbers(secret_hamming_weight_set[0], self.input_value_set[0], numbers)

        if full_corr.max() >= idx0_corr.max():
            last_corr = full_corr
        else:
            last_corr = idx0_corr

        return last_corr.sort_values(ascending=False)
