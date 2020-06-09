# Python Batina
pybatina contains the python implementation of Batina et al. paper presented in USENIX 2019.

## Functions
```python
weight_recovery.build_input_values(component='mantissa',
                                   mantissa_nbits=10)
``` 
The function builds the list of input values which is used to evaluate the weight value.
The user needs to get this input values for attacking the targeted neuron weight. The
weight attack needs three steps: recover mantissa, recover exponent and then recover sign.

---

```python
weight_recovery.recover_weight(secret_hamming_weight_set,
                               input_value_set,
                               guess_range=None,
                               mantissa_nbits=10,
                               max_number_of_best_candidates=10)

```
The function recovers the weight value.
