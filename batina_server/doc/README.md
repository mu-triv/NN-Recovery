# Batina Server
The server provides the neural network weight discovery service

## API
1. Get input values: retrieve 1-dimension array of input values which must be used to acquire electro-magnetic traces for the targeted weight.
1. Get weight value: compute neural network targeted weight. The parameters are:
   1. EM trace
   1. precision, in integer which presents the number of mantissa bits to be recovered.
   1. number of candidates,
