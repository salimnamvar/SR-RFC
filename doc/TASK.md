# Problem #
Understanding of how RNA molecules fold.

# Goal #
Prediction the reactivity (DMS, and 2A3) at every position for a particular RNA sequence.

# Model #
Regressor of two values:

- reactivity_DMS_MaP:
A float value from 0-1 measuring reactivity to the DMS chemical, with 1 being the most reactive and 0 being the least
reactive.
- reactivity_2A3_MaP:
A float value from 0-1 measuring reactivity to the 2A3 chemical, with 1 being the most reactive and 0 being the least 
reactive.
