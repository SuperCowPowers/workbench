"""Canonical fingerprint featurization config for the smiles-to-fingerprints endpoint.

Shared by the inference script (computes the fingerprints) and the deploy script
(records these as the model's hyperparameters, via ``Model(...).hyperparameters()``).

radius=2 (ECFP4): substructures up to 2 bonds from each atom.
n_bits=4096: wide enough to limit count-corrupting bit collisions (a collision
    sums two unrelated substructure counts).
counts=True: count fingerprints suit property prediction.

A different radius/bits/counts mix is a new endpoint version, self-describing via
its own config.
"""

FP_RADIUS = 2
FP_N_BITS = 4096
FP_COUNTS = True

# Recorded as the model's hyperparameters at deploy time.
FP_HYPERPARAMETERS = {"radius": FP_RADIUS, "n_bits": FP_N_BITS, "counts": FP_COUNTS}
