"""Canonical fingerprint featurization config for the smiles-to-fingerprints endpoint.

Single source of truth shared by the inference script (which computes the
fingerprints) and the deploy script (which records these as the model's
hyperparameters, retrievable via ``Model(...).hyperparameters()``).

radius=2 (ECFP4): standard; radius 3 multiplies unique substructures and would
    worsen bit collisions without more bits.
n_bits=4096: count fingerprints are more collision-sensitive than binary — a
    collision sums two unrelated substructure counts rather than OR-ing a bit —
    so 4096 preserves count fidelity that folding to 2048 would corrupt.
counts=True: count fingerprints outperform binary for property prediction.

A different radius/bits/counts mix is a new endpoint version (-v2, ...), whose
own config makes it self-describing.
"""

FP_RADIUS = 2
FP_N_BITS = 4096
FP_COUNTS = True

# Recorded as the model's hyperparameters at deploy time.
FP_HYPERPARAMETERS = {"radius": FP_RADIUS, "n_bits": FP_N_BITS, "counts": FP_COUNTS}
