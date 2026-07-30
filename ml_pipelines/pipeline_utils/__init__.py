"""Shared code importable by any pipeline script.

Staged onto the Batch container's PYTHONPATH from S3 by the pipeline runner, and
onto a local run's PYTHONPATH by ml_pipeline_launcher, so `from pipeline_utils...`
resolves the same way in both.
"""
