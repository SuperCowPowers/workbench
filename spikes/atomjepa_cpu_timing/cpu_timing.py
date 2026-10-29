"""CPU serving cost of a trained Atom-JEPA model, for sizing CPU and serverless endpoints.

Times the two halves of predict_fn on a sample of the model's FeatureSet at several thread
counts: conformer generation (RDKit) and the fold ensemble's encoder passes (torch). Encoder
time scales linearly with conformers per molecule. Local cores aren't c7i vCPUs, so the
numbers say which endpoint shapes are worth a real deploy, not what they will measure.

    python spikes/atomjepa_cpu_timing/cpu_timing.py --model cyp-reg-atomjepa-3a4-readout --molecules 50
"""

import argparse
import json
import os
import resource
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# Workbench Imports
from workbench.api import FeatureSet, Model
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.endpoints.atomjepa_utils import FoldModel, _conformers

CACHE = Path.home() / ".cache" / "workbench" / "atomjepa_timing"


def model_dir(model: Model) -> Path:
    """The model's artifacts, downloaded and extracted once."""
    target = CACHE / model.name
    if not (target / "hyperparameters.json").exists():
        bucket, key = model.model_data_url().removeprefix("s3://").split("/", 1)
        target.mkdir(parents=True, exist_ok=True)
        tar_path = target / "model.tar.gz"
        print(f"Downloading {model.model_data_url()}...")
        AWSAccountClamp().boto3_session.client("s3").download_file(bucket, key, str(tar_path))
        with tarfile.open(tar_path) as tar:
            tar.extractall(target, filter="data")
        tar_path.unlink()
    return target


def peak_rss_gb() -> float:
    # ru_maxrss is bytes on macOS, KiB on Linux
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="cyp-reg-atomjepa-3a4-readout")
    parser.add_argument("--molecules", type=int, default=50)
    parser.add_argument("--threads", default="1,2,4,8", help="comma-separated thread counts")
    args = parser.parse_args()

    model = Model(args.model)
    path = model_dir(model)
    with open(path / "hyperparameters.json", encoding="utf-8") as f:
        n_conformers = json.load(f)["n_conformers"]
    folds = [FoldModel.load(str(path / d), device="cpu") for d in sorted(os.listdir(path)) if d.startswith("fold_")]
    print(f"{len(folds)} folds loaded, peak RSS {peak_rss_gb():.2f} GB")

    smiles_column, *extra_features = model.features()
    df = FeatureSet(model.get_input()).pull_dataframe().sample(args.molecules, random_state=0)
    smiles = df[smiles_column].astype(str).tolist()

    # One-off setup cost (allocator, kernel selection) stays out of the timed runs
    warm = [c for c in (_conformers(s, 2) for s in smiles[:2]) if c is not None]
    for fold in folds:
        fold.predict(warm, df[extra_features].to_numpy(np.float64)[: len(warm)] if extra_features else None)

    print(f"\n{args.model}: {len(smiles)} molecules, {n_conformers} conformers each, {len(folds)} folds")
    print(f"{'threads':>7} {'conformers s/mol':>17} {'encoder s/mol':>14} {'total s/mol':>12} {'mols per 60 s':>14}")
    for threads in (int(t) for t in args.threads.split(",")):
        torch.set_num_threads(threads)
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            conformers = list(pool.map(_conformers, smiles, [n_conformers] * len(smiles)))
        conformer_time = time.perf_counter() - start

        ok = np.array([c is not None for c in conformers])
        valid = [c for c in conformers if c is not None]
        extra = df.loc[ok, extra_features].to_numpy(np.float64) if extra_features else None
        start = time.perf_counter()
        for fold in folds:
            fold.predict(valid, extra)
        encoder_time = time.perf_counter() - start

        per_conf, per_enc = conformer_time / len(smiles), encoder_time / ok.sum()
        total = per_conf + per_enc
        print(f"{threads:>7} {per_conf:>17.3f} {per_enc:>14.3f} {total:>12.3f} {60 / total:>14.0f}")
    print(f"\nPeak RSS {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
