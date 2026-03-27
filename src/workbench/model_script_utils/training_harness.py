"""Training harness for SageMaker V3.

Wraps model training scripts to handle pre/post-training steps that V3's
ModelTrainer doesn't handle natively. V3 handles source code upload and
extraction; this harness handles:

1. Install any extra requirements (requirements.txt) if present
2. Run the model training script
3. Bundle the source code + inference metadata into the model directory

Usage (via ModelTrainer command= parameter):
    python training_harness.py generated_model_script.py
"""

import os
import sys
import shutil
import json
import subprocess


def include_code_and_meta_for_inference(model_dir, code_dir, entry_point):
    """Bundle inference code and metadata into the model directory.

    This ensures the inference container knows which script to run and has
    all the supporting code files available.
    """
    print(f"[Harness] Bundling inference code and metadata into {model_dir}...")

    # Write inference metadata (tells the inference container which script to run)
    metadata = {"inference_script": entry_point}
    metadata_path = os.path.join(model_dir, "inference-metadata.json")
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp)
    print(f"[Harness] Wrote inference metadata: {metadata}")

    # Copy all code files into model directory (except __pycache__)
    for item in os.listdir(code_dir):
        if item == "__pycache__":
            continue
        src = os.path.join(code_dir, item)
        dst = os.path.join(model_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"[Harness] Copied: {item}")


def install_requirements(code_dir):
    """Install extra Python dependencies from requirements.txt if present."""
    requirements_path = os.path.join(code_dir, "requirements.txt")
    if not os.path.exists(requirements_path):
        print("[Harness] No requirements.txt found, skipping.")
        return

    print(f"[Harness] Installing dependencies from {requirements_path}...")
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        # GPU containers on Ubuntu use system Python (no venv) and need this flag
        if os.environ.get("BREAK_SYSTEM_PACKAGES"):
            cmd.append("--break-system-packages")
        cmd.extend(["-r", requirements_path])
        subprocess.check_call(cmd)
        print("[Harness] Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[Harness] ERROR installing requirements: {e}")
        sys.exit(1)


def main():
    # The model script name is passed as the first argument
    if len(sys.argv) < 2:
        print("[Harness] ERROR: Usage: python training_harness.py <model_script.py>")
        sys.exit(1)

    training_script = sys.argv[1]

    # V3 sets SM_SOURCE_DIR to the extracted source code directory
    code_dir = os.environ.get("SM_SOURCE_DIR", os.path.dirname(os.path.abspath(__file__)))
    training_script_path = os.path.join(code_dir, training_script)

    print("[Harness] Training harness started")
    print(f"[Harness] Code directory: {code_dir}")
    print(f"[Harness] Training script: {training_script}")
    print(f"[Harness] Contents: {os.listdir(code_dir)}")

    if not os.path.exists(training_script_path):
        print(f"[Harness] ERROR: Training script not found: {training_script_path}")
        sys.exit(1)

    # Step 1: Install any extra requirements before training
    install_requirements(code_dir)

    # Step 2: Run the model training script
    print(f"[Harness] Running training script: {training_script_path}")
    try:
        subprocess.check_call(
            [
                sys.executable,
                training_script_path,
                "--model-dir",
                os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
                "--output-data-dir",
                os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
                "--train",
                os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
            ]
        )
        print("[Harness] Training script completed successfully.")

        # Step 3: Bundle code and metadata for inference
        include_code_and_meta_for_inference(
            model_dir=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
            code_dir=code_dir,
            entry_point=training_script,
        )

        print("[Harness] Training harness complete.")

    except subprocess.CalledProcessError as e:
        print(f"[Harness] ERROR: Training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
