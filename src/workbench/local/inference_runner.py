"""Run a model directory's inference script the way the serving container does.

The local Endpoint calls these in-process. When the model can't be loaded into the calling
process, the Endpoint runs this module as ``python -m workbench.local.inference_runner
<model_dir> <input.pkl> <output.pkl>`` instead, so the same code serves it from a fresh
process.
"""

import importlib.util
import json
import os
import sys

import pandas as pd


def inference_script(model_dir: str) -> str:
    """Path to the inference script a trained model directory serves with.

    Args:
        model_dir (str): The model artifacts directory

    Returns:
        str: Path to the inference script

    Raises:
        FileNotFoundError: If the directory holds no trained model
    """
    metadata_path = os.path.join(model_dir, "inference-metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"No model artifacts at {model_dir}. The model was deleted or never finished training.")
    with open(metadata_path, encoding="utf-8") as fp:
        return os.path.join(model_dir, json.load(fp)["inference_script"])


def load_inference_module(model_dir: str):
    """Import a model directory's inference script.

    Args:
        model_dir (str): The model artifacts directory

    Returns:
        module: The imported inference module
    """
    script = inference_script(model_dir)

    # Model scripts import modules shipped beside them; the container serves with the
    # model directory on the import path, so local serving does too
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location("workbench_local_inference", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(module, bundle, model_dir: str, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Call the script's predict_fn with SM_MODEL_DIR set, as the container sets it.

    Args:
        module (module): The imported inference module
        bundle: Whatever the module's model_fn returned
        model_dir (str): The model artifacts directory
        eval_df (pd.DataFrame): The data to run inference on

    Returns:
        pd.DataFrame: The predictions
    """
    previous = os.environ.get("SM_MODEL_DIR")
    os.environ["SM_MODEL_DIR"] = model_dir
    try:
        return module.predict_fn(eval_df.copy(), bundle)
    finally:
        if previous is None:
            os.environ.pop("SM_MODEL_DIR", None)
        else:
            os.environ["SM_MODEL_DIR"] = previous


if __name__ == "__main__":
    model_dir, input_path, output_path = sys.argv[1:4]
    module = load_inference_module(model_dir)
    predictions = predict(module, module.model_fn(model_dir), model_dir, pd.read_pickle(input_path))
    predictions.to_pickle(output_path)
