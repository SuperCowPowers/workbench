"""Endpoint: In-process inference against a locally trained model."""

import glob
import os
import subprocess
import sys
import tempfile
from typing import Union

import pandas as pd

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.local import inference_runner
from workbench.local.local_artifact import LocalArtifact
from workbench.local.model import Model
from workbench.local import storage


class Endpoint(LocalArtifact):
    """Endpoint: Workbench Local Endpoint Class

    Loads the model bundle once with `model_fn` and calls `predict_fn` per
    inference, which are the same functions the serving container calls. An
    XGBoost model is served from a child process instead when torch is already
    imported here, since the two OpenMP runtimes cannot share a process.

    Common Usage:
        ```python
        my_endpoint = Endpoint("my-model")
        results = my_endpoint.inference(eval_df)
        ```
    """

    artifact_type = "endpoint"

    def __init__(self, name: str, **kwargs):
        """Initialize a Endpoint

        Args:
            name (str): The name of the endpoint
        """
        Artifact.is_name_valid(name, delimiter="-", lower_case=False)
        super().__init__(name, **kwargs)
        self.inference_dir = os.path.join(self.path, "inference")
        self._model_bundle = None
        self._inference_module = None

    @classmethod
    def from_model(cls, model: Model, name: str = None) -> "Endpoint":
        """Create a Endpoint that serves a Model.

        Args:
            model (Model): The trained model to serve
            name (str, optional): Endpoint name (defaults to the model name)

        Returns:
            Endpoint: The created endpoint

        Raises:
            ValueError: If the model hasn't trained successfully
        """
        if model.training_state().get("state") != "completed":
            raise ValueError(f"Model {model.name} has not trained successfully, cannot serve it")

        endpoint = cls(name or model.name)
        storage.local_root(create=True)
        os.makedirs(endpoint.inference_dir, exist_ok=True)
        endpoint._init_storage(input_name=model.name)
        endpoint.upsert_workbench_meta({"model_name": model.name})
        endpoint.log.important(f"Local endpoint {endpoint.name} serving {model.name}")
        return endpoint

    @property
    def model_name(self) -> str:
        """The model this endpoint serves"""
        return self.workbench_meta().get("model_name")

    @property
    def model_dir(self) -> str:
        """The model artifacts directory this endpoint loads from.

        Resolved from the model name on every access rather than stored, so the
        endpoint keeps working when the storage root moves or the config changes.
        """
        return Model(self.model_name).model_dir

    def inference(self, eval_df: pd.DataFrame, capture_name: str = None) -> pd.DataFrame:
        """Run inference on a DataFrame.

        Args:
            eval_df (pd.DataFrame): The data to run inference on
            capture_name (str, optional): Store the predictions under this name, which
                makes them available from the model as an inference run

        Returns:
            pd.DataFrame: The predictions
        """
        if self._needs_fresh_process():
            predictions = self._inference_in_fresh_process(eval_df)
        else:
            module = self._load_inference_module()
            predictions = inference_runner.predict(module, self._load_model_bundle(), self.model_dir, eval_df)

        if capture_name:
            self._capture(predictions, capture_name)
        return predictions

    def _capture(self, predictions: pd.DataFrame, capture_name: str):
        """Internal: Store an inference run under a capture name

        Args:
            predictions (pd.DataFrame): The predictions to store
            capture_name (str): The name to store them under
        """
        capture_dir = os.path.join(self.inference_dir, capture_name)
        os.makedirs(capture_dir, exist_ok=True)
        predictions.to_parquet(os.path.join(capture_dir, "predictions.parquet"), index=False)
        self.log.important(f"Captured {len(predictions)} predictions as '{capture_name}'")

    def get_inference_predictions(self, capture_name: str = "auto_inference") -> Union[pd.DataFrame, None]:
        """Retrieve a captured inference run.

        Args:
            capture_name (str): The capture to retrieve (default: "auto_inference")

        Returns:
            Union[pd.DataFrame, None]: The predictions, or None if that capture doesn't exist
        """
        path = os.path.join(self.inference_dir, capture_name, "predictions.parquet")
        if not os.path.isfile(path):
            self.log.warning(f"No inference capture named '{capture_name}' for {self.name}")
            return None
        return pd.read_parquet(path)

    def list_captures(self) -> list[str]:
        """The inference captures stored on this endpoint.

        Returns:
            list[str]: Sorted capture names
        """
        if not os.path.isdir(self.inference_dir):
            return []
        return sorted(d.name for d in os.scandir(self.inference_dir) if d.is_dir())

    def _load_inference_module(self):
        """Internal: Import the model's inference script, the way the container does.

        Returns:
            module: The imported inference module
        """
        if self._inference_module is None:
            self._inference_module = inference_runner.load_inference_module(self.model_dir)
        return self._inference_module

    def _load_model_bundle(self) -> dict:
        """Internal: Load the model once via model_fn, then keep it.

        Returns:
            dict: Whatever model_fn returns for this framework
        """
        if self._model_bundle is None:
            self.log.info(f"Loading model bundle from {self.model_dir}...")
            self._model_bundle = self._load_inference_module().model_fn(self.model_dir)
        return self._model_bundle

    def _needs_fresh_process(self) -> bool:
        """Internal: Whether this model has to be served outside this process.

        Torch and XGBoost each bring their own OpenMP runtime. Two in one process is
        undefined behavior, and unpickling an XGBoost Booster there segfaults the
        interpreter outright -- no exception, no traceback. Serving happens in separate
        containers in AWS, so the two never meet; in-process local inference is the
        first place they can.

        Returns:
            bool: True for an XGBoost model when torch is already imported here
        """
        return "torch" in sys.modules and bool(glob.glob(os.path.join(self.model_dir, "xgb*.joblib")))

    def _inference_in_fresh_process(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Internal: Run inference in a child process that never imports torch.

        The child loads the model on every call, so each inference pays a process start
        and a model load rather than reusing a bundle cached on this endpoint.

        Args:
            eval_df (pd.DataFrame): The data to run inference on

        Returns:
            pd.DataFrame: The predictions
        """
        inference_runner.inference_script(self.model_dir)  # missing artifacts fail here, not in the child
        self.log.important(f"torch is loaded in this process; serving XGBoost {self.model_name} from a child")

        with tempfile.TemporaryDirectory() as work_dir:
            input_path = os.path.join(work_dir, "input.pkl")
            output_path = os.path.join(work_dir, "predictions.pkl")
            eval_df.to_pickle(input_path)
            runner = [sys.executable, "-m", "workbench.local.inference_runner"]
            result = subprocess.run(runner + [self.model_dir, input_path, output_path], capture_output=True, text=True)
            if result.returncode != 0:
                tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
                raise RuntimeError(f"Inference for {self.model_name} failed in a child process:\n{tail}")
            return pd.read_pickle(output_path)

    def parent(self):
        """The Model this endpoint serves, if it still exists locally"""
        model = Model(self.get_input())
        return model if model.exists() else None

    def aws_exists(self) -> bool:
        """Does an AWS Endpoint by this name already exist?

        Returns:
            bool: True if AWS already has this Endpoint
        """
        from workbench.api import Endpoint as AWSEndpoint

        return AWSEndpoint(self.name).exists()

    def _aws_artifact(self):
        """Internal: The AWS Endpoint for this local one"""
        from workbench.api import Endpoint as AWSEndpoint

        return AWSEndpoint(self.name)

    def _publish_self(self, **kwargs):
        """Internal: Deploy the published Model as an AWS Endpoint

        Returns:
            Endpoint: The created AWS Endpoint
        """
        from workbench.api import Model as AWSModel

        return AWSModel(self.model_name).to_endpoint(name=self.name, **kwargs)

    def details(self, **kwargs) -> dict:
        """Endpoint Details

        Returns:
            dict: A dictionary of details about the Endpoint
        """
        return {**super().details(), "model_name": self.model_name, "captures": self.list_captures()}
