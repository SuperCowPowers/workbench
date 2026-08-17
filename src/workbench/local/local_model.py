"""LocalModel: A model trained on this machine by the generated model script."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Union

import pandas as pd

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.core.model_types import ModelType, ModelFramework
from workbench.local.local_artifact import LocalArtifact
from workbench.local import storage
from workbench.model_scripts.script_generation import generate_model_script, fill_template
from workbench.utils import job_tracker


class LocalModel(LocalArtifact):
    """LocalModel: Workbench Local Model Class

    Training runs the same generated model script that SageMaker runs, with the
    same arguments, as a subprocess against local directories.

    Common Usage:
        ```python
        my_model = LocalModel("my_model")
        my_model.training_state()
        my_model.oof_predictions()
        ```
    """

    artifact_type = "model"
    data_files = ()

    def __init__(self, name: str, **kwargs):
        """Initialize a LocalModel

        Args:
            name (str): The name of the model
        """
        Artifact.is_name_valid(name, delimiter="-", lower_case=False)
        super().__init__(name, **kwargs)
        self.script_dir = os.path.join(self.path, "script")
        self.model_dir = os.path.join(self.path, "model_artifacts")
        self.output_dir = os.path.join(self.path, "output")
        self.train_dir = os.path.join(self.path, "input", "train")
        self.log_path = os.path.join(self.output_dir, "training.log")
        self.status_path = os.path.join(self.path, "status.json")

    @classmethod
    def from_feature_set(
        cls,
        feature_set,
        name: str,
        model_type: ModelType,
        model_framework: ModelFramework,
        target_column: Union[str, list[str]] = None,
        feature_list: list = None,
        model_class: str = None,
        model_import_str: str = None,
        custom_script: str = None,
        hyperparameters: dict = None,
        sample_weights: Union[dict, pd.DataFrame] = None,
        validation_ids: list = None,
        exclude_ids: list = None,
        wait: bool = True,
    ) -> "LocalModel":
        """Train a LocalModel from a LocalFeatureSet.

        Args:
            feature_set (LocalFeatureSet): The feature set to train on
            name (str): The name of the model to create
            model_type (ModelType): The type of model to create
            model_framework (ModelFramework): The framework to use
            target_column (str or list[str], optional): Target column(s), None for unsupervised
            feature_list (list, optional): Feature columns; derived from the FeatureSet if omitted
            model_class (str, optional): Model class for scikit-learn models (e.g. "KMeans")
            model_import_str (str, optional): Import line for the model class
            custom_script (str, optional): Path to a custom model script or template
            hyperparameters (dict, optional): Hyperparameters for the model
            sample_weights (Union[dict, pd.DataFrame], optional): id -> framework weight
            validation_ids (list, optional): ids held out and scored as a validation set
            exclude_ids (list, optional): ids dropped from training entirely
            wait (bool): Block until training finishes (default: True)

        Returns:
            LocalModel: The model (trained if wait=True, still training otherwise)
        """
        supervised = model_type in (
            ModelType.CLASSIFIER,
            ModelType.REGRESSOR,
            ModelType.UQ_REGRESSOR,
            ModelType.ENSEMBLE_REGRESSOR,
        )
        if target_column is None and supervised:
            raise ValueError("target_column is required for supervised models (pass target_column=...)")

        model = cls(name)
        model._init_dirs(input_name=feature_set.name)

        # Stage the training data: features plus the sample_weight/validation/exclude roles
        train_df = feature_set.training_view(
            sample_weights=sample_weights, validation_ids=validation_ids, exclude_ids=exclude_ids
        )
        train_df.to_csv(os.path.join(model.train_dir, "train.csv"), index=False)
        model.log.important(f"Staged {len(train_df)} training rows for {name}...")

        # Derive the feature list the same way the AWS path does when it isn't given
        target_list = [target_column] if isinstance(target_column, str) else (target_column or [])
        if feature_list is None:
            feature_list = model._derive_feature_list(feature_set, target_list)
        model.log.important(f"Feature List for Modeling: {feature_list}")

        # Generate the script, then keep it with the model so a training run is reproducible
        target_for_template = target_list if model_framework == ModelFramework.CHEMPROP else target_column
        template_params = {
            "model_imports": model_import_str,
            "model_type": model_type,
            "model_framework": model_framework,
            "model_class": model_class,
            "target_column": target_for_template,
            "feature_list": feature_list,
            "compressed_features": [],
            "model_metrics_path": model.output_dir,
            "id_column": feature_set.id_column,
            "hyperparameters": hyperparameters or {},
        }
        script_path = model._build_script(template_params, custom_script)
        shutil.copytree(os.path.dirname(script_path), model.script_dir, dirs_exist_ok=True)

        model.upsert_workbench_meta(
            {
                "workbench_model_features": feature_list,
                "workbench_model_target": target_column,
                "model_type": model_type.value,
                "model_framework": model_framework.value,
                "id_column": feature_set.id_column,
                "hyperparameters": hyperparameters or {},
                # All three row roles are kept, so publish() trains in AWS on the same
                # rows with the same weights. Dropping any of them here would make the
                # published model quietly differ from the local one.
                "sample_weights": cls._weights_as_pairs(sample_weights),
                "validation_ids": list(validation_ids) if validation_ids else None,
                "exclude_ids": list(exclude_ids) if exclude_ids else None,
            }
        )

        model._launch_training(wait=wait)
        return model

    @staticmethod
    def _weights_as_pairs(sample_weights: Union[dict, pd.DataFrame, None]) -> Union[list, None]:
        """Internal: Normalize sample weights to JSON-storable [id, weight] pairs.

        Stored as pairs rather than a mapping because JSON object keys are always
        strings: integer ids would come back as strings and fail to join against the
        FeatureSet's id column, silently dropping the weights.

        Args:
            sample_weights (Union[dict, pd.DataFrame, None]): Weights as given by the caller

        Returns:
            Union[list, None]: [[id, weight], ...], or None when there are no weights
        """
        if sample_weights is None:
            return None
        if isinstance(sample_weights, pd.DataFrame):
            if sample_weights.empty:
                return None
            id_column = sample_weights.columns[0]
            sample_weights = dict(zip(sample_weights[id_column], sample_weights["sample_weight"]))
        pairs = [[key, float(value)] for key, value in sample_weights.items()]
        return pairs or None

    def _build_script(self, template_params: dict, custom_script: str = None) -> str:
        """Internal: Produce the model script, from a built-in template or a custom one.

        Args:
            template_params (dict): Parameters filled into the template
            custom_script (str, optional): Path to a custom script or .template

        Returns:
            str: Path to the script to run
        """
        if not custom_script:
            return generate_model_script(template_params)

        if not str(custom_script).endswith(".template"):
            return str(custom_script)

        # A custom template gets the same params, with the enum flattened like the generator does
        template_params = {**template_params, "model_type": template_params["model_type"].value}
        return fill_template(custom_script, template_params, "generated_model_script.py")

    def _derive_feature_list(self, feature_set, target_list: list) -> list:
        """Internal: Guess a feature list from the FeatureSet's numeric columns.

        Args:
            feature_set (LocalFeatureSet): The feature set being trained on
            target_list (list): Target column(s) to exclude

        Returns:
            list: The derived feature list
        """
        self.log.warning("Guessing at the feature list, HIGHLY RECOMMENDED to specify an explicit feature list!")
        skip = {"id", "auto_id", "__index_level_0__", "event_time", "training", feature_set.id_column} | set(
            target_list
        )
        df = feature_set.pull_dataframe(limit=1)
        return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    def _init_dirs(self, input_name: str):
        """Internal: Create the model's directory layout

        Args:
            input_name (str): Name of this model's input FeatureSet
        """
        storage.local_root(create=True)
        os.makedirs(self.path, exist_ok=True)

        # Start each run from empty dirs. Leftovers from a previous run would otherwise
        # survive a failure and be served as this run's artifacts and predictions.
        for directory in (self.model_dir, self.output_dir, self.train_dir):
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)
        shutil.rmtree(self.script_dir, ignore_errors=True)

        self._init_storage(input_name=input_name)

    def _launch_training(self, wait: bool):
        """Internal: Run the generated model script as a subprocess.

        Args:
            wait (bool): Block until training finishes, streaming output
        """
        script = os.path.join(self.script_dir, "generated_model_script.py")
        command = [
            sys.executable,
            script,
            "--model-dir",
            self.model_dir,
            "--train",
            self.train_dir,
            "--output-data-dir",
            self.output_dir,
        ]
        self.log.important(f"Training {self.name} locally: {' '.join(command)}")
        self.upsert_workbench_meta({"workbench_status": "training"})

        if not wait:
            # The child dups the descriptor, so ours is closed as soon as it's launched
            with open(self.log_path, "w") as log_file:
                proc = subprocess.Popen(
                    command, cwd=self.script_dir, stdout=log_file, stderr=subprocess.STDOUT, text=True
                )
            self._write_status(state="training", pid=proc.pid)
            job_tracker.watch_subprocess(
                self.name,
                proc,
                kind="Local training",
                log_path=self.log_path,
                on_finish=self._record_outcome,
            )
            return

        proc = subprocess.Popen(
            command,
            cwd=self.script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._write_status(state="training", pid=proc.pid)

        # Stream the child's output so a long training run isn't a silent block
        with open(self.log_path, "w") as log_file:
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
        proc.wait()

        self._finish_training(proc.returncode)

    def _record_outcome(self, returncode: int) -> bool:
        """Internal: Write the durable outcome of a training run.

        Called by the wait=True path directly and by the job watcher for wait=False,
        so a detached run leaves the same on-disk state as a blocking one.

        Args:
            returncode (int): The model script's exit code

        Returns:
            bool: True if training succeeded
        """
        success = returncode == 0
        if success:
            self._bundle_for_inference()
        self.refresh_meta()
        self._write_status(state="completed" if success else "failed", returncode=returncode)
        self.upsert_workbench_meta({"workbench_status": "ready" if success else "failed"})
        return success

    def _bundle_for_inference(self):
        """Internal: Put the inference code and metadata in the model directory.

        SageMaker's entry point is ``training_harness.py``, which runs the model script
        and then bundles the code so the inference container knows what to serve. Local
        runs the model script directly and calls the same bundling step here.

        The harness's other job -- pip installing the script's requirements.txt -- is
        deliberately skipped: local dependencies are the user's environment, not
        something a training run should mutate.
        """
        from workbench.training.training_harness import include_code_and_meta_for_inference

        include_code_and_meta_for_inference(
            model_dir=self.model_dir,
            code_dir=self.script_dir,
            entry_point="generated_model_script.py",
        )

    def _finish_training(self, returncode: int):
        """Internal: Record the outcome of a blocking training run, raising on failure

        Args:
            returncode (int): The model script's exit code

        Raises:
            RuntimeError: If the model script exited non-zero
        """
        if not self._record_outcome(returncode):
            tail = self.training_log(lines=15)
            self.log.error(f"Local training failed for {self.name} (exit {returncode}):\n{tail}")
            raise RuntimeError(f"Local training failed for {self.name} (exit {returncode}), see {self.log_path}")
        self.log.important(f"Local training complete: {self.model_dir}")

    def _write_status(self, state: str, pid: int = None, returncode: int = None):
        """Internal: Write the training status file used to reattach across sessions

        Args:
            state (str): One of training/completed/failed
            pid (int, optional): The training process id
            returncode (int, optional): The training process exit code
        """
        status = self.training_state()
        status["state"] = state
        status["updated"] = datetime.now(timezone.utc).isoformat()
        if pid is not None:
            status["pid"] = pid
            status["started"] = status["updated"]
        if returncode is not None:
            status["returncode"] = returncode
            status["finished"] = status["updated"]
        with open(self.status_path, "w") as fp:
            json.dump(status, fp, indent=4)

    def training_state(self) -> dict:
        """The training status for this model, as recorded on disk.

        A run whose watcher never got to record an outcome -- the session exited, or the
        process died -- is reported as "interrupted" rather than left claiming to be
        training forever. Whether the child finished its work is unknown at that point.

        Returns:
            dict: {state, pid, started, updated, returncode, finished}, empty before any run
        """
        try:
            with open(self.status_path, "r") as fp:
                status = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

        if status.get("state") == "training" and not self._pid_alive(status.get("pid")):
            status["state"] = "interrupted"
        return status

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Internal: Is a process still running?

        Args:
            pid (int): The process id to check

        Returns:
            bool: True if the process exists
        """
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def training_log(self, lines: int = None) -> str:
        """The training log for this model.

        Args:
            lines (int, optional): Return only the last N lines (default: the whole log)

        Returns:
            str: The log contents ("" if the model hasn't trained)
        """
        try:
            with open(self.log_path, "r") as fp:
                content = fp.readlines()
        except OSError:
            return ""
        return "".join(content[-lines:] if lines else content)

    def oof_predictions(self) -> pd.DataFrame:
        """Out-of-fold predictions written by the training run.

        Returns:
            pd.DataFrame: The OOF predictions (empty if the model hasn't trained)
        """
        return self._output_csv("oof_predictions.csv")

    def validation_predictions(self) -> pd.DataFrame:
        """Held-out validation predictions written by the training run.

        Returns:
            pd.DataFrame: The validation predictions (empty if there was no validation set)
        """
        return self._output_csv("val_predictions.csv")

    def _output_csv(self, file_name: str) -> pd.DataFrame:
        """Internal: Read a CSV the training run wrote to the output directory

        Args:
            file_name (str): The file name within the output directory

        Returns:
            pd.DataFrame: The file's contents (empty if missing)
        """
        path = os.path.join(self.output_dir, file_name)
        return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()

    def to_endpoint(self, name: str = None) -> "LocalEndpoint":  # noqa: F821
        """Create a LocalEndpoint that serves this model.

        Args:
            name (str, optional): Endpoint name (defaults to the model name)

        Returns:
            LocalEndpoint: The endpoint serving this model
        """
        from workbench.local.local_endpoint import LocalEndpoint

        return LocalEndpoint.from_model(self, name=name)

    def parent(self):
        """The LocalFeatureSet this model trained on, if it still exists locally"""
        from workbench.local.local_feature_set import LocalFeatureSet

        feature_set = LocalFeatureSet(self.get_input())
        return feature_set if feature_set.exists() else None

    def aws_exists(self) -> bool:
        """Does an AWS Model by this name already exist?

        Returns:
            bool: True if AWS already has this Model
        """
        from workbench.api import Model

        return Model(self.name).exists()

    def _aws_artifact(self):
        """Internal: The AWS Model for this local one"""
        from workbench.api import Model

        return Model(self.name)

    def version_drift(self) -> str:
        """Package versions that differ between this machine and the training image.

        Returns:
            str: A drift report, or "" when everything that matters matches
        """
        from workbench.utils.version_drift import drift_summary

        return drift_summary(self.workbench_meta().get("model_framework", "xgboost"))

    def _publish_self(self, **kwargs):
        """Internal: Train this model in AWS from the published FeatureSet.

        Publishing retrains rather than uploading local artifacts, so the model lands in
        the registry the same way any AWS model does. The row roles recorded at local
        training time are replayed, so AWS trains on the same rows.

        Returns:
            Model: The created AWS Model
        """
        from workbench.api import FeatureSet

        meta = self.workbench_meta()
        feature_set = FeatureSet(self.get_input())
        return feature_set.to_model(
            name=self.name,
            model_type=ModelType(meta["model_type"]),
            model_framework=ModelFramework(meta["model_framework"]),
            target_column=meta.get("workbench_model_target"),
            feature_list=meta.get("workbench_model_features"),
            hyperparameters=meta.get("hyperparameters") or {},
            sample_weights=dict(meta["sample_weights"]) if meta.get("sample_weights") else None,
            validation_ids=meta.get("validation_ids"),
            exclude_ids=meta.get("exclude_ids"),
            **kwargs,
        )

    def publish(self, endpoint: bool = True, **kwargs: Any) -> "Model":  # noqa: F821
        """Publish this model and its lineage to AWS, then deploy an endpoint.

        Args:
            endpoint (bool): Also deploy a serverless endpoint (default True)
            **kwargs: Passed to the AWS training job

        Returns:
            Model: The published AWS Model
        """
        from workbench.api import Endpoint

        aws_model = super().publish(**kwargs)
        if not endpoint:
            return aws_model

        endpoint_name = self._endpoint_name()
        if Endpoint(endpoint_name).exists():
            self.log.important(f"AWS endpoint '{endpoint_name}' already exists, skipping...")
        else:
            self.log.important(f"Deploying endpoint '{endpoint_name}' to AWS...")
            aws_model.to_endpoint(name=endpoint_name)
        return aws_model

    def _endpoints(self) -> list:
        """Internal: The local endpoints serving this model"""
        from workbench.local import storage
        from workbench.local.local_endpoint import LocalEndpoint

        serving = [LocalEndpoint(name) for name in storage.list_artifacts("endpoint")]
        return [endpoint for endpoint in serving if endpoint.model_name == self.name]

    def _endpoint_name(self) -> str:
        """Internal: The name to deploy this model's AWS endpoint under.

        A local endpoint may carry a custom name, so publishing reuses it. Otherwise
        this is the same default AWS uses: the model name.

        Returns:
            str: The endpoint name
        """
        serving = self._endpoints()
        return serving[0].name if serving else self.name

    def delete(self):
        """Delete this model and the endpoints serving it.

        An endpoint is not an independent artifact -- it loads from the model's
        directory and is meaningless once that is gone, so it comes down too. This is
        the only cascade: deleting a FeatureSet leaves its models alone.
        """
        for endpoint in self._endpoints():
            endpoint.delete()
        super().delete()

    def details(self, **kwargs) -> dict:
        """LocalModel Details

        Returns:
            dict: A dictionary of details about the LocalModel
        """
        return {**super().details(), "training": self.training_state()}
