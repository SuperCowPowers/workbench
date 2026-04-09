"""MetaModel: A Model that aggregates predictions from multiple endpoints.

MetaModels don't train on feature data - they combine predictions from existing
endpoints using confidence-weighted voting. This provides ensemble benefits
across different model frameworks (XGBoost, PyTorch, ChemProp, etc.).
"""

from __future__ import annotations

from pathlib import Path
import time
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workbench.utils.meta_model_simulator import MetaModelSimulator

from sagemaker.core.resources import TrainingJob, ModelPackageGroup, ModelPackage
from sagemaker.core.shapes.model_card_shapes import InferenceSpecification, ContainersItem
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute, OutputDataConfig

# Workbench Imports
from workbench.api.model import Model
from workbench.api.endpoint import Endpoint
from workbench.api.feature_set import FeatureSet
from workbench.core.artifacts.model_core import ModelCore, ModelType, ModelFramework, ModelImages
from workbench.core.artifacts.artifact import Artifact
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.model_scripts.script_generation import generate_model_script
from workbench.utils.config_manager import ConfigManager
from workbench.utils.workbench_logging import _suppress_sagemaker_logging

# Set up logging
log = logging.getLogger("workbench")


class MetaModel(Model):
    """MetaModel: A Model that aggregates predictions from endpoints.

    Common Usage:
        ```python
        # Simulate ensemble performance before creating
        sim = MetaModel.simulate(["endpoint-1", "endpoint-2", "endpoint-3"])
        sim.report()

        # Create a meta model (auto-simulates to find best strategy)
        meta = MetaModel.create(
            name="my-meta-model",
            endpoints=["endpoint-1", "endpoint-2", "endpoint-3"],
            target_column="target"
        )

        # Deploy like any other model
        endpoint = meta.to_endpoint()
        ```
    """

    @classmethod
    def simulate(
        cls,
        endpoints: list[str],
        capture_name: str = "full_cross_fold",
    ) -> "MetaModelSimulator":
        """Run ensemble simulation to analyze how different strategies perform.

        Backtraces the first endpoint's lineage to automatically resolve the
        ID column from the underlying FeatureSet.

        Args:
            endpoints (list[str]): List of endpoint names to include in the simulation
            capture_name (str): Inference capture name to load predictions from
                (default: 'full_cross_fold')

        Returns:
            MetaModelSimulator: Simulator instance for analysis and reporting
        """
        from workbench.utils.meta_model_simulator import MetaModelSimulator

        id_column = cls._resolve_id_column(endpoints[0])
        model_names = [Endpoint(ep).get_input() for ep in endpoints]
        return MetaModelSimulator(model_names, id_column=id_column, capture_name=capture_name)

    @classmethod
    def create(
        cls,
        name: str,
        endpoints: list[str],
        description: str = None,
        tags: list[str] = None,
        capture_name: str = "full_cross_fold",
    ) -> "MetaModel":
        """Create a new MetaModel from a list of endpoints.

        Automatically backtraces endpoint lineage to resolve the target column
        and ID column, then runs ensemble simulation to find the best aggregation
        strategy, model weights, and confidence calibration parameters.

        Args:
            name (str): Name for the meta model
            endpoints (list[str]): List of endpoint names to aggregate
            description (str): Optional description for the model
            tags (list[str]): Optional list of tags
            capture_name (str): Inference capture name for simulation
                (default: 'full_cross_fold')

        Returns:
            MetaModel: The created meta model
        """
        from workbench.utils.meta_model_simulator import MetaModelSimulator

        Artifact.is_name_valid(name, delimiter="-", lower_case=False)

        # Validate endpoints and get lineage info from primary endpoint
        feature_list, feature_set_name, id_column, target_column = cls._validate_and_get_lineage(endpoints)

        # Map endpoint names → model names for the simulator
        ep_to_model = {ep: Endpoint(ep).get_input() for ep in endpoints}
        model_to_ep = {m: ep for ep, m in ep_to_model.items()}
        model_names = list(ep_to_model.values())

        # Run ensemble simulation to find best strategy
        log.important("Running ensemble simulation to find best strategy...")
        sim = MetaModelSimulator(model_names, id_column=id_column, capture_name=capture_name)
        sim.report()
        config = sim.get_best_strategy_config()

        # Map simulator results (model names) back to endpoint names
        aggregation_strategy = config["aggregation_strategy"]
        model_weights = {model_to_ep[m]: w for m, w in config["model_weights"].items()}
        corr_scale = {model_to_ep[m]: w for m, w in config["corr_scale"].items()}
        optimal_alpha = config["optimal_alpha"]
        final_endpoints = [model_to_ep[m] for m in config["endpoints"]]

        log.important(f"Best strategy: {aggregation_strategy}")
        log.important(f"Endpoints: {final_endpoints}")
        log.important(f"Model weights: {model_weights}")
        log.important(f"Optimal alpha: {optimal_alpha}")

        # Delete existing model if it exists
        log.important(f"Trying to delete existing model {name}...")
        ModelCore.managed_delete(name)

        # Run training and register model
        aws_clamp = AWSAccountClamp()
        training_job_name = cls._run_training(
            name,
            final_endpoints,
            target_column,
            model_weights,
            aggregation_strategy,
            corr_scale,
            optimal_alpha,
            aws_clamp,
        )
        cls._register_model(name, final_endpoints, description, tags, training_job_name, aws_clamp)

        # Set metadata and onboard
        cls._set_metadata(
            name,
            target_column,
            feature_list,
            feature_set_name,
            final_endpoints,
            aggregation_strategy=aggregation_strategy,
            model_weights=model_weights,
            corr_scale=corr_scale,
            optimal_alpha=optimal_alpha,
        )

        log.important(f"MetaModel {name} created successfully!")
        return cls(name)

    @classmethod
    def _resolve_id_column(cls, endpoint_name: str) -> str:
        """Backtrace an endpoint to find the ID column from its underlying FeatureSet.

        Args:
            endpoint_name (str): Endpoint name to backtrace

        Returns:
            str: The ID column name from the FeatureSet
        """
        ep = Endpoint(endpoint_name)
        model = Model(ep.get_input())
        feature_set_name = model.get_input()
        fs = FeatureSet(feature_set_name)
        log.info(f"Resolved id_column='{fs.id_column}' from {endpoint_name} -> {model.name} -> {feature_set_name}")
        return fs.id_column

    @classmethod
    def _validate_and_get_lineage(cls, endpoints: list[str]) -> tuple[list[str], str, str, str]:
        """Validate endpoints exist and get lineage info from primary endpoint.

        Backtraces: endpoint → model → feature_set to resolve features, id_column,
        and target_column automatically.

        Args:
            endpoints (list[str]): List of endpoint names

        Returns:
            tuple: (feature_list, feature_set_name, id_column, target_column)
        """
        log.info("Verifying endpoints...")
        for ep_name in endpoints:
            ep = Endpoint(ep_name)
            if not ep.exists():
                raise ValueError(f"Endpoint '{ep_name}' does not exist")
            log.info(f"  {ep_name}: OK")

        # Use first endpoint as primary - backtrack to get model and feature set
        primary_endpoint = Endpoint(endpoints[0])
        primary_model = Model(primary_endpoint.get_input())
        feature_list = primary_model.features()
        feature_set_name = primary_model.get_input()
        target_column = primary_model.target()
        fs = FeatureSet(feature_set_name)
        id_column = fs.id_column

        log.info(
            f"Primary: {endpoints[0]} -> {primary_model.name} -> {feature_set_name} "
            f"(id_column: {id_column}, target: {target_column})"
        )
        return feature_list, feature_set_name, id_column, target_column

    @classmethod
    def _run_training(
        cls,
        name: str,
        endpoints: list[str],
        target_column: str,
        model_weights: dict[str, float],
        aggregation_strategy: str,
        corr_scale: dict[str, float] | None,
        optimal_alpha: float,
        aws_clamp: AWSAccountClamp,
    ) -> str:
        """Run the minimal training job that saves the meta model config.

        Args:
            name (str): Model name
            endpoints (list[str]): List of endpoint names
            target_column (str): Target column name
            model_weights (dict): Dict mapping endpoint name to weight
            aggregation_strategy (str): Ensemble aggregation strategy name
            corr_scale (dict): Dict mapping endpoint name to |confidence_error_correlation|
            optimal_alpha (float): Blend weight for ensemble confidence
            aws_clamp (AWSAccountClamp): AWS account clamp

        Returns:
            str: The training job name
        """
        sm_session = aws_clamp.sagemaker_session()
        cm = ConfigManager()
        workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
        models_s3_path = f"s3://{workbench_bucket}/models"

        # Generate the model script from template
        template_params = {
            "model_type": ModelType.REGRESSOR,
            "model_framework": ModelFramework.META,
            "endpoints": endpoints,
            "target_column": target_column,
            "model_weights": model_weights,
            "aggregation_strategy": aggregation_strategy,
            "corr_scale": corr_scale or {},
            "optimal_alpha": optimal_alpha,
            "model_metrics_s3_path": f"{models_s3_path}/{name}/training",
            "aws_region": sm_session.boto_region_name,
        }
        script_path = generate_model_script(template_params)

        # Create ModelTrainer (V3 replacement for Estimator)
        training_image = ModelImages.get_image_uri(sm_session.boto_region_name, "base_training")
        log.info(f"Using Meta Training Image: {training_image}")
        entry_point = Path(script_path).name
        source_dir = str(Path(script_path).parent)
        trainer = ModelTrainer(
            training_image=training_image,
            source_code=SourceCode(
                source_dir=source_dir,
                command=f"python training_harness.py {entry_point}",
            ),
            compute=Compute(instance_type="ml.m5.large", instance_count=1),
            output_data_config=OutputDataConfig(s3_output_path=f"{models_s3_path}/{name}/training", compression_type="GZIP"),
            role=aws_clamp.aws_session.get_workbench_execution_role_arn(),
            sagemaker_session=sm_session,
            base_job_name=name,
        )

        # Run training (no input data needed - just saves config)
        log.important(f"Creating MetaModel {name}...")
        _suppress_sagemaker_logging()
        trainer.train(wait=True)

        return trainer._latest_training_job.training_job_name

    @classmethod
    def _register_model(
        cls,
        name: str,
        endpoints: list[str],
        description: str,
        tags: list[str],
        training_job_name: str,
        aws_clamp: AWSAccountClamp,
    ):
        """Create model group and register the model.

        Args:
            name (str): Model name
            endpoints (list[str]): List of endpoint names
            description (str): Model description
            tags (list[str]): Model tags
            training_job_name (str): Name of the completed training job
            aws_clamp (AWSAccountClamp): AWS account clamp
        """
        sm_session = aws_clamp.sagemaker_session()
        boto3_session = aws_clamp.boto3_session
        model_description = description or f"Meta model aggregating: {', '.join(endpoints)}"

        # Create model group
        aws_tags = [{"key": "workbench_tags", "value": "::".join(tags or [name])}]
        try:
            ModelPackageGroup.create(
                model_package_group_name=name,
                model_package_group_description=model_description,
                tags=aws_tags,
                session=boto3_session,
            )
        except Exception:
            log.info(f"Model Package Group {name} may already exist, continuing...")

        # Get the model artifacts URL from the completed training job
        training_job = TrainingJob.get(training_job_name, session=boto3_session)
        model_data_url = training_job.model_artifacts.s3_model_artifacts

        # Register the model with meta inference image
        inference_image = ModelImages.get_image_uri(sm_session.boto_region_name, "base_inference")
        log.important(f"Registering model {name} with Inference Image {inference_image}...")

        container = ContainersItem(image=inference_image, model_data_url=model_data_url)
        ModelPackage.create(
            model_package_group_name=name,
            model_package_description=model_description,
            inference_specification=InferenceSpecification(containers=[container]),
            model_approval_status="Approved",
            tags=aws_tags,
            session=boto3_session,
        )

    @classmethod
    def _set_metadata(
        cls,
        name: str,
        target_column: str,
        feature_list: list[str],
        feature_set_name: str,
        endpoints: list[str],
        aggregation_strategy: str = None,
        model_weights: dict[str, float] = None,
        corr_scale: dict[str, float] = None,
        optimal_alpha: float = None,
    ):
        """Set model metadata and onboard.

        Args:
            name (str): Model name
            target_column (str): Target column name
            feature_list (list[str]): List of feature names
            feature_set_name (str): Name of the input FeatureSet
            endpoints (list[str]): List of endpoint names
            aggregation_strategy (str): Ensemble aggregation strategy name
            model_weights (dict): Dict mapping endpoint name to weight
            corr_scale (dict): Dict mapping endpoint name to |confidence_error_correlation|
            optimal_alpha (float): Blend weight for ensemble confidence
        """
        time.sleep(3)
        output_model = ModelCore(name)
        output_model._set_model_type(ModelType.UQ_REGRESSOR)
        output_model._set_model_framework(ModelFramework.META)
        output_model.set_input(feature_set_name, force=True)
        output_model.upsert_workbench_meta({"workbench_model_target": target_column})
        output_model.upsert_workbench_meta({"workbench_model_features": feature_list})
        output_model.upsert_workbench_meta({"endpoints": endpoints})
        if aggregation_strategy:
            output_model.upsert_workbench_meta({"aggregation_strategy": aggregation_strategy})
        if model_weights:
            output_model.upsert_workbench_meta({"model_weights": model_weights})
        if corr_scale:
            output_model.upsert_workbench_meta({"corr_scale": corr_scale})
        if optimal_alpha is not None:
            output_model.upsert_workbench_meta({"optimal_alpha": optimal_alpha})
        output_model.onboard_with_args(ModelType.UQ_REGRESSOR, target_column, feature_list=feature_list)

    def get_meta_config(self) -> dict:
        """Retrieve the meta model configuration (strategy, weights, corr_scale).

        Returns:
            dict: Meta config with keys: endpoints, aggregation_strategy,
                model_weights, corr_scale, target_column

        Raises:
            ValueError: If aggregation_strategy or model_weights are not in workbench_meta.
                Re-create the meta model to populate these fields.
        """
        meta = self.workbench_meta()

        if not meta.get("aggregation_strategy") or not meta.get("model_weights"):
            raise ValueError(
                f"Meta config (aggregation_strategy, model_weights) not found in workbench_meta "
                f"for '{self.name}'. Re-create the meta model to populate these fields."
            )

        return {
            "endpoints": meta["endpoints"],
            "aggregation_strategy": meta["aggregation_strategy"],
            "model_weights": meta["model_weights"],
            "corr_scale": meta.get("corr_scale", {}),
            "optimal_alpha": meta.get("optimal_alpha", 0.5),
            "target_column": meta.get("workbench_model_target"),
        }


if __name__ == "__main__":
    """Exercise the MetaModel Class"""

    # Simulate ensemble performance first
    sim = MetaModel.simulate(["logd-xgb", "logd-pytorch", "logd-chemprop"])
    sim.report()

    # Create meta model (auto-simulates internally)
    meta = MetaModel.create(
        name="logd-meta",
        endpoints=["logd-xgb", "logd-pytorch", "logd-chemprop"],
        description="Meta model for LogD prediction",
        tags=["meta", "logd", "ensemble"],
    )
    print(meta.summary())

    # Create an endpoint for the meta model
    end = meta.to_endpoint(tags=["meta", "logd"])
    end.set_owner("BW")
    end.auto_inference()

    # Test loading an existing meta model
    meta = MetaModel("logd-meta")
    print(meta.details())
