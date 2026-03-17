"""MetaModel: A Model that aggregates predictions from multiple endpoints.

MetaModels don't train on feature data - they combine predictions from existing
endpoints using confidence-weighted voting. This provides ensemble benefits
across different model frameworks (XGBoost, PyTorch, ChemProp, etc.).
"""

from pathlib import Path
import time
import logging

from sagemaker.estimator import Estimator

# Workbench Imports
from workbench.api.model import Model
from workbench.api.endpoint import Endpoint
from workbench.api.feature_set import FeatureSet
from workbench.core.artifacts.model_core import ModelCore, ModelType, ModelFramework, ModelImages
from workbench.core.artifacts.artifact import Artifact
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.model_scripts.script_generation import generate_model_script
from workbench.utils.config_manager import ConfigManager
from workbench.utils.model_utils import supported_instance_types

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
    ):
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
        return MetaModelSimulator(endpoints, id_column=id_column, capture_name=capture_name)

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

        # Run ensemble simulation to find best strategy
        log.important("Running ensemble simulation to find best strategy...")
        sim = MetaModelSimulator(endpoints, id_column=id_column, capture_name=capture_name)
        sim.report()
        config = sim.get_best_strategy_config()

        # Use the simulator's recommended config
        aggregation_strategy = config["aggregation_strategy"]
        model_weights = config["model_weights"]
        corr_scale = config["corr_scale"]
        final_endpoints = config["endpoints"]  # May differ if drop_worst won

        log.important(f"Best strategy: {aggregation_strategy}")
        log.important(f"Endpoints: {final_endpoints}")
        log.important(f"Model weights: {model_weights}")

        # Delete existing model if it exists
        log.important(f"Trying to delete existing model {name}...")
        ModelCore.managed_delete(name)

        # Run training and register model
        aws_clamp = AWSAccountClamp()
        estimator = cls._run_training(
            name, final_endpoints, target_column, model_weights, aggregation_strategy, corr_scale, aws_clamp
        )
        cls._register_model(name, final_endpoints, description, tags, estimator, aws_clamp)

        # Set metadata and onboard
        cls._set_metadata(name, target_column, feature_list, feature_set_name, final_endpoints)

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
        aws_clamp: AWSAccountClamp,
    ) -> Estimator:
        """Run the minimal training job that saves the meta model config.

        Args:
            name (str): Model name
            endpoints (list[str]): List of endpoint names
            target_column (str): Target column name
            model_weights (dict): Dict mapping endpoint name to weight
            aggregation_strategy (str): Ensemble aggregation strategy name
            corr_scale (dict): Dict mapping endpoint name to |confidence_error_correlation|
            aws_clamp (AWSAccountClamp): AWS account clamp

        Returns:
            Estimator: The fitted estimator
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
            "model_metrics_s3_path": f"{models_s3_path}/{name}/training",
            "aws_region": sm_session.boto_region_name,
        }
        script_path = generate_model_script(template_params)

        # Create estimator
        training_image = ModelImages.get_image_uri(sm_session.boto_region_name, "meta_training")
        log.info(f"Using Meta Training Image: {training_image}")
        estimator = Estimator(
            entry_point=Path(script_path).name,
            source_dir=str(Path(script_path).parent),
            role=aws_clamp.aws_session.get_workbench_execution_role_arn(),
            instance_count=1,
            instance_type="ml.m5.large",
            sagemaker_session=sm_session,
            image_uri=training_image,
        )

        # Run training (no input data needed - just saves config)
        log.important(f"Creating MetaModel {name}...")
        estimator.fit()

        return estimator

    @classmethod
    def _register_model(
        cls,
        name: str,
        endpoints: list[str],
        description: str,
        tags: list[str],
        estimator: Estimator,
        aws_clamp: AWSAccountClamp,
    ):
        """Create model group and register the model.

        Args:
            name (str): Model name
            endpoints (list[str]): List of endpoint names
            description (str): Model description
            tags (list[str]): Model tags
            estimator (Estimator): Fitted estimator
            aws_clamp (AWSAccountClamp): AWS account clamp
        """
        sm_session = aws_clamp.sagemaker_session()
        model_description = description or f"Meta model aggregating: {', '.join(endpoints)}"

        # Create model group
        aws_clamp.sagemaker_client().create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription=model_description,
            Tags=[{"Key": "workbench_tags", "Value": "::".join(tags or [name])}],
        )

        # Register the model with meta inference image
        inference_image = ModelImages.get_image_uri(sm_session.boto_region_name, "meta_inference")
        log.important(f"Registering model {name} with Inference Image {inference_image}...")
        estimator.create_model(role=aws_clamp.aws_session.get_workbench_execution_role_arn()).register(
            model_package_group_name=name,
            image_uri=inference_image,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=supported_instance_types("x86_64"),
            transform_instances=["ml.m5.large", "ml.m5.xlarge"],
            approval_status="Approved",
            description=model_description,
        )

    @classmethod
    def _set_metadata(
        cls, name: str, target_column: str, feature_list: list[str], feature_set_name: str, endpoints: list[str]
    ):
        """Set model metadata and onboard.

        Args:
            name (str): Model name
            target_column (str): Target column name
            feature_list (list[str]): List of feature names
            feature_set_name (str): Name of the input FeatureSet
            endpoints (list[str]): List of endpoint names
        """
        time.sleep(3)
        output_model = ModelCore(name)
        output_model._set_model_type(ModelType.UQ_REGRESSOR)
        output_model._set_model_framework(ModelFramework.META)
        output_model.set_input(feature_set_name, force=True)
        output_model.upsert_workbench_meta({"workbench_model_target": target_column})
        output_model.upsert_workbench_meta({"workbench_model_features": feature_list})
        output_model.upsert_workbench_meta({"endpoints": endpoints})
        output_model.onboard_with_args(ModelType.UQ_REGRESSOR, target_column, feature_list=feature_list)


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
