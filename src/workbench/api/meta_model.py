"""MetaModel: A Model that aggregates predictions from multiple child endpoints.

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
from workbench.core.artifacts.model_core import ModelCore, ModelType, ModelFramework, ModelImages
from workbench.core.artifacts.artifact import Artifact
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.model_scripts.script_generation import generate_model_script
from workbench.utils.config_manager import ConfigManager
from workbench.utils.model_utils import supported_instance_types

# Set up logging
log = logging.getLogger("workbench")


class MetaModel(Model):
    """MetaModel: A Model that aggregates predictions from child endpoints.

    Common Usage:
        ```python
        # Create a meta model from existing endpoints
        meta = MetaModel.create(
            name="my-meta-model",
            child_endpoints=["endpoint-1", "endpoint-2", "endpoint-3"],
            target_column="target"
        )

        # Deploy like any other model
        endpoint = meta.to_endpoint()
        ```
    """

    @classmethod
    def create(
        cls,
        name: str,
        child_endpoints: list[str],
        target_column: str,
        description: str = None,
        tags: list[str] = None,
    ) -> "MetaModel":
        """Create a new MetaModel from a list of child endpoints.

        Args:
            name: Name for the meta model
            child_endpoints: List of endpoint names to aggregate
            target_column: Name of the target column (for metadata)
            description: Optional description for the model
            tags: Optional list of tags

        Returns:
            MetaModel: The created meta model
        """
        Artifact.is_name_valid(name, delimiter="-", lower_case=False)

        # Validate endpoints and get lineage info from primary endpoint
        feature_list, feature_set_name, model_weights = cls._validate_and_get_lineage(child_endpoints)

        # Delete existing model if it exists
        log.important(f"Trying to delete existing model {name}...")
        ModelCore.managed_delete(name)

        # Run training and register model
        aws_clamp = AWSAccountClamp()
        estimator = cls._run_training(name, child_endpoints, target_column, model_weights, aws_clamp)
        cls._register_model(name, child_endpoints, description, tags, estimator, aws_clamp)

        # Set metadata and onboard
        cls._set_metadata(name, target_column, feature_list, feature_set_name, child_endpoints)

        log.important(f"MetaModel {name} created successfully!")
        return cls(name)

    @classmethod
    def _validate_and_get_lineage(cls, child_endpoints: list[str]) -> tuple[list[str], str, dict[str, float]]:
        """Validate child endpoints exist and get lineage info from primary endpoint.

        Args:
            child_endpoints: List of endpoint names

        Returns:
            tuple: (feature_list, feature_set_name, model_weights) from the primary endpoint's model
        """
        log.info("Verifying child endpoints and gathering model metrics...")
        mae_scores = {}

        for ep_name in child_endpoints:
            ep = Endpoint(ep_name)
            if not ep.exists():
                raise ValueError(f"Child endpoint '{ep_name}' does not exist")

            # Get model MAE from full_inference metrics
            model = Model(ep.get_input())
            metrics = model.get_inference_metrics("full_inference")
            if metrics is not None and "mae" in metrics.columns:
                mae = float(metrics["mae"].iloc[0])
                mae_scores[ep_name] = mae
                log.info(f"  {ep_name} -> {model.name}: MAE={mae:.4f}")
            else:
                log.warning(f"  {ep_name}: No full_inference metrics found, using default weight")
                mae_scores[ep_name] = None

        # Compute inverse-MAE weights (higher weight for lower MAE)
        valid_mae = {k: v for k, v in mae_scores.items() if v is not None}
        if valid_mae:
            inv_mae = {k: 1.0 / v for k, v in valid_mae.items()}
            total = sum(inv_mae.values())
            model_weights = {k: v / total for k, v in inv_mae.items()}
            # Fill in missing weights with equal share of remaining weight
            missing = [k for k in mae_scores if mae_scores[k] is None]
            if missing:
                equal_weight = (1.0 - sum(model_weights.values())) / len(missing)
                for k in missing:
                    model_weights[k] = equal_weight
        else:
            # No metrics available, use equal weights
            model_weights = {k: 1.0 / len(child_endpoints) for k in child_endpoints}
            log.warning("No MAE metrics found, using equal weights")

        log.info(f"Model weights: {model_weights}")

        # Use first endpoint as primary - backtrack to get model and feature set
        primary_endpoint = Endpoint(child_endpoints[0])
        primary_model = Model(primary_endpoint.get_input())
        feature_list = primary_model.features()
        feature_set_name = primary_model.get_input()

        log.info(
            f"Primary endpoint: {child_endpoints[0]} -> Model: {primary_model.name} -> FeatureSet: {feature_set_name}"
        )
        return feature_list, feature_set_name, model_weights

    @classmethod
    def _run_training(
        cls,
        name: str,
        child_endpoints: list[str],
        target_column: str,
        model_weights: dict[str, float],
        aws_clamp: AWSAccountClamp,
    ) -> Estimator:
        """Run the minimal training job that saves the meta model config.

        Args:
            name: Model name
            child_endpoints: List of endpoint names
            target_column: Target column name
            model_weights: Dict mapping endpoint name to weight
            aws_clamp: AWS account clamp

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
            "child_endpoints": child_endpoints,
            "target_column": target_column,
            "model_weights": model_weights,
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
        child_endpoints: list[str],
        description: str,
        tags: list[str],
        estimator: Estimator,
        aws_clamp: AWSAccountClamp,
    ):
        """Create model group and register the model.

        Args:
            name: Model name
            child_endpoints: List of endpoint names
            description: Model description
            tags: Model tags
            estimator: Fitted estimator
            aws_clamp: AWS account clamp
        """
        sm_session = aws_clamp.sagemaker_session()
        model_description = description or f"Meta model aggregating: {', '.join(child_endpoints)}"

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
        cls, name: str, target_column: str, feature_list: list[str], feature_set_name: str, child_endpoints: list[str]
    ):
        """Set model metadata and onboard.

        Args:
            name: Model name
            target_column: Target column name
            feature_list: List of feature names
            feature_set_name: Name of the input FeatureSet
            child_endpoints: List of child endpoint names
        """
        time.sleep(3)
        output_model = ModelCore(name)
        output_model._set_model_type(ModelType.UQ_REGRESSOR)
        output_model._set_model_framework(ModelFramework.META)
        output_model.set_input(feature_set_name, force=True)
        output_model.upsert_workbench_meta({"workbench_model_target": target_column})
        output_model.upsert_workbench_meta({"workbench_model_features": feature_list})
        output_model.upsert_workbench_meta({"child_endpoints": child_endpoints})
        output_model.onboard_with_args(ModelType.UQ_REGRESSOR, target_column, feature_list=feature_list)


if __name__ == "__main__":
    """Exercise the MetaModel Class"""

    meta = MetaModel.create(
        name="logd-meta",
        child_endpoints=["logd-xgb", "logd-pytorch", "logd-chemprop"],
        target_column="logd",
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
