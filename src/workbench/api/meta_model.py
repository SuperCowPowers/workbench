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
        feature_list, feature_set_name = cls._validate_and_get_lineage(child_endpoints)

        # Delete existing model if it exists
        log.important(f"Trying to delete existing model {name}...")
        ModelCore.managed_delete(name)

        # Run training and register model
        aws_clamp = AWSAccountClamp()
        sm_session = aws_clamp.sagemaker_session()
        estimator = cls._run_training(name, child_endpoints, target_column, aws_clamp, sm_session)
        cls._register_model(name, child_endpoints, description, tags, estimator, aws_clamp, sm_session)

        # Set metadata and onboard
        cls._set_metadata(name, target_column, feature_list, feature_set_name, child_endpoints)

        log.important(f"MetaModel {name} created successfully!")
        return cls(name)

    @classmethod
    def _validate_and_get_lineage(cls, child_endpoints: list[str]) -> tuple[list[str], str]:
        """Validate child endpoints exist and get lineage info from primary endpoint.

        Args:
            child_endpoints: List of endpoint names

        Returns:
            tuple: (feature_list, feature_set_name) from the primary endpoint's model
        """
        log.info("Verifying child endpoints...")
        for ep_name in child_endpoints:
            ep = Endpoint(ep_name)
            if not ep.exists():
                raise ValueError(f"Child endpoint '{ep_name}' does not exist")

        # Use first endpoint as primary - backtrack to get model and feature set
        primary_endpoint = Endpoint(child_endpoints[0])
        primary_model = Model(primary_endpoint.get_input())
        feature_list = primary_model.features()
        feature_set_name = primary_model.get_input()

        log.info(
            f"Primary endpoint: {child_endpoints[0]} -> Model: {primary_model.name} -> FeatureSet: {feature_set_name}"
        )
        return feature_list, feature_set_name

    @classmethod
    def _run_training(
        cls, name: str, child_endpoints: list[str], target_column: str, aws_clamp: AWSAccountClamp, sm_session
    ) -> Estimator:
        """Run the minimal training job that saves the meta model config.

        Args:
            name: Model name
            child_endpoints: List of endpoint names
            target_column: Target column name
            aws_clamp: AWS account clamp
            sm_session: SageMaker session

        Returns:
            Estimator: The fitted estimator
        """
        cm = ConfigManager()
        workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
        models_s3_path = f"s3://{workbench_bucket}/models"

        # Generate the model script from template
        template_params = {
            "model_type": ModelType.REGRESSOR,
            "model_framework": ModelFramework.META,
            "child_endpoints": child_endpoints,
            "target_column": target_column,
            "model_metrics_s3_path": f"{models_s3_path}/{name}/training",
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
        sm_session,
    ):
        """Create model group and register the model.

        Args:
            name: Model name
            child_endpoints: List of endpoint names
            description: Model description
            tags: Model tags
            estimator: Fitted estimator
            aws_clamp: AWS account clamp
            sm_session: SageMaker session
        """
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
    end.auto_inference(capture=True)

    # Test loading an existing meta model
    meta = MetaModel("logd-meta")
    print(meta.details())
