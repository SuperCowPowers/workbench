"""ModelToEndpoint: Deploy an Endpoint for a Model

FIXME: Investigate using V3's ModelBuilder for deployment instead of manually creating
Model/EndpointConfig/Endpoint resources. ModelBuilder handles inference code bundling,
SAGEMAKER_PROGRAM env var, and model artifact repacking automatically. May eliminate
the need for our inference-metadata.json workaround. Need to verify it works with
our custom inference Docker images (which use their own main.py, not SageMaker's
built-in serving stack). See https://sagemaker.readthedocs.io/en/stable/inference/index.html
"""

import time
from botocore.exceptions import ClientError

# SageMaker V3 Resource Classes
from sagemaker.core.resources import Model as SagemakerModel, EndpointConfig, Endpoint as SagemakerEndpoint
from sagemaker.core.shapes.shapes import (
    AsyncInferenceClientConfig,
    AsyncInferenceConfig,
    AsyncInferenceOutputConfig,
    ContainerDefinition,
    ProductionVariant,
    ProductionVariantServerlessConfig,
    DataCaptureConfig as DataCaptureConfigShape,
    CaptureOption,
    Tag,
)

# FIXME: sagemaker-core 2.6.0 is missing snake_to_pascal mapping for 'memory_size_in_mb'.
# They have 'volume_size_in_gb' -> 'VolumeSizeInGB' but forgot the MB equivalent.
# Patching at import time. See https://github.com/aws/sagemaker-core/issues/225
from sagemaker.core.utils.utils import SPECIAL_SNAKE_TO_PASCAL_MAPPINGS

SPECIAL_SNAKE_TO_PASCAL_MAPPINGS["memory_size_in_mb"] = "MemorySizeInMB"

# Local Imports (after monkey-patch above)
from workbench.core.transforms.transform import Transform, TransformInput, TransformOutput  # noqa: E402
from workbench.core.artifacts.model_core import ModelCore  # noqa: E402
from workbench.core.artifacts.endpoint_core import EndpointCore  # noqa: E402
from workbench.core.artifacts.artifact import Artifact  # noqa: E402
from workbench.utils.endpoint_autoscaling import register_autoscaling  # noqa: E402


class ModelToEndpoint(Transform):
    """ModelToEndpoint: Deploy an Endpoint for a Model

    Common Usage:
        ```python
        to_endpoint = ModelToEndpoint(model_name, endpoint_name)
        to_endpoint.set_output_tags(["aqsol", "public", "whatever"])
        to_endpoint.transform()
        ```
    """

    def __init__(
        self,
        model_name: str,
        endpoint_name: str,
        serverless: bool = True,
        instance: str = None,
        async_endpoint: bool = False,
        max_instances: int = None,
    ):
        """ModelToEndpoint Initialization
        Args:
            model_name(str): The Name of the input Model
            endpoint_name(str): The Name of the output Endpoint
            serverless(bool): Deploy the Endpoint in serverless mode (default: True)
            instance(str): The instance type for Realtime Endpoints (default: None = auto-select)
            async_endpoint(bool): Deploy as an async endpoint (default: False). Async
                endpoints support up to 15-minute invocations and use S3 for I/O.
                Incompatible with serverless — if both are True, serverless is forced off.
            max_instances(int): Autoscaler upper bound for async endpoints (default: None =
                use register_autoscaling's default of 8). Ignored for realtime endpoints.
        """
        # Make sure the endpoint_name is a valid name
        Artifact.is_name_valid(endpoint_name, delimiter="-", lower_case=False)

        # Call superclass init
        super().__init__(model_name, endpoint_name)

        # Async endpoints are always realtime (not serverless)
        if async_endpoint and serverless:
            self.log.warning("Async endpoints are not compatible with serverless. Forcing serverless=False.")
            serverless = False

        # Set up all my instance attributes
        self.serverless = serverless
        self.instance = instance
        self.async_endpoint = async_endpoint
        self.max_instances = max_instances
        self.input_type = TransformInput.MODEL
        self.output_type = TransformOutput.ENDPOINT

    def transform_impl(self, **kwargs):
        """Deploy an Endpoint for a Model"""

        # Delete endpoint (if it already exists)
        EndpointCore.managed_delete(self.output_name)

        # Get the Model Package ARN for our input model
        workbench_model = ModelCore(self.input_name)

        # Deploy the model
        self._deploy_model(workbench_model, **kwargs)

        # Add this endpoint to the set of registered endpoints for the model
        workbench_model.register_endpoint(self.output_name)

        # This ensures that the endpoint is ready for use
        time.sleep(5)  # We wait for AWS Lag
        end = EndpointCore(self.output_name)
        self.log.important(f"Endpoint {end.name} is ready for use")

    def _deploy_model(
        self,
        workbench_model: ModelCore,
        mem_size: int = 2048,
        max_concurrency: int = 5,
        data_capture: bool = False,
        capture_percentage: int = 100,
    ):
        """Internal Method: Deploy the Model

        Args:
            workbench_model(ModelCore): The Workbench ModelCore object to deploy
            mem_size(int): Memory size for serverless deployment
            max_concurrency(int): Max concurrency for serverless deployment
            data_capture(bool): Enable data capture during deployment
            capture_percentage(int): Percentage of data to capture. Defaults to 100.
        """
        # Grab the specified Model Package ARN and inference image
        model_package_arn = workbench_model.model_package_arn()
        inference_image = workbench_model.container_image()
        self.log.important(f"Deploying Model Package: {self.input_name} with Inference Image: {inference_image}")

        # Get the metadata/tags to push into AWS
        aws_tags = self.get_aws_tags()
        sagemaker_tags = [Tag(key=t["key"], value=t["value"]) for t in aws_tags]

        # Check the model framework for resource requirements
        from workbench.api import ModelFramework

        self.log.info(f"Model Framework: {workbench_model.model_framework}")
        needs_more_resources = workbench_model.model_framework in [ModelFramework.PYTORCH, ModelFramework.CHEMPROP]

        # Determine serverless config and instance type
        serverless_config = None
        if self.serverless:
            # For PyTorch or ChemProp we need at least 4GB of memory
            if needs_more_resources and mem_size < 4096:
                self.log.important(f"{workbench_model.model_framework} needs at least 4GB of memory (setting to 4GB)")
                mem_size = 4096
            serverless_config = ProductionVariantServerlessConfig(
                memory_size_in_mb=mem_size,
                max_concurrency=max_concurrency,
            )
            instance_type = None  # Not used for serverless
            self.log.important(f"Serverless Config: Memory={mem_size}MB, MaxConcurrency={max_concurrency}")
        else:
            # Use explicit instance if provided, otherwise auto-select.
            # Async endpoints default to a beefier CPU instance — they're typically
            # used for long-running compute work (RDKit conformer gen, etc.) where
            # the default realtime sizing would be undersized.
            if self.instance:
                instance_type = self.instance
                self.log.important(f"Endpoint: Using specified instance type: {instance_type}")
            elif self.async_endpoint:
                instance_type = "ml.c7i.xlarge"
                self.log.important(f"Async Endpoint: Default instance type: {instance_type}")
            elif needs_more_resources:
                instance_type = "ml.c7i.large"
                self.log.important(f"{workbench_model.model_framework} needs more resources (using {instance_type})")
            else:
                instance_type = "ml.t2.medium"
                self.log.important(f"Realtime Endpoint: Instance Type={instance_type}")

        # Configure data capture if requested (and not serverless)
        data_capture_config = None
        if data_capture and not self.serverless:
            # Set up the S3 path for data capture
            base_endpoint_path = f"{workbench_model.endpoints_s3_path}/{self.output_name}"
            data_capture_path = f"{base_endpoint_path}/data_capture"
            self.log.important(f"Configuring Data Capture --> {data_capture_path}")
            data_capture_config = DataCaptureConfigShape(
                enable_capture=True,
                initial_sampling_percentage=capture_percentage,
                destination_s3_uri=data_capture_path,
                capture_options=[
                    CaptureOption(capture_mode="Input"),
                    CaptureOption(capture_mode="Output"),
                ],
            )
        elif data_capture and self.serverless:
            self.log.warning(
                "Data capture is not supported for serverless endpoints. Skipping data capture configuration."
            )

        # For async endpoints, resolve the per-instance concurrency knob.
        # Default 1: typical async workloads (RDKit conformers, ML inference)
        # already saturate the CPU per invocation, so dispatching multiple in
        # parallel just causes context-switching overhead. Backlog also grows
        # faster which makes autoscaling trigger sooner.
        # Override per-model via workbench_meta["async_max_concurrent_per_instance"]
        # for IO-bound or lightweight async workloads.
        async_max_concurrent = None
        if self.async_endpoint:
            model_meta = workbench_model.workbench_meta() or {}
            async_max_concurrent = int(model_meta.get("async_max_concurrent_per_instance", 1))
            self.log.important(f"Async MaxConcurrentInvocationsPerInstance={async_max_concurrent}")

        # Deploy the Endpoint using V3 Resource Classes
        self.log.important(f"Deploying the Endpoint {self.output_name}...")
        try:
            self._create_endpoint_resources(
                model_package_arn=model_package_arn,
                serverless_config=serverless_config,
                instance_type=instance_type,
                data_capture_config=data_capture_config,
                tags=sagemaker_tags,
                async_max_concurrent=async_max_concurrent,
            )
        except ClientError as e:
            # Check if this is the "endpoint config already exists" error
            if "Cannot create already existing endpoint configuration" in str(e):
                self.log.warning("Endpoint config already exists, deleting and retrying...")
                EndpointConfig.get(self.output_name, session=self.boto3_session).delete()
                # Retry
                self._create_endpoint_resources(
                    model_package_arn=model_package_arn,
                    serverless_config=serverless_config,
                    instance_type=instance_type,
                    data_capture_config=data_capture_config,
                    tags=sagemaker_tags,
                    async_max_concurrent=async_max_concurrent,
                )
            else:
                raise

    def _create_endpoint_resources(
        self,
        model_package_arn: str,
        serverless_config=None,
        instance_type: str = None,
        data_capture_config=None,
        tags=None,
        async_max_concurrent: int = None,
    ):
        """Internal: Create the SageMaker Model, EndpointConfig, and Endpoint resources.

        Args:
            model_package_arn (str): The model package ARN to deploy
            serverless_config: ServerlessConfig for serverless deployments
            instance_type (str): Instance type for realtime deployments
            data_capture_config: Data capture configuration
            tags: List of Tag objects
            async_max_concurrent (int): MaxConcurrentInvocationsPerInstance for async endpoints.
                Only used when ``self.async_endpoint`` is True.
        """
        model_name = self.output_name
        config_name = self.output_name

        # Step 1: Create the SageMaker Model from the Model Package
        container = ContainerDefinition(model_package_name=model_package_arn)
        try:
            SagemakerModel.create(
                model_name=model_name,
                primary_container=container,
                execution_role_arn=self.workbench_role_arn,
                tags=tags,
                session=self.boto3_session,
            )
        except ClientError as e:
            if "Cannot create already existing model" in str(e):
                self.log.warning("Model already exists, deleting and recreating...")
                SagemakerModel.get(model_name, session=self.boto3_session).delete()
                SagemakerModel.create(
                    model_name=model_name,
                    primary_container=container,
                    execution_role_arn=self.workbench_role_arn,
                    tags=tags,
                    session=self.boto3_session,
                )
            else:
                raise

        # Step 2: Create the EndpointConfig
        production_variant = ProductionVariant(
            variant_name="AllTraffic",
            model_name=model_name,
            initial_variant_weight=1.0,
        )
        if serverless_config:
            production_variant.serverless_config = serverless_config
        else:
            production_variant.initial_instance_count = 1
            production_variant.instance_type = instance_type
            production_variant.container_startup_health_check_timeout_in_seconds = 300

        # Build async inference config if requested
        async_inference_config = None
        if self.async_endpoint:
            base_path = f"{self.endpoints_s3_path}/{self.output_name}"
            async_inference_config = AsyncInferenceConfig(
                output_config=AsyncInferenceOutputConfig(
                    s3_output_path=f"{base_path}/async-output",
                    s3_failure_path=f"{base_path}/async-failures",
                ),
                client_config=AsyncInferenceClientConfig(
                    max_concurrent_invocations_per_instance=async_max_concurrent,
                ),
            )
            self.log.important(f"Async Endpoint Config: output → {base_path}/async-output")

        EndpointConfig.create(
            endpoint_config_name=config_name,
            production_variants=[production_variant],
            async_inference_config=async_inference_config,
            data_capture_config=data_capture_config,
            tags=tags,
            session=self.boto3_session,
        )

        # Step 3: Create the Endpoint and wait for it to be InService
        endpoint = SagemakerEndpoint.create(
            endpoint_name=self.output_name,
            endpoint_config_name=config_name,
            tags=tags,
            session=self.boto3_session,
        )
        endpoint.wait_for_status("InService")

        # For async endpoints, register a scale-to-zero auto-scaling policy.
        # This must be done after the endpoint is InService — AWS doesn't
        # allow managed instance scaling on the ProductionVariant for async configs.
        if self.async_endpoint:
            autoscale_kwargs = {}
            if self.max_instances is not None:
                autoscale_kwargs["max_capacity"] = self.max_instances
            register_autoscaling(self.boto3_session, self.output_name, **autoscale_kwargs)

    def post_transform(self, **kwargs):
        """Post-Transform: Calling onboard() for the Endpoint"""
        self.log.info("Post-Transform: Calling onboard() for the Endpoint...")

        # Onboard the Endpoint
        output_endpoint = EndpointCore(self.output_name)
        output_endpoint.onboard_with_args(input_model=self.input_name)

        # Persist deploy-time sizing config to workbench_meta so later code
        # (and operators) can see what the endpoint was deployed with.
        # Only non-None values are stored — no point cluttering meta with defaults.
        deploy_meta = {
            k: v for k, v in {
                "instance": self.instance,
                "max_instances": self.max_instances,
                "async_endpoint": self.async_endpoint,
                "serverless": self.serverless,
            }.items() if v is not None
        }
        if deploy_meta:
            output_endpoint.upsert_workbench_meta(deploy_meta)


if __name__ == "__main__":
    """Exercise the ModelToEndpoint Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_name = "abalone-regression"
    output_name = f"{input_name}"
    to_endpoint = ModelToEndpoint(input_name, output_name, serverless=True)
    to_endpoint.set_output_tags(["abalone", "public"])
    to_endpoint.transform()
