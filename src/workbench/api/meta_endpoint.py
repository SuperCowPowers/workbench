"""MetaEndpoint: An Endpoint backed by a directed acyclic graph (DAG) of
child endpoints and aggregation nodes.

A MetaEndpoint behaves identically to a regular Endpoint at runtime —
callers do ``endpoint.inference(df)`` and get a DataFrame back. The DAG
machinery is server-side: the deployed container loads the serialized
DAG, dispatches each child invocation to ``fast_inference`` (sync) or
``async_inference`` (async), and runs aggregation nodes locally.

Common usage::

    from workbench.api import MetaEndpoint
    from workbench.utils.meta_endpoint_dag import MetaEndpointDAG
    from workbench.utils.aggregation_nodes import Concat

    dag = MetaEndpointDAG()
    dag.add_endpoint("smiles-to-2d-v1")
    dag.add_endpoint("smiles-to-3d-fast-v1")
    dag.add_aggregation(Concat(name="combine"))
    dag.add_edge("smiles-to-2d-v1", "combine")
    dag.add_edge("smiles-to-3d-fast-v1", "combine")
    dag.set_input_node("smiles-to-2d-v1", "smiles-to-3d-fast-v1")
    dag.set_output_node("combine")

    end = MetaEndpoint.create(name="my-features-meta", dag=dag)
    df = end.inference(input_df)

If any child endpoint in the DAG is async (e.g. ``smiles-to-3d-full-v1``),
the MetaEndpoint is automatically deployed as async too — its invocation
budget needs to accommodate the slowest child.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from sagemaker.core.resources import ModelPackage, ModelPackageGroup, TrainingJob
from sagemaker.core.shapes.model_card_shapes import ContainersItem, InferenceSpecification
from sagemaker.core.training.configs import Compute, OutputDataConfig, SourceCode
from sagemaker.train.model_trainer import ModelTrainer

from workbench.api.endpoint import Endpoint
from workbench.api.model import Model
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.model_core import ModelCore, ModelFramework, ModelImages, ModelType
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.model_scripts.script_generation import generate_model_script
from workbench.utils.config_manager import ConfigManager
from workbench.utils.meta_endpoint_dag import MetaEndpointDAG
from workbench.utils.workbench_logging import _suppress_sagemaker_logging

log = logging.getLogger("workbench")


class MetaEndpoint(Endpoint):
    """Endpoint backed by a :class:`MetaEndpointDAG`.

    Constructor wraps an existing deployed MetaEndpoint by name, identical
    to :class:`Endpoint`. Use :meth:`create` to build and deploy a new one
    from a DAG.
    """

    @classmethod
    def create(
        cls,
        name: str,
        dag: MetaEndpointDAG,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> "MetaEndpoint":
        """Build, register, and deploy a MetaEndpoint from a DAG.

        Steps:
          1. Validate the DAG; populate per-endpoint async flags.
          2. Backtrace lineage from a primary endpoint to satisfy
             Workbench's Model machinery (FeatureSet, target, features).
          3. Run a SageMaker training job that writes the DAG JSON +
             runtime config as the model artifact.
          4. Register a Model package with the meta inference container.
          5. Deploy the endpoint (async if any DAG child is async).

        Args:
            name: Endpoint / Model name.
            dag: A :class:`MetaEndpointDAG` describing the data flow.
            description: Optional description for the registered model.
            tags: Optional list of Workbench tags.

        Returns:
            The deployed MetaEndpoint, ready for ``.inference()``.
        """
        Artifact.is_name_valid(name, delimiter="-", lower_case=False)

        log.important(f"Validating DAG for MetaEndpoint '{name}'...")
        dag.validate()
        dag.populate_async_flags()
        is_async = dag.has_async_endpoint()
        log.important(
            f"DAG: {len(dag._endpoints)} endpoints, {len(dag._aggregations)} aggregation nodes "
            f"({'async' if is_async else 'sync'} deployment)"
        )

        # Backtrace lineage from a primary endpoint to satisfy Workbench Model
        # machinery (every Model needs a FeatureSet to hang off of).
        feature_list, feature_set_name, target_column = cls._derive_lineage(dag)

        log.important(f"Trying to delete existing model {name}...")
        ModelCore.managed_delete(name)

        aws_clamp = AWSAccountClamp()
        training_job_name = cls._run_training(name, dag, aws_clamp)
        cls._register_model(name, dag, description, tags, training_job_name, aws_clamp)
        cls._set_metadata(name, dag, target_column, feature_list, feature_set_name)

        log.important(f"Deploying MetaEndpoint '{name}' ({'async' if is_async else 'sync'})...")
        model = Model(name)
        model.to_endpoint(
            tags=tags or [name],
            async_endpoint=is_async,
        )

        log.important(f"MetaEndpoint '{name}' created successfully!")
        return cls(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _derive_lineage(cls, dag: MetaEndpointDAG) -> tuple[list[str], str, str | None]:
        """Backtrace from the first input endpoint to find a FeatureSet + lineage.

        Workbench Models need to trace back to a FeatureSet. For DAG-based
        MetaEndpoints there isn't a single canonical FeatureSet, so we use
        the first input node's lineage as a representative anchor.

        Returns ``(feature_list, feature_set_name, target_column)``.
        ``target_column`` may be ``None`` for pure feature-pipeline DAGs
        whose primary endpoint is a feature endpoint.
        """
        if not dag._input_nodes:
            raise ValueError("DAG has no input nodes — cannot derive lineage")
        primary_endpoint_name = dag._endpoints[dag._input_nodes[0]]

        ep = Endpoint(primary_endpoint_name)
        if not ep.exists():
            raise ValueError(f"Primary endpoint '{primary_endpoint_name}' does not exist")

        primary_model = Model(ep.get_input())
        feature_list = primary_model.features() or list(dag.input_columns())
        feature_set_name = primary_model.get_input()
        target_column = primary_model.target()

        log.info(
            f"Lineage anchor: {primary_endpoint_name} -> {primary_model.name} -> {feature_set_name} "
            f"(target: {target_column})"
        )
        return feature_list, feature_set_name, target_column

    @classmethod
    def _run_training(cls, name: str, dag: MetaEndpointDAG, aws_clamp: AWSAccountClamp) -> str:
        """Run the training job that persists the DAG JSON as a model artifact."""
        sm_session = aws_clamp.sagemaker_session()
        cm = ConfigManager()
        workbench_bucket = cm.get_config("WORKBENCH_BUCKET")
        models_s3_path = f"s3://{workbench_bucket}/models"

        template_params = {
            "model_type": ModelType.REGRESSOR,
            "model_framework": ModelFramework.META,
            "dag_json": dag.to_json(),
            "aws_region": sm_session.boto_region_name,
            "s3_bucket": workbench_bucket,
            "model_metrics_s3_path": f"{models_s3_path}/{name}/training",
        }
        script_path = generate_model_script(template_params)

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
            output_data_config=OutputDataConfig(
                s3_output_path=f"{models_s3_path}/{name}/training", compression_type="GZIP"
            ),
            role=aws_clamp.aws_session.get_workbench_execution_role_arn(),
            sagemaker_session=sm_session,
            base_job_name=name,
        )

        log.important(f"Running training job for MetaEndpoint {name}...")
        _suppress_sagemaker_logging()
        trainer.train(wait=True)

        return trainer._latest_training_job.training_job_name

    @classmethod
    def _register_model(
        cls,
        name: str,
        dag: MetaEndpointDAG,
        description: str | None,
        tags: list[str] | None,
        training_job_name: str,
        aws_clamp: AWSAccountClamp,
    ) -> None:
        """Create model group + register the model package with the meta inference image."""
        sm_session = aws_clamp.sagemaker_session()
        boto3_session = aws_clamp.boto3_session
        endpoint_names = list(dag._endpoints.values())
        model_description = description or f"MetaEndpoint DAG over: {', '.join(endpoint_names)}"

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

        training_job = TrainingJob.get(training_job_name, session=boto3_session)
        model_data_url = training_job.model_artifacts.s3_model_artifacts

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
        dag: MetaEndpointDAG,
        target_column: str | None,
        feature_list: list[str],
        feature_set_name: str,
    ) -> None:
        """Populate workbench_meta with DAG-derived fields for downstream introspection."""
        # Brief delay to let the model package settle before we read it back.
        time.sleep(3)
        output_model = ModelCore(name)
        output_model._set_model_type(ModelType.REGRESSOR)
        output_model._set_model_framework(ModelFramework.META)
        if feature_set_name:
            output_model.set_input(feature_set_name, force=True)
        if target_column:
            output_model.upsert_workbench_meta({"workbench_model_target": target_column})
        if feature_list:
            output_model.upsert_workbench_meta({"workbench_model_features": feature_list})
        output_model.upsert_workbench_meta({"endpoints": list(dag._endpoints.values())})
        output_model.upsert_workbench_meta({"meta_endpoint_dag": dag.to_dict()})
        output_model.onboard_with_args(ModelType.REGRESSOR, target_column, feature_list=feature_list)

    def get_dag(self) -> MetaEndpointDAG:
        """Reconstruct the MetaEndpointDAG from this endpoint's stored metadata."""
        meta = self.workbench_meta() or {}
        dag_dict = meta.get("meta_endpoint_dag")
        if not dag_dict:
            raise ValueError(
                f"MetaEndpoint '{self.name}' has no DAG in workbench_meta. "
                f"Recreate via MetaEndpoint.create()."
            )
        if isinstance(dag_dict, str):
            return MetaEndpointDAG.from_json(dag_dict)
        return MetaEndpointDAG.from_dict(dag_dict)


if __name__ == "__main__":
    """Exercise the MetaEndpoint Class"""
    from workbench.utils.aggregation_nodes import Concat

    sample_dag = MetaEndpointDAG()
    sample_dag.add_endpoint("smiles-to-2d-v1")
    sample_dag.add_endpoint("smiles-to-3d-fast-v1")
    sample_dag.add_aggregation(Concat(name="combine"))
    sample_dag.add_edge("smiles-to-2d-v1", "combine")
    sample_dag.add_edge("smiles-to-3d-fast-v1", "combine")
    sample_dag.set_input_node("smiles-to-2d-v1", "smiles-to-3d-fast-v1")
    sample_dag.set_output_node("combine")

    sample_end = MetaEndpoint.create(
        name="meta-2d-3d-features",
        dag=sample_dag,
        description="2D + 3D features merged",
        tags=["meta", "features"],
    )
    print(sample_end.summary())
