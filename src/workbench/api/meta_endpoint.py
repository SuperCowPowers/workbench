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
    # Input does not need any id column — the DAG handles row alignment internally.
    df = end.inference(input_df)

If any child endpoint in the DAG is async (e.g. ``smiles-to-3d-full-v1``),
the MetaEndpoint is automatically deployed as async too — its invocation
budget needs to accommodate the slowest child.
"""

from __future__ import annotations

import logging

import pandas as pd

from workbench.api.endpoint import Endpoint
from workbench.api.feature_set import FeatureSet
from workbench.api.model import Model
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.model_core import ModelFramework, ModelType
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.meta_endpoint_dag import MetaEndpointDAG

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
          3. Run the standard ``FeatureSet.to_model()`` flow, passing the
             DAG / region / bucket as ``custom_args`` so the meta-endpoint
             template fills them in at training time.
          4. Set DAG-specific ``workbench_meta`` keys on the resulting Model.
          5. Deploy the endpoint (async if any DAG child is async).

        Args:
            name: Endpoint / Model name.
            dag: A :class:`MetaEndpointDAG` describing the data flow.
            description: Optional description for the registered model.
            tags: Optional list of Workbench tags.

        Returns:
            The deployed MetaEndpoint, ready for ``.inference()``.
        """
        if not Artifact.is_name_valid(name, delimiter="-", lower_case=False):
            raise ValueError(f"Invalid MetaEndpoint name: '{name}' (use only alphanumerics and '-')")

        log.important(f"Validating DAG for MetaEndpoint '{name}'...")
        dag.validate()
        dag.populate_async_flags()
        is_async = dag.has_async_endpoint()
        log.important(
            f"DAG: {len(dag.endpoints)} endpoints, {len(dag.aggregations)} aggregation nodes "
            f"({'async' if is_async else 'sync'} deployment)"
        )

        # Backtrace lineage from a primary endpoint to satisfy Workbench Model
        # machinery (every Model needs a FeatureSet to hang off of).
        feature_list, feature_set_name, target_column = cls._derive_lineage(dag)

        # Build the model via the standard FeatureSet → Model flow. The
        # meta-endpoint template's `{{dag}}`, `{{aws_region}}`, `{{s3_bucket}}`
        # placeholders are filled from custom_args.
        aws_clamp = AWSAccountClamp()
        sm_session = aws_clamp.sagemaker_session()
        workbench_bucket = ConfigManager().get_config("WORKBENCH_BUCKET")

        feature_set = FeatureSet(feature_set_name)
        feature_set.to_model(
            name=name,
            model_type=cls._derive_model_type(dag),
            model_framework=ModelFramework.META,
            tags=tags or [name],
            description=description or f"MetaEndpoint DAG over: {', '.join(dag.endpoints.values())}",
            target_column=target_column,
            feature_list=feature_list,
            custom_args={
                "dag": dag.to_dict(),
                "aws_region": sm_session.boto_region_name,
                "s3_bucket": workbench_bucket,
            },
        )

        # MetaEndpoint containers are I/O-bound orchestrators (S3 staging,
        # child invocations, S3 polling) — so max_instances=1 and concurrency=32.
        log.important(f"Deploying MetaEndpoint '{name}' ({'async' if is_async else 'sync'})...")
        model = Model(name)
        if is_async:
            endpoint = model.to_endpoint(
                tags=tags or [name],
                async_endpoint=True,
                max_instances=1,
                scale_in_idle_minutes=5,
                async_max_concurrent=32,
            )
        else:
            endpoint = model.to_endpoint(
                tags=tags or [name],
                async_endpoint=False,
            )

        # Auto-derive inference_batch_size from the smallest tolerance among
        # children — chunks the meta receives from SageMaker get fanned out
        # as-is to every child, so the meta's chunk size shouldn't exceed
        # the smallest child's batch size.
        min_batch = dag.min_child_batch_size()

        # Publish the largest child fleet size as effective_max_instances so
        # callers (e.g. InferenceCache) can size their work units to fill
        # downstream child capacity rather than the meta's own
        # max_instances=1 (which only describes the orchestrator layer).
        effective_max = dag.max_child_max_instances()

        # The DAG dict is the runtime artifact get_dag() / run_dag_test()
        # rehydrate from. Stored on the endpoint (not the model) since it
        # describes inference-time orchestration, not training.
        endpoint.upsert_workbench_meta(
            {
                "inference_batch_size": min_batch,
                "effective_max_instances": effective_max,
                "meta_endpoint_dag": dag.to_dict(),
            }
        )
        log.important(f"Set inference_batch_size={min_batch} (min across DAG children)")
        log.important(f"Set effective_max_instances={effective_max} (max across DAG children)")

        log.important(f"MetaEndpoint '{name}' created successfully!")
        return cls(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _derive_model_type(cls, dag: MetaEndpointDAG) -> ModelType:
        """Pick the most accurate :class:`ModelType` for the DAG's output.

        - Output node is a terminal endpoint → borrow that endpoint's declared
          type (e.g., a downstream predictor endpoint contributes its own type).
        - Output node is :class:`~workbench.utils.aggregation_nodes.Concat` →
          ``TRANSFORMER`` (column-union of feature outputs).
        - Output node is :class:`~workbench.utils.aggregation_nodes.Vote` →
          ``CLASSIFIER`` (majority vote of class labels).
        - Output node is any other prediction aggregator → ``REGRESSOR``.
        """
        from workbench.utils.aggregation_nodes import Concat, Vote

        output_name = dag.output_node
        if output_name in dag.endpoints:
            return Model(dag.endpoints[output_name]).model_type

        agg = dag.aggregations[output_name]
        if isinstance(agg, Concat):
            return ModelType.TRANSFORMER
        if isinstance(agg, Vote):
            return ModelType.CLASSIFIER
        return ModelType.REGRESSOR

    @classmethod
    def _derive_lineage(cls, dag: MetaEndpointDAG) -> tuple[list[str], str, str | None]:
        """Derive a (feature_list, feature_set_name, target_column) tuple for the meta.

        Workbench Models need to trace back to a FeatureSet, and downstream
        tooling (``auto_inference``, ``register_input_columns`` /
        ``register_output_columns``) all anchor on
        ``model.features()`` + ``model.get_input()``. For DAG-based
        MetaEndpoints we satisfy that contract by:

          - ``feature_list`` / ``feature_set_name`` — borrowed from the first
            input endpoint, since the meta consumes that endpoint's input
            columns and that endpoint's FeatureSet is guaranteed to contain
            usable smoke-test data.
          - ``target_column`` — :meth:`MetaEndpointDAG.terminal_target`,
            which walks back from the output node to the closest predictor
            endpoint(s) so mixed DAGs (smiles → features → predictor) report
            what the meta actually predicts, not what the input endpoint
            happens to produce.
        """
        if not dag.input_nodes:
            raise ValueError("DAG has no input nodes — cannot derive lineage")
        primary_endpoint_name = dag.endpoints[dag.input_nodes[0]]

        ep = Endpoint(primary_endpoint_name)
        if not ep.exists():
            raise ValueError(f"Primary endpoint '{primary_endpoint_name}' does not exist")

        primary_model = Model(ep.get_input())
        feature_list = primary_model.features() or list(dag.input_columns())
        feature_set_name = primary_model.get_input()
        # Target reflects what the DAG ultimately predicts (walks back from
        # the output node), not what the input endpoint happens to produce.
        # For mixed DAGs (smiles → features → predictor), this surfaces the
        # downstream predictor's target instead of the feature endpoint's None.
        target_column = dag.terminal_target()

        log.info(
            f"Lineage anchor: {primary_endpoint_name} -> {primary_model.name} -> {feature_set_name} "
            f"(target: {target_column})"
        )
        return feature_list, feature_set_name, target_column

    def get_dag(self) -> MetaEndpointDAG:
        """Reconstruct the MetaEndpointDAG from this endpoint's stored metadata."""
        meta = self.workbench_meta() or {}
        dag_dict = meta.get("meta_endpoint_dag")
        if not dag_dict:
            raise ValueError(
                f"MetaEndpoint '{self.name}' has no DAG in workbench_meta. Recreate via MetaEndpoint.create()."
            )
        return MetaEndpointDAG.from_dict(dag_dict)

    def run_dag_test(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Execute the DAG client-side against ``input_df``.

        Bypasses the deployed meta endpoint entirely: each child endpoint is
        invoked directly via the regular ``Endpoint(name).inference()`` API
        from this process. Useful for debugging the DAG topology, isolating
        which child is misbehaving, or running the DAG when the deployed
        meta endpoint is unavailable.

        Result is identical to ``self.inference(input_df)`` modulo transport
        and any container-only side effects (data capture, etc.).
        """
        return self.get_dag().run(input_df)


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
