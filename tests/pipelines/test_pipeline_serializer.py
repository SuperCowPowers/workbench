"""Tests for pipeline_serializer: node-link serialization + job-id uniqueness.

The DAG is built in-memory (PipelineManager.from_jobs), serialized, and then collapsed
the same way the dashboard renderer does (jobs -> artifact->artifact edges). The focus
is the invariant that distinct jobs never merge -- a job's human node_id can repeat when
one script drives several stages, and the UI merges several pipelines into one card and
dedups nodes by id, so job ids must be unique both within and across pipelines. A
collision silently cross-wires inputs to outputs.
"""

import pytest

from workbench.lambda_layer.pipeline_manager import Job, PipelineManager
from workbench.utils.pipeline_serializer import _serialize, linearize


def _job(script, inputs=None, outputs=None, pipeline=None):
    return Job(script, None, outputs or [], inputs or [], pipeline)


def _collapse(node_link):
    """Mirror the renderer: collapse job nodes into artifact -> artifact edges."""
    is_job = {n["id"] for n in node_link["nodes"] if n["kind"] == "job"}
    inbound, outbound = {}, {}
    for link in node_link["links"]:
        if link["target"] in is_job:
            inbound.setdefault(link["target"], []).append(link["source"])
        if link["source"] in is_job:
            outbound.setdefault(link["source"], []).append(link["target"])
    edges = set()
    for j in is_job:
        for a in inbound.get(j, []):
            for b in outbound.get(j, []):
                if a != b:
                    edges.add((a, b))
    return edges


def _merge(node_links):
    """Mirror the renderer's mergeGraphs: dedup nodes by id, concatenate links."""
    nodes, links = {}, []
    for nl in node_links:
        for n in nl["nodes"]:
            nodes.setdefault(n["id"], n)
        links += nl["links"]
    return {"nodes": list(nodes.values()), "links": links}


def _serialized(jobs, name):
    return _serialize(PipelineManager.from_jobs(jobs).get_pipeline(name))


def test_job_ids_are_unique_when_one_script_drives_every_stage():
    """A single script chained ds->fs->model->endpoint must not merge its jobs."""
    jobs = [
        _job("chain.py", inputs=["ds:d"], outputs=["fs:f"], pipeline="p"),
        _job("chain.py", inputs=["fs:f"], outputs=["model:m"], pipeline="p"),
        _job("chain.py", inputs=["model:m"], outputs=["endpoint:e"], pipeline="p"),
    ]
    nl = _serialized(jobs, "p")

    job_ids = [n["id"] for n in nl["nodes"] if n["kind"] == "job"]
    assert len(job_ids) == len(set(job_ids)) == 3  # distinct despite the shared script

    edges = _collapse(nl)
    assert edges == {("ds:d", "fs:f"), ("fs:f", "model:m"), ("model:m", "endpoint:e")}
    assert ("ds:d", "model:m") not in edges  # the collision bug produced this spurious edge


def test_merged_pipelines_do_not_cross_wire():
    """Two pipelines whose jobs share a script/label must stay separate when merged."""
    p1 = _serialized(
        [
            _job("featurize.py", inputs=["ds:a"], outputs=["fs:a"], pipeline="p1"),
            _job("train.py", inputs=["fs:a"], outputs=["model:a"], pipeline="p1"),
        ],
        "p1",
    )
    p2 = _serialized(
        [
            _job("featurize.py", inputs=["ds:b"], outputs=["fs:b"], pipeline="p2"),
            _job("train.py", inputs=["fs:b"], outputs=["model:b"], pipeline="p2"),
        ],
        "p2",
    )
    edges = _collapse(_merge([p1, p2]))
    # Exactly the two independent chains -- no cross-pipeline edges (e.g. ds:a -> fs:b).
    assert edges == {
        ("ds:a", "fs:a"),
        ("fs:a", "model:a"),
        ("ds:b", "fs:b"),
        ("fs:b", "model:b"),
    }


def test_single_job_with_many_outputs_fans_from_its_input():
    """One script, real output list: input wires to each output (a fan, not a chain)."""
    jobs = [
        _job("build_all.py", inputs=["fs:f"], outputs=["model:a", "model:b", "endpoint:a"], pipeline="p"),
    ]
    edges = _collapse(_serialized(jobs, "p"))
    assert edges == {("fs:f", "model:a"), ("fs:f", "model:b"), ("fs:f", "endpoint:a")}


# ---- linearize(): the artifact-only lineage view the UI renders ----


def _lin(jobs, name):
    nl = linearize(PipelineManager.from_jobs(jobs).get_pipeline(name))
    return nl, {(link["source"], link["target"]) for link in nl["links"]}


def test_linearize_threads_a_mega_script_into_a_chain():
    """One job ds -> {fs, model, endpoint} becomes the chain, not a ds-fanned mesh."""
    nl, edges = _lin([_job("mega.py", inputs=["ds:d"], outputs=["fs:f", "model:m", "endpoint:e"], pipeline="p")], "p")
    assert {n["id"] for n in nl["nodes"]} == {"ds:d", "fs:f", "model:m", "endpoint:e"}  # artifact-only
    assert all("kind" not in n for n in nl["nodes"])  # no job nodes in the lineage view
    assert all("type" in n for n in nl["nodes"])  # artifact nodes keep their type marker
    assert edges == {("ds:d", "fs:f"), ("fs:f", "model:m"), ("model:m", "endpoint:e")}
    assert ("ds:d", "model:m") not in edges and ("ds:d", "endpoint:e") not in edges  # no band-skips


def test_linearize_keeps_per_stage_pipelines_and_promotion_fan_in():
    """Separate per-stage scripts stay a chain; a promote job fans its models into one endpoint."""
    jobs = [
        _job("fs.py", inputs=["ds:d"], outputs=["fs:f"], pipeline="p"),
        _job("m1.py", inputs=["fs:f"], outputs=["model:a"], pipeline="p"),
        _job("m2.py", inputs=["fs:f"], outputs=["model:b"], pipeline="p"),
        _job("promote.py", inputs=["model:a", "model:b"], outputs=["endpoint:e"], pipeline="p"),
    ]
    _, edges = _lin(jobs, "p")
    assert edges == {
        ("ds:d", "fs:f"),
        ("fs:f", "model:a"),
        ("fs:f", "model:b"),
        ("model:a", "endpoint:e"),
        ("model:b", "endpoint:e"),
    }


def test_linearize_matches_models_to_endpoints_by_name():
    """One assay job with N models + N endpoints threads model:x -> endpoint:x by name."""
    jobs = [
        _job(
            "caco2_efflux.py",
            inputs=["fs:f"],
            outputs=["model:a", "model:b", "endpoint:a", "endpoint:b"],
            pipeline="p",
        )
    ]
    _, edges = _lin(jobs, "p")
    assert edges == {
        ("fs:f", "model:a"),
        ("fs:f", "model:b"),
        ("model:a", "endpoint:a"),  # paired by name, not fs-fanned
        ("model:b", "endpoint:b"),
    }
    # no band-skip: fs never wires straight to an endpoint
    assert not any(a.startswith("fs:") and b.startswith("endpoint:") for a, b in edges)


def test_linearize_threads_public_into_the_datasource_not_parallel():
    """abalone-shaped: a public INPUT feeds the ds it creates, then chains up the ladder.

    ``DataSource(PublicData().get(...))`` -- public flows *into* the ds, so the lineage is
    ``public -> ds -> fs -> model -> endpoint``, never public and ds both fanning to fs.
    """
    jobs = [
        _job(
            "abalone.py",
            inputs=["public:testing/abalone"],
            outputs=[
                "ds:abalone_data",
                "fs:abalone_features",
                "model:abalone-regression",
                "endpoint:abalone-regression",
            ],
            pipeline="p",
        )
    ]
    _, edges = _lin(jobs, "p")
    assert edges == {
        ("public:testing/abalone", "ds:abalone_data"),
        ("ds:abalone_data", "fs:abalone_features"),
        ("fs:abalone_features", "model:abalone-regression"),
        ("model:abalone-regression", "endpoint:abalone-regression"),
    }
    # the bug: public paralleling ds straight into the featureset
    assert ("public:testing/abalone", "fs:abalone_features") not in edges


def test_linearize_keeps_parallel_ds_and_public_inputs_into_one_featureset():
    """The legitimate rare case: a ds INPUT and a public INPUT both feed a featureset.

    Both are inputs (neither is an output that chains), so they stay parallel into fs --
    this is why threading is role-aware rather than giving public its own band.
    """
    jobs = [_job("featurize.py", inputs=["ds:existing", "public:testing/extra"], outputs=["fs:combined"], pipeline="p")]
    _, edges = _lin(jobs, "p")
    assert edges == {("ds:existing", "fs:combined"), ("public:testing/extra", "fs:combined")}


def test_get_pipeline_unknown_name_raises():
    """An unknown (or empty) pipeline name raises KeyError rather than returning an empty graph."""
    with pytest.raises(KeyError):
        PipelineManager.from_jobs([]).get_pipeline("p")


def test_linearize_falls_back_to_fan_when_model_endpoint_counts_differ():
    """Unequal model/endpoint counts (2 models, 3 endpoints) must also use the plain fan."""
    jobs = [
        _job(
            "all_models.py",
            inputs=["fs:f"],
            outputs=["model:a", "model:b", "endpoint:x", "endpoint:y", "endpoint:z"],
            pipeline="p",
        )
    ]
    _, edges = _lin(jobs, "p")
    assert edges == {
        ("fs:f", "model:a"),
        ("fs:f", "model:b"),
        ("fs:f", "endpoint:x"),
        ("fs:f", "endpoint:y"),
        ("fs:f", "endpoint:z"),
    }
    # crucially, no invented model -> endpoint edges
    assert not any(a.startswith("model:") and b.startswith("endpoint:") for a, b in edges)


def test_linearize_falls_back_to_fan_when_names_do_not_pair():
    """N models + M endpoints whose names don't line up 1:1 -> plain fan, never a guess."""
    jobs = [
        _job(
            "all_models.py",
            inputs=["fs:f"],
            outputs=["model:a", "model:b", "endpoint:x", "endpoint:y"],
            pipeline="p",
        )
    ]
    _, edges = _lin(jobs, "p")
    assert edges == {("fs:f", "model:a"), ("fs:f", "model:b"), ("fs:f", "endpoint:x"), ("fs:f", "endpoint:y")}
    # crucially, no invented model -> endpoint edges
    assert not any(a.startswith("model:") and b.startswith("endpoint:") for a, b in edges)
