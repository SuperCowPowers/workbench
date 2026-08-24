"""artifact_times: resolve a typed artifact ref to its last-modified time.

The freshness primitive behind pipeline scheduling ("is this consumer older than
its inputs?") and any other caller that needs to know when an artifact last
changed. Raw boto3 so it stays inside the layer's dependency budget.

    ds:<name>       -> Glue table UpdateTime
    fs:<name>       -> FeatureGroup CreationTime
    model:<name>    -> latest model package CreationTime
    public:<name>   -> PublicData S3 object LastModified (unsigned, no creds)
    endpoint:<name> -> its EndpointConfig CreationTime

Absent vs. unknown is the distinction that matters. A genuinely-absent artifact
returns None, which callers read as "must build it". A lookup we *couldn't
complete* (bad creds, no region, AccessDenied, throttling) raises instead:
returning None there would silently rebuild everything.
"""

import logging

# boto3 error codes that mean "this artifact does not exist (yet)" -- a legitimate
# "must run". Distinct from auth/region/throttle failures, which mean "could not
# determine freshness" and must NOT be silently treated as must-run.
_ARTIFACT_NOT_FOUND_CODES = {
    "EntityNotFoundException",  # Glue get_table
    "ResourceNotFound",  # SageMaker describe_feature_group
    "ResourceNotFoundException",
    "ValidationException",  # SageMaker list_model_packages against a missing group
}

# Public datasets live in this anonymous, read-only S3 bucket. Mirrors
# workbench.api.PublicData.BUCKET, duplicated here (rather than imported) to keep the
# layer dependency-light -- importing PublicData would pull pandas. Resolved via an
# unsigned client, so a `public:` ref's mtime needs no credentials.
PUBLIC_DATA_BUCKET = "workbench-public-data"
PUBLIC_DATA_EXTENSIONS = (".parquet", ".csv")


class ArtifactTimes:
    """Last-modified resolution for typed artifact refs, with a lazy client cache.

    Args:
        session: Optional boto3 Session. The launcher passes workbench's
            region-bound, assumed-role session; a Lambda passes None and gets the
            default client (region from AWS_REGION).
    """

    def __init__(self, session=None):
        self._session = session
        self._clients: dict = {}
        self.log = logging.getLogger("workbench")

    def mtime(self, ref: str):
        """Last-modified time of ``ref``, or None if it doesn't exist."""
        from botocore.exceptions import ClientError

        kind, _, name = ref.partition(":")
        if not name:
            self.log.error(f"Unrecognized artifact ref (no type prefix): {ref!r}")
            return None
        resolver = getattr(self, f"_{kind}_time", None)
        if resolver is None:
            self.log.error(f"Unknown artifact type in ref: {ref!r}")
            return None
        try:
            return resolver(name)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in _ARTIFACT_NOT_FOUND_CODES:
                # Artifact doesn't exist -> "must run". Expected on a first build; a *wall*
                # of these is caught by the plan-level guard in PipelineManager.plan().
                self.log.warning(f"mtime({ref!r}) -> absent ({code})")
                return None
            # Couldn't determine freshness (AccessDenied, throttling, ...). Fail loudly
            # rather than guess "must run" and rebuild everything.
            self.log.error(f"mtime({ref!r}) -> lookup failed ({code}); cannot assess freshness")
            raise

    # -- clients --------------------------------------------------------------

    def client(self, name: str):
        """A cached boto3 client bound to this instance's session."""
        import boto3  # lazy: from the Lambda runtime / workbench's boto3

        if name not in self._clients:
            self._clients[name] = (self._session or boto3).client(name)
        return self._clients[name]

    def public_s3(self):
        """Anonymous (unsigned) S3 client for the public data bucket -- no creds, us-west-2."""
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        if "__public__" not in self._clients:
            self._clients["__public__"] = boto3.client(
                "s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED)
            )
        return self._clients["__public__"]

    # -- per-type resolvers (dispatched by mtime() on the ref's type prefix) ---

    def _ds_time(self, name: str):
        return self.client("glue").get_table(DatabaseName="workbench", Name=name)["Table"]["UpdateTime"]

    def _fs_time(self, name: str):
        return self.client("sagemaker").describe_feature_group(FeatureGroupName=name)["CreationTime"]

    def _model_time(self, name: str):
        packages = self.client("sagemaker").list_model_packages(
            ModelPackageGroupName=name, SortBy="CreationTime", SortOrder="Descending", MaxResults=1
        )["ModelPackageSummaryList"]
        return packages[0]["CreationTime"] if packages else None

    def _endpoint_time(self, name: str):
        """CreationTime of the endpoint's EndpointConfig -- when its content last changed.

        The endpoint's own LastModifiedTime tracks state transitions (capacity,
        serverless scaling), not content, so it drifts forward on an untouched
        endpoint. A re-deploy always mints a new EndpointConfig, so the config's
        CreationTime is the timestamp freshness should be judged against.

        Future self: for a MetaEndpoint this is the *parent's* config, so
        redeploying a child (smiles-to-3d-v2 under smiles-to-2d-3d-v2) doesn't
        move it. The child list is in the endpoint's chunked workbench tags, and
        decoding those lives in workbench.utils -- outside this layer's budget.
        Meanwhile a pipeline that cares declares the children as inputs next to
        the parent: they resolve as roots and the forward flood does the rest.
        """
        sm = self.client("sagemaker")
        config = sm.describe_endpoint(EndpointName=name)["EndpointConfigName"]
        return sm.describe_endpoint_config(EndpointConfigName=config)["CreationTime"]

    def _public_time(self, name: str):
        """LastModified of a PublicData object, trying each known extension.

        Returns None if the dataset isn't found under any extension (-> "must run").
        A non-404 error (throttle, etc.) propagates to ``mtime``'s ClientError
        handler -- same "don't guess on failure" rule as the other resolvers.
        """
        from botocore.exceptions import ClientError

        s3 = self.public_s3()
        for ext in PUBLIC_DATA_EXTENSIONS:
            try:
                return s3.head_object(Bucket=PUBLIC_DATA_BUCKET, Key=name + ext)["LastModified"]
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                    continue  # try the next extension
                raise
        self.log.warning(f"mtime('public:{name}') -> absent (no .parquet/.csv in {PUBLIC_DATA_BUCKET})")
        return None
