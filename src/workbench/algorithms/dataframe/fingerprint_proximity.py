import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union
import logging

# Cross-module imports: try the workbench package path first (Jupyter / library use);
# fall back to in-package sibling imports when this module is symlinked into a
# SageMaker script bundle's `model_script_utils/` package (no workbench installed).
try:
    from workbench.algorithms.dataframe.proximity import Proximity
    from workbench.utils.chem_utils.fingerprints import compute_morgan_fingerprints
except ImportError:
    from .proximity import Proximity
    from .fingerprints import compute_morgan_fingerprints

# Note: Projection2D is imported lazily inside project_2d() — it's only needed
# for visualization, not for inference, and we don't want to pay its import cost
# (or fail to import the module) in a bundle context.

# Set up logging
log = logging.getLogger("workbench")


class _SparseRuzickaNN:
    """Sklearn-compatible NearestNeighbors-style wrapper that computes Ruzicka
    (weighted Tanimoto) distances on-the-fly against a stored sparse reference set.

    No precomputed N×N matrix — supports novel queries and scales to large reference
    sets (50k+ compounds). Memory is O(N × nnz) for storage; query memory is bounded
    by `chunk_size × N` regardless of query batch size.

    Identity used for Ruzicka distance:
        ruzicka_dist = 2*L1 / (S_q + S_r + L1)
    where L1 is Manhattan distance and S_q / S_r are row sums of query / reference.
    """

    DEFAULT_CHUNK_SIZE = 1024

    def __init__(self, X_sparse: csr_matrix, row_sums: np.ndarray, chunk_size: Optional[int] = None):
        """
        Args:
            X_sparse: Reference fingerprint matrix as CSR sparse (n_ref, n_features)
            row_sums: Row sums of X_sparse, shape (n_ref,)
            chunk_size: Query rows processed per batch. Bounds transient memory to
                chunk_size × n_ref × 4 bytes (float32). Default: 1024 — ~210 MB
                transient at n_ref=50k.
        """
        self._X = X_sparse
        self._row_sums = row_sums.astype(np.float32)
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE

    @staticmethod
    def _as_csr_float32(X) -> csr_matrix:
        """Coerce input to float32 CSR for sparse Manhattan/sum ops."""
        if not isinstance(X, csr_matrix):
            return csr_matrix(np.asarray(X, dtype=np.float32))
        return X.astype(np.float32) if X.dtype != np.float32 else X

    def _ruzicka_block(self, X_query_chunk: csr_matrix) -> np.ndarray:
        """Compute Ruzicka distance for one chunk of queries against the full reference.

        Memory footprint: O(chunk_rows × n_ref) — three transient float32 arrays
        (l1, denom, output) of that shape.

        Args:
            X_query_chunk: Float32 CSR matrix (chunk_rows, n_features)

        Returns:
            np.ndarray of shape (chunk_rows, n_ref) with Ruzicka distances in [0, 1]
        """
        l1 = pairwise_distances(X_query_chunk, self._X, metric="manhattan", n_jobs=-1).astype(np.float32)
        q_sums = np.asarray(X_query_chunk.sum(axis=1)).ravel().astype(np.float32)

        S = q_sums[:, np.newaxis]
        T = self._row_sums[np.newaxis, :]
        denom = S + T + l1
        return np.divide(2.0 * l1, denom, where=denom > 0, out=np.zeros_like(l1))

    def pairwise_ruzicka_matrix(self, X_query: Optional[csr_matrix] = None) -> np.ndarray:
        """Materialize the full (n_query, n_ref) Ruzicka distance matrix, chunk-filled.

        Use when a full matrix is genuinely required (e.g. UMAP precomputed-metric
        projection). Output is O(n_query × n_ref) memory; transient overhead is
        bounded by chunk_size × n_ref.

        Args:
            X_query: Sparse or dense query matrix. If None, defaults to the stored
                reference (returns the symmetric self-distance matrix).

        Returns:
            np.ndarray of shape (n_query, n_ref) with Ruzicka distances in [0, 1]
        """
        X_query = self._X if X_query is None else self._as_csr_float32(X_query)
        n_query = X_query.shape[0]
        n_ref = self._X.shape[0]
        out = np.empty((n_query, n_ref), dtype=np.float32)
        for start in range(0, n_query, self.chunk_size):
            end = min(start + self.chunk_size, n_query)
            out[start:end] = self._ruzicka_block(X_query[start:end])
        return out

    @staticmethod
    def _topk_per_row(dist: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k smallest entries per row of `dist`, sorted ascending.

        Returns (top_dist, top_idx) — both shape (M, k).
        """
        n_ref = dist.shape[1]
        if k >= n_ref:
            idx = np.argsort(dist, axis=1)
            return np.take_along_axis(dist, idx, axis=1), idx

        # argpartition for top-k, then sort within the k
        part_idx = np.argpartition(dist, k, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist, part_idx, axis=1)
        order = np.argsort(part_dist, axis=1)
        return np.take_along_axis(part_dist, order, axis=1), np.take_along_axis(part_idx, order, axis=1)

    def kneighbors(self, X_query, n_neighbors: int):
        """Return distances and indices of the k nearest neighbors for each query row.

        Matches sklearn.neighbors.NearestNeighbors.kneighbors signature. Chunked
        internally — transient memory bounded by chunk_size × n_ref regardless
        of how many queries are passed.

        Args:
            X_query: Sparse or dense query matrix (n_query, n_features)
            n_neighbors: Number of neighbors to return

        Returns:
            (distances, indices) — both shape (n_query, n_neighbors), sorted ascending
        """
        X_query = self._as_csr_float32(X_query)
        n_query = X_query.shape[0]
        n_ref = self._X.shape[0]
        k = min(n_neighbors, n_ref)

        out_dist = np.empty((n_query, k), dtype=np.float32)
        out_idx = np.empty((n_query, k), dtype=np.int64)
        for start in range(0, n_query, self.chunk_size):
            end = min(start + self.chunk_size, n_query)
            chunk_dist = self._ruzicka_block(X_query[start:end])
            out_dist[start:end], out_idx[start:end] = self._topk_per_row(chunk_dist, k)
            del chunk_dist
        return out_dist, out_idx

    def radius_neighbors(self, X_query, radius: float):
        """Return all neighbors within `radius` for each query row.

        Matches sklearn.neighbors.NearestNeighbors.radius_neighbors signature. Chunked
        internally — transient memory bounded by chunk_size × n_ref.

        Args:
            X_query: Sparse or dense query matrix (n_query, n_features)
            radius: Maximum distance threshold

        Returns:
            (distances, indices) — both lists-of-ndarrays, one entry per query row,
            sorted ascending by distance.
        """
        X_query = self._as_csr_float32(X_query)
        n_query = X_query.shape[0]

        distances_out = []
        indices_out = []
        for start in range(0, n_query, self.chunk_size):
            end = min(start + self.chunk_size, n_query)
            chunk_dist = self._ruzicka_block(X_query[start:end])
            for i in range(chunk_dist.shape[0]):
                mask = chunk_dist[i] <= radius
                row_idx = np.where(mask)[0]
                row_dist = chunk_dist[i, row_idx]
                order = np.argsort(row_dist)
                distances_out.append(row_dist[order])
                indices_out.append(row_idx[order])
            del chunk_dist
        return distances_out, indices_out


class FingerprintProximity(Proximity):
    """Proximity computations using Tanimoto similarity on molecular fingerprints.

    Implements the Proximity ABC contract:
        - `neighbors(id_or_ids)`     id-based lookups
        - `neighbors_from_query_df`  novel-input lookups (query_df needs a 'smiles'
                                     or 'fingerprint' column)

    Supports both binary and count fingerprints (auto-detected):
        - Binary: uses Jaccard distance (equivalent to 1 - Tanimoto for binary vectors)
        - Count: uses Ruzicka distance (weighted Tanimoto for count vectors), computed
          on-the-fly via sparse operations — supports novel queries and scales to large N.

    Result DataFrames include a `similarity = 1 - distance` column as a
    FingerprintProximity-specific extra (in addition to the canonical `distance`).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        fingerprint_column: Optional[str] = None,
        target: Optional[str] = None,
        include_all_columns: bool = False,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> None:
        """
        Initialize FingerprintProximity for Tanimoto similarity on molecular fingerprints.

        Args:
            df: DataFrame containing fingerprints or SMILES.
            id_column: Name of the column used as an identifier.
            fingerprint_column: Name of the column containing fingerprints (bit strings).
                If None, looks for existing "fingerprint" column or computes from SMILES.
            target: Name of the target column. Defaults to None.
            include_all_columns: Include all DataFrame columns in neighbor results. Defaults to False.
            radius: Radius for Morgan fingerprint computation (default: 2).
            n_bits: Number of bits for fingerprint (default: 2048).
        """
        self._fp_radius = radius
        self._fp_n_bits = n_bits
        self.fingerprint_column = self._resolve_fingerprint_column_name(df, fingerprint_column)

        super().__init__(
            df,
            id_column=id_column,
            features=[self.fingerprint_column],
            target=target,
            include_all_columns=include_all_columns,
        )

    @staticmethod
    def _resolve_fingerprint_column_name(df: pd.DataFrame, fingerprint_column: Optional[str]) -> str:
        """Determine the fingerprint column name, validating it exists or can be computed."""
        if fingerprint_column is not None:
            if fingerprint_column not in df.columns:
                raise ValueError(f"Fingerprint column '{fingerprint_column}' not found in DataFrame")
            return fingerprint_column

        if "fingerprint" in df.columns:
            log.info("Using existing 'fingerprint' column")
            return "fingerprint"

        # Will need to compute from SMILES - validate SMILES column exists
        smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
        if smiles_column is None:
            raise ValueError(
                "No fingerprint column provided and no SMILES column found. "
                "Either provide a fingerprint_column or include a 'smiles' column in the DataFrame."
            )

        return "fingerprint"

    def _prepare_data(self) -> None:
        """Compute fingerprints from SMILES if needed."""
        if self.fingerprint_column not in self.df.columns:
            log.info(f"Computing Morgan fingerprints (radius={self._fp_radius}, n_bits={self._fp_n_bits})...")
            self.df = compute_morgan_fingerprints(self.df, radius=self._fp_radius, n_bits=self._fp_n_bits)

    def _build_model(self) -> None:
        """Build the fingerprint proximity model for Tanimoto similarity.

        For binary fingerprints: uses Jaccard distance (1 - Tanimoto) via sklearn ball_tree.
        For count fingerprints: stores a sparse CSR reference matrix and a custom NN wrapper
            that computes Ruzicka (weighted Tanimoto) distance on-the-fly. No precomputed
            N×N matrix — supports novel queries and scales to large reference sets.
        """
        X, self._is_count_fp = self._fingerprints_to_matrix(self.df)

        if self._is_count_fp:
            log.info("Building NearestNeighbors model (sparse on-the-fly Ruzicka for count fingerprints)...")
            self._X_sparse = csr_matrix(X.astype(np.float32))
            self._row_sums = np.asarray(self._X_sparse.sum(axis=1)).ravel().astype(np.float32)
            self.nn = _SparseRuzickaNN(self._X_sparse, self._row_sums)
            self.X = None  # not used for count FPs
        else:
            log.info("Building NearestNeighbors model (Jaccard/Tanimoto for binary fingerprints)...")
            self.X = X
            self.nn = NearestNeighbors(metric="jaccard", algorithm="ball_tree").fit(self.X)

        # Cache: id → row index in the reference set. Used by _transform_features to
        # answer id-based queries without re-parsing fingerprint strings (and works
        # even after the artifact is slimmed by UQModelV1._slim_proximity).
        self._id_to_row = {row_id: i for i, row_id in enumerate(self.df[self.id_column].values)}

    def _transform_features(self, df: pd.DataFrame) -> Union[np.ndarray, csr_matrix]:
        """Transform features for querying the NN model.

        Three paths, in order of cost:
            1. Identity fast path: when df is the reference DataFrame itself,
               return the cached matrix directly.
            2. ID-based row lookup: when all IDs in `df[id_column]` are known in
               the reference set, slice rows from `_X_sparse` (or `self.X`) directly.
               No fingerprint parsing, no Morgan recomputation. This path works
               even after the artifact is slimmed (fingerprint column dropped).
            3. Novel-query path: parse fingerprints from `df`, computing Morgan
               from SMILES if needed.

        For count fingerprints the matrix is sparse CSR; for binary, dense.
        """
        # Path 1: reference DataFrame itself
        if df is self.df:
            return self._X_sparse if self._is_count_fp else self.X

        # Path 2: id-based row lookup. Cheap and works post-slim.
        if self.id_column in df.columns:
            ids = df[self.id_column].values
            id_to_row = getattr(self, "_id_to_row", None)
            if id_to_row is not None:
                try:
                    indices = np.fromiter((id_to_row[i] for i in ids), dtype=np.int64, count=len(ids))
                except KeyError:
                    indices = None
                if indices is not None:
                    if self._is_count_fp:
                        return self._X_sparse[indices]
                    return self.X[indices]

        # Path 3: novel-query path. Need fingerprints or SMILES.
        if self.fingerprint_column not in df.columns:
            if "smiles" not in df.columns and "SMILES" not in df.columns:
                raise ValueError(
                    f"Query DataFrame must contain either '{self.fingerprint_column}' " "or a 'smiles' column"
                )
            df = compute_morgan_fingerprints(df, radius=self._fp_radius, n_bits=self._fp_n_bits)

        matrix, _ = self._fingerprints_to_matrix(df)
        if self._is_count_fp:
            return csr_matrix(matrix.astype(np.float32))
        return matrix

    def _fingerprints_to_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, bool]:
        """Convert fingerprint strings to a numpy matrix.

        Supports two formats (auto-detected):
            - Bitstrings: "10110010..." → binary matrix (bool), is_count=False
            - Count vectors: "0,3,0,1,5,..." → count matrix (uint8), is_count=True
        """
        sample = str(df[self.fingerprint_column].iloc[0])
        if "," in sample:
            fingerprint_values = df[self.fingerprint_column].apply(
                lambda fp: np.array([int(x) for x in fp.split(",")], dtype=np.uint8)
            )
            return np.vstack(fingerprint_values), True
        else:
            fingerprint_bits = df[self.fingerprint_column].apply(
                lambda fp: np.array([int(bit) for bit in fp], dtype=np.bool_)
            )
            return np.vstack(fingerprint_bits), False

    def neighbors(
        self,
        id_or_ids,
        n_neighbors: Optional[int] = 5,
        min_similarity: Optional[float] = None,
        include_self: bool = True,
    ) -> pd.DataFrame:
        """Return neighbors for ID(s) already in the reference dataset.

        Args:
            id_or_ids: Single ID or list of IDs to look up
            n_neighbors: Number of neighbors to return (default: 5, ignored if min_similarity is set)
            min_similarity: If provided, find all neighbors with Tanimoto similarity >= this value (0-1)
            include_self: Whether to include self in results (default: True)

        Returns:
            DataFrame with columns: id_column, neighbor_id, similarity, [target], [in_model],
            and any other passthrough columns.
        """
        radius = 1 - min_similarity if min_similarity is not None else None
        result = super().neighbors(
            id_or_ids=id_or_ids,
            n_neighbors=n_neighbors,
            radius=radius,
            include_self=include_self,
        )
        return self._add_similarity_column(result)

    def neighbors_from_query_df(
        self,
        query_df: pd.DataFrame,
        n_neighbors: int = 5,
        min_similarity: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return neighbors for novel queries not in the reference dataset.

        Args:
            query_df: DataFrame with either a 'smiles' or 'fingerprint' column. If a
                'query_id' column is present it's used to label results; otherwise
                positional indices are used.
            n_neighbors: Number of neighbors to return (default: 5, ignored if min_similarity is set)
            min_similarity: If provided, find all neighbors with Tanimoto similarity >= this value (0-1)

        Returns:
            DataFrame with columns: query_id, neighbor_id, similarity, [target], [in_model].
            Queries whose SMILES couldn't be parsed by RDKit are dropped with a
            warning — their rows simply don't appear in the result. Upstream
            consumers (residual_features._aggregate) reindex against the full
            input id list so missing queries surface as NaN rows there.
        """
        # Pre-validate SMILES. `compute_morgan_fingerprints` (called downstream
        # by _transform_features for novel queries) silently drops rows with
        # invalid SMILES, which then causes an array-length mismatch in
        # _neighbors_impl's result assembly. Drop bad rows here so the feature
        # matrix and query_ids stay aligned.
        smiles_col = next((c for c in ("smiles", "SMILES") if c in query_df.columns), None)
        if smiles_col is not None and self.fingerprint_column not in query_df.columns:
            from rdkit import Chem

            valid = query_df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) is not None)
            if not valid.all():
                n_bad = int((~valid).sum())
                bad_sample = query_df.loc[~valid, smiles_col].head(3).tolist()
                log.warning(
                    f"FingerprintProximity.neighbors_from_query_df: dropping {n_bad} "
                    f"row(s) with SMILES that RDKit can't parse "
                    f"(sample: {bad_sample}{'...' if n_bad > 3 else ''}). "
                    "These rows will be absent from the result."
                )
                query_df = query_df[valid].reset_index(drop=True)
                if len(query_df) == 0:
                    # All queries invalid — return an empty result rather than crash in the NN backend.
                    return pd.DataFrame(columns=["query_id", "neighbor_id", "similarity"])

        radius = 1 - min_similarity if min_similarity is not None else None
        result = super().neighbors_from_query_df(
            query_df=query_df,
            n_neighbors=n_neighbors,
            radius=radius,
        )
        return self._add_similarity_column(result)

    @staticmethod
    def _add_similarity_column(result_df: pd.DataFrame) -> pd.DataFrame:
        """Append `similarity = 1 - distance` and drop the raw distance column."""
        result_df["similarity"] = 1 - result_df["distance"]
        result_df.drop(columns=["distance"], inplace=True)
        # Re-sort: similarity descending (was ascending by distance).
        # Use the leading id column (first column) and similarity.
        id_col = result_df.columns[0]
        return result_df.sort_values([id_col, "similarity"], ascending=[True, False]).reset_index(drop=True)

    def project_2d(self) -> pd.DataFrame:
        """Project the fingerprint matrix to 2D for visualization using UMAP.

        For count fingerprints: lazily materializes the full N×N Ruzicka distance matrix
        for UMAP's precomputed-metric path. Memory cost is O(N²) — transient.
        For binary fingerprints: uses Jaccard distance directly on the fingerprint matrix.

        Returns the reference DataFrame with 'x' / 'y' columns added.

        Note: Projection2D is imported lazily so the module loads in script bundles
        that don't have UMAP / workbench's projection helper installed.
        """
        from workbench.algorithms.dataframe.projection_2d import Projection2D

        if self._is_count_fp:
            dist_matrix = self.nn.pairwise_ruzicka_matrix()
            # Symmetric jitter breaks tied eigenvalues in UMAP's spectral init
            rng = np.random.default_rng(seed=0)
            n = dist_matrix.shape[0]
            noise = rng.uniform(0.0, 1e-4, size=(n, n)).astype(np.float32)
            noise = (noise + noise.T) / 2.0
            np.fill_diagonal(noise, 0.0)
            jittered = np.clip(dist_matrix + noise, 0.0, 1.0)
            self.df = Projection2D().fit_transform(self.df, feature_matrix=jittered, metric="precomputed")
        else:
            self.df = Projection2D().fit_transform(self.df, feature_matrix=self.X, metric="jaccard")
        return self.df


# Testing the FingerprintProximity class
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Binary FP basics
    data = {
        "id": ["a", "b", "c", "d", "e"],
        "fingerprint": ["101010", "111010", "101110", "011100", "000111"],
        "target": [1, 0, 1, 0, 5],
    }
    df = pd.DataFrame(data)
    prox = FingerprintProximity(df, fingerprint_column="fingerprint", id_column="id", target="target")
    print("\nNeighbors for 'a' (k=3):")
    print(prox.neighbors("a", n_neighbors=3))
    print("\nNeighbors for 'a' (min_similarity=0.5):")
    print(prox.neighbors("a", min_similarity=0.5))

    # Novel-input via query_df with explicit fingerprint
    novel = pd.DataFrame({"fingerprint": ["111111"], "query_id": ["novel_compound_1"]})
    print("\nNovel-input query (binary FP):")
    print(prox.neighbors_from_query_df(novel, n_neighbors=3))
