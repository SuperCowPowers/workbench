"""Archive: a pile of a client's files (an S3 prefix or a directory) read into one frame of rows"""

import fnmatch
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd

log = logging.getLogger("workbench")

# Parallel downloads for `Archive.fetch`
FETCH_WORKERS = 12


@dataclass
class ArchiveRead:
    """The result of `Archive.read`: the rows the files held, and the files that could not be read.

    Attributes:
        rows (pd.DataFrame): Every reader's rows, with `source_file` (the file's name) and
            `source_row` (the row's position in what the reader returned for it).
        unreadable (pd.DataFrame): One row per file the reader raised on: `source_file`, `reason`.
        files (int): Files the reader was given.
    """

    rows: pd.DataFrame
    unreadable: pd.DataFrame
    files: int


def _read_one(reader: Callable, path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Run the reader on one file; its rows, or why it could not be read."""
    try:
        return reader(path), None
    except Exception as exc:  # noqa: BLE001 - one bad file is a finding, not the end of the read
        return None, f"{type(exc).__name__}: {exc}"


class Archive:
    """Archive: a pile of a client's files (an S3 prefix or a directory) read into one frame of rows.

    The Archive owns where the files are, a local copy of them, and running a reader over each.
    The reader is the client's own code: given one file, it returns that file's rows, and it
    raises on a file it cannot read. Every row keeps the file it came from.

    Common Usage:
        ```python
        from workbench.utils.archive import Archive

        def read_upload(path):
            return pd.read_excel(path, sheet_name="Upload")

        archive = Archive("s3://bucket/pk_studies", read_upload, pattern="*.xlsx")
        archive.fetch("~/work/datasets/pk_studies")
        result = archive.read("~/work/datasets/pk_studies")
        print(result.rows.head(), result.unreadable)
        ```
    """

    def __init__(
        self,
        location: str,
        reader: Callable[[Path], pd.DataFrame],
        name: str = "Archive",
        pattern: str = "*",
        boto3_session=None,
    ):
        """Archive Init

        Args:
            location (str): Where the files are: an S3 prefix ("s3://bucket/prefix") or a local
                or network directory. Reports print it under the Archive's name.
            reader (Callable[[Path], pd.DataFrame]): Reads one file into its rows; raises on a
                file it cannot read. Must be a module-level function to read with `workers` > 1.
            name (str): What reports call it, and what a `Note(about=...)` names.
            pattern (str): Which file names belong to it, as a glob. Defaults to every file.
            boto3_session: Session for an S3 location. Defaults to the Workbench role's; pass
                another when the Archive lives in an account that role cannot read.
        """
        self.location, self.reader, self.name, self.pattern = location, reader, name, pattern
        self._session = boto3_session

    @property
    def on_s3(self) -> bool:
        """Whether the files live on S3."""
        return self.location.startswith("s3://")

    def _bucket_prefix(self) -> Tuple[str, str]:
        """The bucket and the key prefix of an S3 location; the prefix is empty for a whole bucket."""
        bucket, _, prefix = self.location.removeprefix("s3://").partition("/")
        prefix = prefix.strip("/")
        return bucket, f"{prefix}/" if prefix else ""

    def _s3(self):
        if self._session is None:
            from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

            self._session = AWSAccountClamp().boto3_session
        return self._session.client("s3")

    def files(self) -> pd.DataFrame:
        """The files the Archive holds.

        Returns:
            pd.DataFrame: One row per file matching `pattern`: `name`, `key` (its S3 key or its
                path), `size` in bytes, and `modified`.
        """
        rows = []
        if self.on_s3:
            bucket, prefix = self._bucket_prefix()
            for page in self._s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    name = obj["Key"].rsplit("/", 1)[-1]
                    if name and fnmatch.fnmatch(name, self.pattern):
                        rows.append([name, obj["Key"], obj["Size"], obj["LastModified"]])
        else:
            for path in sorted(Path(self.location).expanduser().rglob(self.pattern)):
                if path.is_file():
                    stat = path.stat()
                    rows.append([path.name, str(path), stat.st_size, pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")])
        return pd.DataFrame(rows, columns=["name", "key", "size", "modified"])

    def fetch(self, local_dir: Union[str, Path], names: Optional[list] = None) -> pd.DataFrame:
        """Copy the Archive's files from S3 into a local directory, skipping those already there.

        Args:
            local_dir (Union[str, Path]): Where the copies go, flat, by file name.
            names (Optional[list]): Only these file names. Defaults to every file.

        Returns:
            pd.DataFrame: One row per file: `name` and `status` ("fetched", "present", or why it failed).

        Raises:
            ValueError: If the Archive is a directory, which is read in place.
        """
        if not self.on_s3:
            raise ValueError(f"{self.location} is a directory; read it in place")
        local = Path(local_dir).expanduser()
        local.mkdir(parents=True, exist_ok=True)
        listed = self.files()
        if names is not None:
            listed = listed[listed["name"].isin(names)]
        bucket, client = self._bucket_prefix()[0], self._s3()

        def get(name: str, key: str) -> Tuple[str, str]:
            if (local / name).exists():
                return name, "present"
            try:
                client.download_file(bucket, key, str(local / name))
                return name, "fetched"
            except Exception as exc:  # noqa: BLE001 - report each file, keep the sweep going
                return name, f"failed: {exc}"

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            done = list(pool.map(get, listed["name"], listed["key"]))
        status = pd.DataFrame(done, columns=["name", "status"])
        log.info(f"Archive {self.name}: {status['status'].value_counts().to_dict()}")
        return status

    def read(self, local_dir: Union[str, Path, None] = None, workers: int = 1) -> ArchiveRead:
        """Run the reader over every file.

        Args:
            local_dir (Union[str, Path, None]): The local copy of an S3 Archive (see `fetch`).
                A directory Archive is read in place.
            workers (int): Processes to read with. Defaults to 1, in this process.

        Returns:
            ArchiveRead: The rows, the files that could not be read, and how many were given.

        Raises:
            ValueError: If an S3 Archive is read without a local copy.
        """
        if self.on_s3 and local_dir is None:
            raise ValueError("An S3 Archive is read from its local copy: fetch it, then pass local_dir")
        root = Path(local_dir if local_dir is not None else self.location).expanduser()
        paths = sorted(path for path in root.rglob(self.pattern) if path.is_file())
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_read_one, [self.reader] * len(paths), paths))
        else:
            results = [_read_one(self.reader, path) for path in paths]

        frames, unreadable = [], []
        for path, (frame, reason) in zip(paths, results):
            if reason is not None:
                unreadable.append([path.name, reason])
            elif frame is not None and len(frame):
                frames.append(frame.reset_index(drop=True).assign(source_file=path.name, source_row=range(len(frame))))
        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["source_file", "source_row"])
        return ArchiveRead(rows, pd.DataFrame(unreadable, columns=["source_file", "reason"]), len(paths))
