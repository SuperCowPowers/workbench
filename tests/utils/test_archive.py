"""Tests for Archive: listing, fetching and reading a pile of files, on disk and on S3"""

import uuid

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp  # noqa: E402
from workbench.reports import Note, Report, sections  # noqa: E402
from workbench.utils.archive import Archive  # noqa: E402
from workbench.utils.config_manager import ConfigManager  # noqa: E402


def read_csv(path) -> pd.DataFrame:
    """A reader that refuses a file with no `value` column."""
    frame = pd.read_csv(path)
    if "value" not in frame.columns:
        raise ValueError("no value column")
    return frame


@pytest.fixture
def pile(tmp_path):
    (tmp_path / "studies").mkdir()
    (tmp_path / "studies" / "a.csv").write_text("arm,value\nx,1\ny,2\n")
    (tmp_path / "studies" / "b.csv").write_text("arm,value\nz,3\n")
    (tmp_path / "studies" / "broken.csv").write_text("arm,cmax\nq,9\n")
    (tmp_path / "studies" / "notes.txt").write_text("not a study")
    return tmp_path / "studies"


def test_files_lists_what_matches_the_pattern(pile):
    files = Archive(str(pile), read_csv, pattern="*.csv").files()
    assert list(files["path"]) == ["a.csv", "b.csv", "broken.csv"]
    assert list(files.columns) == ["path", "name", "key", "size", "modified"] and (files["size"] > 0).all()


def test_read_keeps_the_source_of_every_row_and_lists_what_it_could_not_read(pile):
    result = Archive(str(pile), read_csv, pattern="*.csv").read()

    assert result.files == 3
    assert result.rows[["arm", "source_file", "source_row"]].values.tolist() == [
        ["x", "a.csv", 0],
        ["y", "a.csv", 1],
        ["z", "b.csv", 0],
    ]
    assert result.unreadable.values.tolist() == [["broken.csv", "ValueError: no value column"]]


def test_files_of_one_name_in_different_folders_stay_apart(pile):
    for folder, value in (("study_a", 1), ("study_b", 2)):
        (pile / folder).mkdir()
        (pile / folder / "upload.csv").write_text(f"arm,value\nx,{value}\n")
    archive = Archive(str(pile), read_csv, pattern="upload.csv")

    assert list(archive.files()["path"]) == ["study_a/upload.csv", "study_b/upload.csv"]
    rows = archive.read().rows
    assert rows[["source_file", "value"]].values.tolist() == [["study_a/upload.csv", 1], ["study_b/upload.csv", 2]]


def test_a_missing_directory_is_refused_rather_than_read_as_empty(pile):
    with pytest.raises(FileNotFoundError):
        Archive(str(pile / "mistyped"), read_csv).read()
    with pytest.raises(FileNotFoundError):
        Archive("s3://bucket/prefix", read_csv).read(pile / "mistyped")


def test_read_in_several_processes_gives_the_same_rows(pile):
    archive = Archive(str(pile), read_csv, pattern="*.csv")
    pd.testing.assert_frame_equal(archive.read(workers=2).rows, archive.read().rows)


def test_an_s3_archive_is_read_from_its_local_copy_and_a_directory_is_not_fetched(pile):
    with pytest.raises(ValueError, match="local copy"):
        Archive("s3://bucket/prefix", read_csv).read()
    with pytest.raises(ValueError, match="in place"):
        Archive(str(pile), read_csv).fetch(pile)


def test_a_whole_bucket_is_a_location():
    assert Archive("s3://bucket", read_csv)._bucket_prefix() == ("bucket", "")
    assert Archive("s3://bucket/a/b/", read_csv)._bucket_prefix() == ("bucket", "a/b/")


def test_the_archive_section_reports_the_location_and_the_unreadable_files(pile, tmp_path):
    archive = Archive(str(pile), read_csv, pattern="*.csv")
    note = Note("Two studies were rerun.", severity="warning", about="Archive")
    report = Report("Archive read", notes=[note]).add(sections.archive(archive.read(), archive))
    assert report.save(tmp_path / "report.pdf", publish=False) == {"overlaps": [], "glyphs": []}

    class Text:
        pages = []

        def savefig(self, fig, **_):
            self.pages.append(" | ".join(t.get_text() for t in fig.texts))

    text = Text()
    report.build(text)
    page = text.pages[1]
    assert "2 of 3 files read, giving 3 rows. 1 could not be read." in page
    assert str(pile) in page and "broken.csv" in page and "Two studies were rerun." in page
    assert report.frames["archive_unreadable"]["source_file"].tolist() == ["broken.csv"]


def test_an_s3_archive_lists_and_fetches_its_files(tmp_path):
    session = AWSAccountClamp().boto3_session
    bucket, prefix = ConfigManager().get_config("WORKBENCH_BUCKET"), f"testing/archive_{uuid.uuid4().hex[:8]}"
    s3 = session.client("s3")
    objects = (("a.csv", "arm,value\nx,1\n"), ("sub/a.csv", "arm,value\ny,2\n"), ("skip.txt", "no"))
    for name, body in objects:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/{name}", Body=body.encode())
    try:
        archive = Archive(f"s3://{bucket}/{prefix}", read_csv, pattern="*.csv")
        assert list(archive.files()["path"]) == ["a.csv", "sub/a.csv"]

        assert archive.fetch(tmp_path)["status"].tolist() == ["fetched", "fetched"]
        assert archive.fetch(tmp_path)["status"].tolist() == ["present", "present"]
        rows = archive.read(tmp_path).rows
        assert rows[["source_file", "arm"]].values.tolist() == [["a.csv", "x"], ["sub/a.csv", "y"]]
        assert archive.fetch(tmp_path / "one", paths=["sub/a.csv"])["path"].tolist() == ["sub/a.csv"]
    finally:
        for name, _ in objects:
            s3.delete_object(Bucket=bucket, Key=f"{prefix}/{name}")


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--no-cov"])
