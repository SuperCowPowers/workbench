"""Find orphan tables -- views and supplemental data whose base table is gone.

Workbench derives tables from a base table: views are named ``{base}___{view}`` and
supplemental data tables ``_{base}___{view}``. When the base table goes away (deleting a
DataSource, or SageMaker dropping a FeatureSet's offline table) any derived table left
behind is an orphan: the view is permanently INVALID and the supplemental table's S3 data
is unreachable.

Detection is a Glue catalog scan, so it costs the same whether there are ten orphans or
ten thousand.

    python find_orphan_tables.py              List the orphans
    python find_orphan_tables.py --delete     List them, then delete after confirmation
"""

import argparse
import logging
from collections import defaultdict

import awswrangler as wr
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

log = logging.getLogger("workbench")

DATABASES = ["workbench", "sagemaker_featurestore"]
BATCH_SIZE = 100  # Glue batch_delete_table limit


def find_orphans(database: str, boto3_session) -> (list[dict], list[dict]):
    """Find derived tables in a database whose base table no longer exists

    Args:
        database (str): The database name
        boto3_session: The boto3 session

    Returns:
        list[dict]: Orphaned views, each {"name", "base"}
        list[dict]: Orphaned supplemental data tables, each {"name", "base", "s3_path"}
    """
    tables = list(wr.catalog.get_tables(database=database, boto3_session=boto3_session))
    base_tables = {t["Name"] for t in tables if t["TableType"] != "VIRTUAL_VIEW"}

    orphan_views = []
    orphan_supplemental = []
    for table in tables:
        name = table["Name"]

        # Views are {base}___{view}, supplemental data tables are _{base}___{view}
        if table["TableType"] == "VIRTUAL_VIEW":
            if "___" not in name:
                continue
            base = name.split("___", 1)[0]
            if base not in base_tables:
                orphan_views.append({"name": name, "base": base})
        elif name.startswith("_") and "___" in name:
            base = name[1:].split("___", 1)[0]
            if base not in base_tables:
                orphan_supplemental.append(
                    {"name": name, "base": base, "s3_path": table.get("StorageDescriptor", {}).get("Location")}
                )
    return orphan_views, orphan_supplemental


def report(database: str, orphan_views: list[dict], orphan_supplemental: list[dict]):
    """Print a summary of the orphans grouped by their (missing) base table

    Args:
        database (str): The database name
        orphan_views (list[dict]): Orphaned views
        orphan_supplemental (list[dict]): Orphaned supplemental data tables
    """
    by_base = defaultdict(lambda: [0, 0])
    for view in orphan_views:
        by_base[view["base"]][0] += 1
    for table in orphan_supplemental:
        by_base[table["base"]][1] += 1

    print(f"\n=== {database} ===")
    print(f"orphaned views:              {len(orphan_views)}")
    print(f"orphaned supplemental data:  {len(orphan_supplemental)}")
    print(f"missing base tables:         {len(by_base)}")
    if by_base:
        print("\n  views  supp  base table (does not exist)")
        for base, (views, supplemental) in sorted(by_base.items(), key=lambda kv: -sum(kv[1]))[:20]:
            print(f"  {views:5d} {supplemental:5d}  {base}")
        if len(by_base) > 20:
            print(f"  ... and {len(by_base) - 20} more")


def delete_tables(database: str, table_names: list[str], boto3_session):
    """Delete tables from the Glue catalog in batches

    Args:
        database (str): The database name
        table_names (list[str]): The table names to delete
        boto3_session: The boto3 session
    """
    glue_client = boto3_session.client("glue")
    for i in range(0, len(table_names), BATCH_SIZE):
        batch = table_names[i : i + BATCH_SIZE]
        response = glue_client.batch_delete_table(DatabaseName=database, TablesToDelete=batch)
        for failure in response.get("Errors", []):
            log.error(f"Failed to delete {failure['TableName']}: {failure['ErrorDetail']['ErrorMessage']}")
        print(f"  deleted {min(i + BATCH_SIZE, len(table_names))}/{len(table_names)}...")


def delete_s3_data(s3_paths: list[str], boto3_session):
    """Delete the S3 data behind supplemental data tables

    Args:
        s3_paths (list[str]): The S3 paths to delete
        boto3_session: The boto3 session
    """
    for s3_path in s3_paths:
        path = s3_path if s3_path.endswith("/") else f"{s3_path}/"
        print(f"  deleting S3 objects at {path}...")
        wr.s3.delete_objects(path, boto3_session=boto3_session)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete", action="store_true", help="Delete the orphans (prompts for confirmation)")
    parser.add_argument("--databases", nargs="+", default=DATABASES, help=f"Databases to scan (default: {DATABASES})")
    parser.add_argument("--keep-s3", action="store_true", help="Delete catalog entries but keep supplemental S3 data")
    args = parser.parse_args()

    boto3_session = AWSAccountClamp().boto3_session
    for database in args.databases:
        orphan_views, orphan_supplemental = find_orphans(database, boto3_session)
        report(database, orphan_views, orphan_supplemental)

        if not (orphan_views or orphan_supplemental):
            continue
        if not args.delete:
            print("\n  (dry run: pass --delete to remove these)")
            continue

        total = len(orphan_views) + len(orphan_supplemental)
        s3_note = "" if args.keep_s3 else f" (and the S3 data behind {len(orphan_supplemental)} of them)"
        if input(f"\n  Delete all {total} tables in {database}{s3_note}? [y/N]: ").strip().lower() != "y":
            print("  skipped")
            continue

        print(f"  DELETING {len(orphan_views)} views and {len(orphan_supplemental)} supplemental tables...")
        delete_tables(database, [v["name"] for v in orphan_views], boto3_session)
        delete_tables(database, [t["name"] for t in orphan_supplemental], boto3_session)
        if args.keep_s3:
            print("  keeping supplemental S3 data (--keep-s3)")
        else:
            delete_s3_data([t["s3_path"] for t in orphan_supplemental if t["s3_path"]], boto3_session)


if __name__ == "__main__":
    main()
