from rich.console import Console
from pathlib import Path
import configparser
import click
import os
import uuid


console = Console()

###############################################################################
# Config file
###############################################################################


def get_config_file_from_env():
    return os.environ.get("SAGEWORKS_CONFIG_FILE", "")


@click.command()
@click.option(
    "--bucket-name",
    prompt="S3 Bucket base name",
    default="",
    help="The base name for the S3 bucket to be created. A random uuid will be appended to this name.",
)
@click.option(
    "--file-path",
    prompt="Path to config file",
    default=get_config_file_from_env,
    help="The path to the local SageWorks config file.",
)
def cli_create_config(bucket_name, file_path):
    """Create Sageworks config file"""
    bucket_name += "-" + str(uuid.uuid4())
    bucket_name = bucket_name.lower()
    if len(bucket_name) > 63:
        bucket_name = bucket_name[0:63]
    if not file_path or not Path(file_path).is_file():
        file_path = str(Path().home() / ".config" / "sageworks" / "sageworks_config.ini")
    config = configparser.ConfigParser()
    config["SAGEWORKS_AWS"] = {
        "S3_BUCKET_NAME": bucket_name,
        "SAGEWORKS_ROLE_NAME": "SageWorks-ExecutionRole",
    }
    config["SAGEWORKS_REDIS"] = {"HOST": "localhost", "PORT": "6379", "PASSWORD": ""}
    with open(file_path, "w") as f:
        config.write(f)
    console.print(
        f"SageWorks config file successfully created at {file_path}\n",
        style="bold green",
    )


msg = r"""
   ____             _      __         __
  / __/__ ____ ____| | /| / /__  ____/ /__ ___
 _\ \/ _ `/ _ `/ -_) |/ |/ / _ \/ __/  '_/(_-<
/___/\_,_/\_, /\__/|__/|__/\___/_/ /_/\_\/___/
         /___/   """


@click.group()
@click.pass_context
def cli(ctx):
    console.print(msg, style="rgb(109,125,176)", highlight=False)
    console.print(":zap: Welcome to SageWorks! :zap:", style="bold blue")
    console.print("")
    if ctx.invoked_subcommand == "config":
        console.print("Creating SageWorks Config File...")


cli.add_command(cli_create_config, name="config")


if __name__ == "__main__":
    cli()
