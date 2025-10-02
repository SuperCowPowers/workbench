import argparse
import logging
import json
from pathlib import Path

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.utils.config_manager import ConfigManager
from workbench.utils.s3_utils import upload_content_to_s3

log = logging.getLogger("workbench")
cm = ConfigManager()
workbench_bucket = cm.get_config("WORKBENCH_BUCKET")


def submit_to_sqs(script_path: str, size: str = "small", realtime: bool = False, recreate: bool = False) -> None:
    """
    Upload script to S3 and submit message to SQS queue for processing.
    Args:
        script_path: Local path to the ML pipeline script
        size: Job size tier - "small" (default), "medium", or "large"
        realtime: If True, sets serverless=False for real-time processing (default: False, meaning serverless=True)
        recreate: If True, sets RECREATE=True in environment (default: False)
    """
    print(f"\n{'=' * 60}")
    print("🚀  SUBMITTING ML PIPELINE JOB")
    print(f"{'=' * 60}")

    if size not in ["small", "medium", "large"]:
        raise ValueError(f"Invalid size '{size}'. Must be 'small', 'medium', or 'large'")
    # Validate script exists
    script_file = Path(script_path)
    if not script_file.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"📄  Script: {script_file.name}")
    print(f"📏  Size tier: {size}")
    print(f"⚡  Mode: {'Real-time' if realtime else 'Serverless'} (serverless={'False' if realtime else 'True'})")
    print(f"🔄  Recreate: {recreate}")
    print(f"🪣  Bucket: {workbench_bucket}")
    sqs = AWSAccountClamp().boto3_session.client("sqs")
    script_name = script_file.name

    # List Workbench queues
    print("\n📋  Listing Workbench SQS queues...")
    try:
        queues = sqs.list_queues(QueueNamePrefix="workbench-")
        queue_urls = queues.get("QueueUrls", [])
        if queue_urls:
            print(f"✅  Found {len(queue_urls)} workbench queue(s):")
            for url in queue_urls:
                queue_name = url.split("/")[-1]
                print(f"   • {queue_name}")
        else:
            print("⚠️  No workbench queues found")
    except Exception as e:
        print(f"❌  Error listing queues: {e}")

    # Upload script to S3
    s3_path = f"s3://{workbench_bucket}/batch-jobs/{script_name}"
    print("\n📤  Uploading script to S3...")
    print(f"   Source: {script_path}")
    print(f"   Destination: {s3_path}")

    try:
        upload_content_to_s3(script_file.read_text(), s3_path)
        print("✅  Script uploaded successfully")
    except Exception as e:
        print(f"❌  Upload failed: {e}")
        raise
    # Get queue URL and info
    queue_name = "workbench-ml-pipeline-queue.fifo"
    print("\n🎯  Getting queue information...")
    print(f"   Queue name: {queue_name}")

    try:
        queue_url = sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]
        print(f"   Queue URL: {queue_url}")

        # Get queue attributes for additional info
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
        )
        messages_available = attrs["Attributes"].get("ApproximateNumberOfMessages", "0")
        messages_in_flight = attrs["Attributes"].get("ApproximateNumberOfMessagesNotVisible", "0")
        print(f"   Messages in queue: {messages_available}")
        print(f"   Messages in flight: {messages_in_flight}")

    except Exception as e:
        print(f"❌  Error accessing queue: {e}")
        raise

    # Prepare message
    message = {"script_path": s3_path, "size": size}

    # Set environment variables
    message["environment"] = {"SERVERLESS": "False" if realtime else "True"}
    if recreate:
        message["environment"]["RECREATE"] = "True"

    print("\n📨  Sending message to SQS...")

    # Send the message to SQS
    try:
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message, indent=2),
            MessageGroupId="ml-pipeline-jobs",  # Required for FIFO
        )
        message_id = response["MessageId"]
        print("✅  Message sent successfully!")
        print(f"   Message ID: {message_id}")
    except Exception as e:
        print(f"❌  Failed to send message: {e}")
        raise

    # Success summary
    print(f"\n{'=' * 60}")
    print("✅  JOB SUBMISSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"📄  Script: {script_name}")
    print(f"📏  Size: {size}")
    print(f"⚡  Mode: {'Real-time' if realtime else 'Serverless'} (SERVERLESS={'False' if realtime else 'True'})")
    print(f"🔄  Recreate: {recreate}")
    print(f"🆔  Message ID: {message_id}")
    print("\n🔍  MONITORING LOCATIONS:")
    print(f"   • SQS Queue: AWS Console → SQS → {queue_name}")
    print("   • Lambda Logs: AWS Console → Lambda → Functions")
    print("   • Batch Jobs: AWS Console → Batch → Jobs")
    print("   • CloudWatch: AWS Console → CloudWatch → Log groups")
    print("\n⏳  Your job should start processing soon...")


def main():
    """CLI entry point for submitting ML pipelines via SQS."""
    parser = argparse.ArgumentParser(description="Submit ML pipeline to SQS queue for Batch processing")
    parser.add_argument("script_file", help="Local path to ML pipeline script")
    parser.add_argument(
        "--size", default="small", choices=["small", "medium", "large"], help="Job size tier (default: small)"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Create realtime endpoints (default is serverless)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Set RECREATE=True (will force recreation of resources)",
    )
    args = parser.parse_args()
    try:
        submit_to_sqs(args.script_file, args.size, realtime=args.realtime, recreate=args.recreate)
    except Exception as e:
        print(f"\n❌  ERROR: {e}")
        log.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
