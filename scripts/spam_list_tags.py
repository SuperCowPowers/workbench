import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sageworks.api import FeatureSet
from sageworks.utils.aws_utils import list_tags_with_throttle

# Set up logging
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


def spam_list_tags(arn, sm_session, index):
    log.info(f"Spamming list_tags_with_throttle: {index}")
    return list_tags_with_throttle(arn, sm_session)


if __name__ == "__main__":

    # Grab our test FeatureSet
    my_features = FeatureSet("abalone_features")
    arn = my_features.arn()
    sm_session = my_features.sm_session

    # Test on an ARN that doesn't exist
    log.info("Testing on an ARN that doesn't exist")
    tags = list_tags_with_throttle("arn:aws:sagemaker:us-west-2:123456789012:fake-arn", sm_session)

    # Run spamming in parallel to induce throttling
    log.info("Spamming list_tags_with_throttle")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(spam_list_tags, arn, sm_session, i) for i in range(1000)]

        # Collect results
        for future in as_completed(futures):
            try:
                tags = future.result()
            except Exception as e:
                log.error(f"Encountered an error: {e}")
