"""Artifact: Abstract Base Class for all Artifact classes in SageWorks.
                Artifacts simply reflect and aggregate one or more AWS Services"""
from abc import ABC, abstractmethod
from datetime import datetime
import logging

# SageWorks Imports
from sageworks.aws_service_broker.aws_sageworks_role_manager import AWSSageWorksRoleManager
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class Artifact(ABC):
    def __init__(self):
        """Artifact: Abstract Base Class for all Artifact classes in SageWorks.
                        Artifacts simply reflect and aggregate one or more AWS Services"""
        self.log = logging.getLogger(__name__)

        # FIXME: We should have this come from AWS or Config
        self.data_catalog_db = 'sageworks'
        self.data_source_s3_path = 's3://scp-sageworks-artifacts/data-sources'
        self.feature_sets_s3_path = 's3://scp-sageworks-artifacts/feature-sets'

        # Grab a SageWorks Boto3 Session
        self.boto_session = AWSSageWorksRoleManager().boto_session()

    @abstractmethod
    def check(self) -> bool:
        """Does the Artifact exist? Can we connect to it?"""
        pass

    @abstractmethod
    def uuid(self) -> int:
        """Return the unique SageWorks identifier for this artifact"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the size of this artifact in MegaBytes"""
        pass

    @abstractmethod
    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        pass

    @abstractmethod
    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        pass

    @abstractmethod
    def meta(self):
        """Get the metadata for this artifact"""
        pass

    @abstractmethod
    def tags(self):
        """Get the tags for this artifact"""
        pass

    @abstractmethod
    def add_tag(self, tag):
        """Add a tag to this artifact"""
        pass

    @abstractmethod
    def aws_url(self):
        """AWS console/web interface for this artifact"""
        pass
