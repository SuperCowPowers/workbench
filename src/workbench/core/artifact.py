"""Artifact: Abstract Base Class for all Artifact classes in Workbench.

``Artifact`` is the storage-agnostic contract: naming rules, the abstract
method set, and every helper that can be expressed in terms of
``workbench_meta()``/``upsert_workbench_meta()`` (tags, owner, input, status,
health). ``AWSArtifact`` backs that metadata with AWS tags and provides the
shared AWS session/bucket resources; ``workbench.local.LocalArtifact`` backs it
with an on-disk ``meta.json``.
"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Union


class Artifact(ABC):
    """Artifact: Abstract Base Class for all Artifact classes in Workbench"""

    # Class-level shared resources
    log = logging.getLogger("workbench")

    # Delimiter for storing lists in metadata
    tag_delimiter = "::"

    def __init__(self, name: str, **kwargs):
        """Initialize the Artifact Base Class

        Args:
            name (str): The Name of this artifact
        """
        self.name = name

    def __post_init__(self):
        """Artifact Post Initialization"""

        # Do I exist? (very metaphysical)
        if not self.exists():
            self.log.debug(f"Artifact {self.name} does not exist")
            return

        # Conduct a Health Check on this Artifact
        health_issues = self.health_check()
        if health_issues:
            if "needs_onboard" in health_issues:
                self.log.important(f"Artifact {self.name} needs to be onboarded")
            elif health_issues == ["no_activity"]:
                self.log.debug(f"Artifact {self.name} has no activity, which is fine")
            else:
                self.log.warning(f"Health Check Failed {self.name}: {health_issues}")
            for issue in health_issues:
                self.add_health_tag(issue)
        else:
            self.log.info(f"Health Check Passed {self.name}")

    @classmethod
    def is_name_valid(cls, name: str, delimiter: str = "_", lower_case: bool = True) -> bool:
        """Check if the name adheres to the naming conventions for this Artifact.

        Args:
            name (str): The name/id to check.
            delimiter (str): The delimiter to use in the name/id string (default: "_")
            lower_case (bool): Should the name be lowercased? (default: True)

        Returns:
            bool: True if the name is valid, False otherwise.
        """
        valid_name = cls.generate_valid_name(name, delimiter=delimiter, lower_case=lower_case)
        if name != valid_name:
            cls.log.warning(f"Artifact name: '{name}' is not valid. Convert it to something like: '{valid_name}'")
            return False
        return True

    @staticmethod
    def generate_valid_name(name: str, delimiter: str = "_", lower_case: bool = True) -> str:
        """Only allow letters and the specified delimiter, also lowercase the string.

        Args:
            name (str): The name/id string to check.
            delimiter (str): The delimiter to use in the name/id string (default: "_")
            lower_case (bool): Should the name be lowercased? (default: True)

        Returns:
            str: A generated valid name/id.
        """
        valid_name = "".join(c for c in name if c.isalnum() or c in ["_", "-"])
        if lower_case:
            valid_name = valid_name.lower()

        # Replace with the chosen delimiter
        return valid_name.replace("_", delimiter).replace("-", delimiter)

    @abstractmethod
    def exists(self) -> bool:
        """Does the Artifact exist? Can we connect to it?"""
        pass

    @abstractmethod
    def workbench_meta(self) -> Union[dict, None]:
        """Get the Workbench specific metadata for this Artifact

        Returns:
            Union[dict, None]: Dictionary of Workbench metadata for this Artifact
        """
        pass

    @abstractmethod
    def upsert_workbench_meta(self, new_meta: dict):
        """Add Workbench specific metadata to this Artifact

        Args:
            new_meta (dict): Dictionary of NEW metadata to add
        """
        pass

    def expected_meta(self) -> list[str]:
        """Metadata we expect to see for this Artifact when it's ready
        Returns:
            list[str]: List of expected metadata keys
        """

        # If an artifact has additional expected metadata override this method
        return ["workbench_status"]

    @abstractmethod
    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        pass

    def ready(self) -> bool:
        """Is the Artifact ready? Is initial setup complete and expected metadata populated?"""

        # If anything goes wrong, assume the artifact is not ready
        try:
            # Check for the expected metadata
            expected_meta = self.expected_meta()
            existing_meta = self.workbench_meta()
            ready = set(existing_meta.keys()).issuperset(expected_meta)
            if ready:
                return True
            else:
                self.log.info("Artifact is not ready!")
                return False
        except Exception as e:
            self.log.error(f"Artifact malformed: {e}")
            return False

    @abstractmethod
    def onboard(self) -> bool:
        """Onboard this Artifact into Workbench
        Returns:
            bool: True if the Artifact was successfully onboarded, False otherwise
        """
        pass

    @abstractmethod
    def details(self) -> dict:
        """Additional Details about this Artifact"""
        pass

    @abstractmethod
    def size(self) -> float:
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
    def hash(self) -> str:
        """Return the hash of this artifact, useful for content validation"""
        pass

    @abstractmethod
    def delete(self):
        """Delete this artifact including all related objects"""
        pass

    def get_tags(self, tag_type="user") -> list:
        """Get the tags for this artifact
        Args:
            tag_type (str): Type of tags to return (user or health)
        Returns:
            list[str]: List of tags for this artifact
        """
        if tag_type == "user":
            user_tags = self.workbench_meta().get("workbench_tags")
            return user_tags.split(self.tag_delimiter) if user_tags else []

        # Grab our health tags
        health_tags = self.workbench_meta().get("workbench_health_tags")

        # If we don't have health tags, create the storage and return an empty list
        if health_tags is None:
            self.log.important(f"{self.name} creating workbench_health_tags storage...")
            self.upsert_workbench_meta({"workbench_health_tags": ""})
            return []

        # Otherwise, return the health tags
        return health_tags.split(self.tag_delimiter) if health_tags else []

    def set_tags(self, tags):
        self.upsert_workbench_meta({"workbench_tags": self.tag_delimiter.join(tags)})

    def add_tag(self, tag, tag_type="user"):
        """Add a tag for this artifact, ensuring no duplicates and maintaining order.
        Args:
            tag (str): Tag to add for this artifact
            tag_type (str): Type of tag to add (user or health)
        """
        current_tags = self.get_tags(tag_type) if tag_type == "user" else self.get_health_tags()
        if tag not in current_tags:
            current_tags.append(tag)
            combined_tags = self.tag_delimiter.join(current_tags)
            if tag_type == "user":
                self.upsert_workbench_meta({"workbench_tags": combined_tags})
            else:
                self.upsert_workbench_meta({"workbench_health_tags": combined_tags})

    def remove_workbench_tag(self, tag, tag_type="user"):
        """Remove a tag from this artifact if it exists.
        Args:
            tag (str): Tag to remove from this artifact
            tag_type (str): Type of tag to remove (user or health)
        """
        current_tags = self.get_tags(tag_type) if tag_type == "user" else self.get_health_tags()
        if tag in current_tags:
            current_tags.remove(tag)
            combined_tags = self.tag_delimiter.join(current_tags)
            if tag_type == "user":
                self.upsert_workbench_meta({"workbench_tags": combined_tags})
            elif tag_type == "health":
                self.upsert_workbench_meta({"workbench_health_tags": combined_tags})

    # Syntactic sugar for health tags
    def get_health_tags(self):
        return self.get_tags(tag_type="health")

    def set_health_tags(self, tags):
        self.upsert_workbench_meta({"workbench_health_tags": self.tag_delimiter.join(tags)})

    def add_health_tag(self, tag):
        self.add_tag(tag, tag_type="health")

    def remove_health_tag(self, tag):
        self.remove_workbench_tag(tag, tag_type="health")

    # Owner of this artifact
    def get_owner(self) -> str:
        """Get the owner of this artifact"""
        return self.workbench_meta().get("workbench_owner", "unknown")

    def set_owner(self, owner: str):
        """Set the owner of this artifact

        Args:
            owner (str): Owner to set for this artifact
        """
        self.upsert_workbench_meta({"workbench_owner": owner})

    def get_input(self) -> str:
        """Get the input data for this artifact"""
        return self.workbench_meta().get("workbench_input", "unknown")

    def get_status(self) -> str:
        """Get the status for this artifact"""
        return self.workbench_meta().get("workbench_status", "unknown")

    def set_status(self, status: str):
        """Set the status for this artifact
        Args:
            status (str): Status to set for this artifact
        """
        self.upsert_workbench_meta({"workbench_status": status})

    def health_check(self, deep: bool = False) -> list[str]:
        """Perform a health check on this artifact

        Args:
            deep (bool): If True, perform more extensive (expensive) health checks (default: False)

        Returns:
            list[str]: List of health issues
        """
        health_issues = []
        if not self.ready():
            return ["needs_onboard"]
        # FIXME: Revisit AWS URL check ("unknown" in aws_url() -> "aws_url_unknown" health issue)
        return health_issues

    def summary(self) -> dict:
        """This is generic summary information for all Artifacts. If you
        want to get more detailed information, call the details() method
        which is implemented by the specific Artifact class"""
        basic = {
            "name": self.name,
            "health_tags": self.get_health_tags(),
            "size": self.size(),
            "created": self.created(),
            "modified": self.modified(),
            "input": self.get_input(),
        }
        # Combine the workbench metadata with the basic metadata
        return {**basic, **self.workbench_meta()}

    def __repr__(self) -> str:
        """String representation of this artifact

        Returns:
            str: String representation of this artifact
        """

        # If the artifact does not exist, return a message
        if not self.exists():
            return f"{self.__class__.__name__}: {self.name} does not exist"

        summary_dict = self.summary()
        display_keys = [
            "aws_arn",
            "health_tags",
            "size",
            "created",
            "modified",
            "input",
            "workbench_status",
            "workbench_tags",
        ]
        summary_items = [f"  {repr(key)}: {repr(value)}" for key, value in summary_dict.items() if key in display_keys]
        summary_str = f"{self.__class__.__name__}: {self.name}\n" + ",\n".join(summary_items)
        return summary_str
