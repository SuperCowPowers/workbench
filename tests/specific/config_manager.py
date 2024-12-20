from pprint import pprint
from workbench.utils.config_manager import ConfigManager


def test_config_manager():
    """Exercise the ConfigManager class"""
    cm = ConfigManager()
    workbench_role = cm.get_config("WORKBENCH_ROLE")
    print(f"WORKBENCH_ROLE: {workbench_role}")
    workbench_plugins = cm.get_config("WORKBENCH_PLUGINS")
    print(f"WORKBENCH_PLUGINS: {workbench_plugins}")

    # License ID
    print(f"WORKBENCH_LICENSE_ID: {cm.get_license_id()}")

    # API Key Info
    my_api_key_info = cm.get_api_key_info()
    pprint(my_api_key_info)

    # All config
    pprint(cm.get_all_config())


def test_running_as_service(monkeypatch):
    # Mock running_as_service to always return True
    monkeypatch.setattr("workbench.utils.config_manager.running_as_service", lambda: True)

    # Reset the ConfigManager class
    ConfigManager._instance = None
    cm = ConfigManager()
    workbench_role = cm.get_config("WORKBENCH_ROLE")
    print(f"WORKBENCH_ROLE: {workbench_role}")
    workbench_plugins = cm.get_config("WORKBENCH_PLUGINS")
    print(f"WORKBENCH_PLUGINS: {workbench_plugins}")

    # License ID
    print(f"WORKBENCH_LICENSE_ID: {cm.get_license_id()}")

    # API Key Info
    my_api_key_info = cm.get_api_key_info()
    pprint(my_api_key_info)

    # All config
    pprint(cm.get_all_config())


if __name__ == "__main__":
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()

    test_config_manager()
    test_running_as_service(monkeypatch)

    # Clean up the monkeypatch fixture
    monkeypatch.undo()
