from pprint import pprint
from sageworks.utils.config_manager import ConfigManager


def test_config_manager():
    """Exercise the ConfigManager class"""
    cm = ConfigManager()
    sageworks_role = cm.get_config("SAGEWORKS_ROLE")
    print(f"SAGEWORKS_ROLE: {sageworks_role}")
    sageworks_plugins = cm.get_config("SAGEWORKS_PLUGINS")
    print(f"SAGEWORKS_PLUGINS: {sageworks_plugins}")

    # License ID
    print(f"SAGEWORKS_LICENSE_ID: {cm.get_license_id()}")

    # API Key Info
    my_api_key_info = cm.get_api_key_info()
    pprint(my_api_key_info)

    # All config
    pprint(cm.get_all_config())


def test_running_as_service(monkeypatch):
    # Mock running_as_service to always return True
    monkeypatch.setattr("sageworks.utils.config_manager.running_as_service", lambda: True)

    # Reset the ConfigManager class
    ConfigManager._instance = None
    cm = ConfigManager()
    sageworks_role = cm.get_config("SAGEWORKS_ROLE")
    print(f"SAGEWORKS_ROLE: {sageworks_role}")
    sageworks_plugins = cm.get_config("SAGEWORKS_PLUGINS")
    print(f"SAGEWORKS_PLUGINS: {sageworks_plugins}")

    # License ID
    print(f"SAGEWORKS_LICENSE_ID: {cm.get_license_id()}")

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
