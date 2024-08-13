from sageworks.utils.config_manager import ConfigManager


def test_config_manager():
    """Exercise the ConfigManager class"""
    from pprint import pprint

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

    test_config_manager()
