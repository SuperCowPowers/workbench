import pytest

from applications.aws_dashboard.pages.models.navigation import selected_model_url


def test_selected_model_url_updates_model_query_parameter():
    selected_rows = [{"name": "demo-model-v2"}]

    assert selected_model_url(selected_rows, "/models?name=demo-model-v1") == "/models?name=demo-model-v2"


def test_selected_model_url_encodes_model_name():
    selected_rows = [{"name": "demo model/v2"}]

    assert selected_model_url(selected_rows, "/models") == "/models?name=demo+model%2Fv2"


@pytest.mark.parametrize(
    ("selected_rows", "href"),
    [
        ([], "/models?name=demo-model"),
        ([None], "/models?name=demo-model"),
        ([{}], "/models?name=demo-model"),
        ([{"name": "demo-model"}], "/models?name=demo-model"),
        ([{"name": "demo-model"}], "/feature_sets?name=demo-model"),
    ],
)
def test_selected_model_url_prevents_unneeded_navigation(selected_rows, href):
    assert selected_model_url(selected_rows, href) is None
