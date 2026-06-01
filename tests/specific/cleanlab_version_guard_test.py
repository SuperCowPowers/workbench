from workbench.algorithms.models.cleanlab_model import _is_old_cleanlab_with_new_datasets


def test_cleanlab_2_8_allows_datasets_4():
    assert not _is_old_cleanlab_with_new_datasets("2.8.0", "4.0.0")


def test_cleanlab_2_9_allows_current_datasets():
    assert not _is_old_cleanlab_with_new_datasets("2.9.0", "4.8.5")


def test_cleanlab_2_7_blocks_datasets_4():
    assert _is_old_cleanlab_with_new_datasets("2.7.1", "4.0.0")


def test_cleanlab_2_7_allows_datasets_3():
    assert not _is_old_cleanlab_with_new_datasets("2.7.1", "3.6.0")
