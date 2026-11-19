"""Project sessions: reports owned by a named project rather than a user.

The invariants: a project session lives apart from user sessions, anyone's save lands
on the same report, and the report closes with exactly one line naming who saved last.
"""

from workbench.agent.tools import dispatch
from workbench.utils import bosco_utils


def as_user(monkeypatch, user: str):
    monkeypatch.setattr(bosco_utils, "current_user", lambda: user)


def test_project_session_lives_apart_from_user_sessions(monkeypatch):
    as_user(monkeypatch, "briford")
    assert bosco_utils.save_session("analog-split", "mine") == "/workbench/bosco/sessions/briford/analog-split"
    path = bosco_utils.save_session("analog-split", "ours", project="PXR")
    assert path == "/workbench/bosco/projects/pxr/analog-split"

    assert bosco_utils.read_session("analog-split") == "mine"
    assert bosco_utils.read_session("analog-split", project="pxr").startswith("ours")
    assert bosco_utils.list_sessions(all_users=True)["session"].tolist() == ["briford/analog-split"]


def test_any_user_updates_the_same_report_and_the_last_saver_is_named(monkeypatch):
    as_user(monkeypatch, "briford")
    bosco_utils.save_session("analog-split", "v1", project="pxr")
    assert bosco_utils.read_session("analog-split", project="pxr") == "v1\n\n_Saved by briford_"

    # A colleague reads it, updates it, and saves it back with the old closing line intact
    as_user(monkeypatch, "eromm")
    report = bosco_utils.read_session("analog-split", project="pxr").replace("v1", "v2")
    bosco_utils.save_session("analog-split", report, project="pxr")
    assert bosco_utils.read_session("analog-split", project="pxr") == "v2\n\n_Saved by eromm_"


def test_listing_by_project_and_across_projects(monkeypatch):
    as_user(monkeypatch, "briford")
    bosco_utils.save_session("analog-split", "a", project="pxr")
    bosco_utils.save_session("ordering", "b", project="cyp")
    bosco_utils.save_session("scratch", "c")

    assert bosco_utils.list_sessions(project="pxr")["session"].tolist() == ["pxr/analog-split"]
    assert sorted(bosco_utils.list_sessions(projects=True)["session"]) == ["cyp/ordering", "pxr/analog-split"]
    assert bosco_utils.recent_sessions(project="cyp")["session"].tolist() == ["cyp/ordering"]

    # A listing row addresses the session it names
    assert bosco_utils.read_session("pxr/analog-split", project="pxr").startswith("a")


def test_missing_project_session_names_the_project_sessions(monkeypatch):
    as_user(monkeypatch, "briford")
    bosco_utils.save_session("analog-split", "a", project="pxr")
    bosco_utils.save_session("scratch", "c")
    message = bosco_utils.read_session("nope", project="pxr")
    assert "pxr/analog-split" in message and "scratch" not in message


def test_tools_carry_the_project(monkeypatch):
    as_user(monkeypatch, "briford")
    saved = dispatch("save_session", {"name": "analog-split", "report": "ours", "project": "pxr"}, {})
    assert saved == "Saved to /workbench/bosco/projects/pxr/analog-split"
    assert dispatch("read_session", {"name": "analog-split", "project": "pxr"}, {}).startswith("ours")
    assert (
        dispatch("save_session", {"name": "mine", "report": "r"}, {})
        == "Saved to /workbench/bosco/sessions/briford/mine"
    )
