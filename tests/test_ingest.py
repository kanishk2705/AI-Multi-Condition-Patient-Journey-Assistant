from pathlib import Path

import pytest

from patient_journey_assistant.ingest import IngestionError, _check_robots, ingest_approved_sources


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.response


def test_robots_404_means_no_published_robots_file():
    session = FakeSession(FakeResponse(404))
    _check_robots("https://medlineplus.gov/asthma.html", session)
    assert session.calls == ["https://medlineplus.gov/robots.txt"]


def test_robots_disallow_is_respected():
    response = FakeResponse(
        200,
        "User-agent: *\nDisallow: /lab-tests/",
    )
    session = FakeSession(response)
    with pytest.raises(IngestionError, match="robots.txt disallows"):
        _check_robots("https://medlineplus.gov/lab-tests/lung-function-tests/", session)


def test_dry_run_validates_manifest_without_network_or_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    session = FakeSession(FakeResponse(200, "User-agent: *\\n"))
    report = ingest_approved_sources(
        root / "data" / "source_manifest.csv",
        tmp_path / "output",
        fetch=False,
        session=session,
    )
    assert report["manifest_rows"] == 9
    assert report["approved_rows"] == 5
    assert len(report["skipped_rows"]) == 4
    assert session.calls == []
    assert not (tmp_path / "output").exists()
