import pytest

from patient_journey_assistant.ingest import IngestionError, _check_robots


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
