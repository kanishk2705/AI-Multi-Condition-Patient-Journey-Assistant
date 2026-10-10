import pytest

from patient_journey_assistant.text_processing import (
    TextExtractionError,
    extract_medlineplus_medical_test,
    extract_medlineplus_topic_summary,
    normalize_text,
)


def test_topic_summary_extractor_stops_at_next_major_section():
    html = """
    <html><body><main>
      <h1>Asthma</h1>
      <h2>Summary</h2>
      <h3>What is asthma?</h3><p>Asthma affects the airways and can cause wheezing, coughing, chest tightness, and shortness of breath. Symptoms and triggers may vary between people and over time.</p>
      <h3>Why does it matter?</h3><p>People may have different experiences. A healthcare professional can answer questions about an individual's circumstances and help explain the information.</p>
      <h2>Start Here</h2><p>This section must not be included.</p>
    </main></body></html>
    """
    result = extract_medlineplus_topic_summary(html)
    assert "Asthma affects the airways" in result
    assert "Symptoms and triggers may vary" in result
    assert "This section must not be included" not in result


def test_topic_summary_fails_closed_when_heading_missing():
    with pytest.raises(TextExtractionError):
        extract_medlineplus_topic_summary("<html><body><p>no summary</p></body></html>")


def test_medical_test_extractor_ignores_site_navigation_and_scripts():
    html = """
    <html><body><nav>Navigation that should be removed</nav>
    <main><h1>Test title</h1><p>This medical test article provides enough text to pass the minimum threshold.
    It gives general information about the test and what a person might expect when a provider orders it.
    It does not interpret results or provide diagnosis, and the reader should discuss questions with a healthcare professional.</p>
    <script>not medical information</script></main></body></html>
    """
    result = extract_medlineplus_medical_test(html)
    assert "medical test article" in result
    assert "Navigation that should be removed" not in result
    assert "not medical information" not in result


def test_normalize_text_collapses_duplicate_lines():
    assert normalize_text(" A   B \n\n A   B \n C ") == "A B\n\nC"
def test_normalize_text_removes_spaces_before_punctuation():
    assert normalize_text("disease . coughing , tightness !") == (
        "disease. coughing, tightness!"
    )


def test_topic_summary_extractor_handles_utf8_html_bytes():
    html = """
    <html><body><main>
      <h2>Summary</h2>
      <p>Type 2 diabetes&nbsp;affects how the body uses glucose.
      Glucose is an important source of energy. Information about this
      condition can help people understand the condition and prepare
      questions to discuss with a qualified healthcare professional.
      General information does not replace individual medical advice.</p>
    </main></body></html>
    """.encode("utf-8")

    result = extract_medlineplus_topic_summary(html)

    assert "Â" not in result
    assert "Type 2 diabetes affects" in result
def test_medical_test_extractor_stops_before_references():
    html = """
    <html><body><main>
      <h1>Blood Glucose Test</h1>
      <p>This medical test measures glucose in the blood and is used
      to help evaluate blood sugar levels. The test may be used in
      different circumstances, and preparation depends on the type of
      test ordered. People should ask their healthcare professional
      about preparation and questions concerning their results.</p>
      <h2>References</h2>
      <p>Third-party bibliography material must not enter the corpus.</p>
    </main></body></html>
    """

    result = extract_medlineplus_medical_test(html)

    assert "This medical test measures glucose" in result
    assert "Third-party bibliography material" not in result