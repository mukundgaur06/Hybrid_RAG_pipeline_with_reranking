"""Unit tests for Streamlit user interface using AppTest.

Tests initialization, empty-state rendering, sidebar controls,
and error handling completely offline without network requests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class MockDenseModel:
    def encode(self, texts, convert_to_numpy=True):
        return np.ones((len(texts), 3), dtype=np.float32)


class MockCrossEncoder:
    def predict(self, pairs, batch_size=16):
        return np.array([2.5] * len(pairs), dtype=np.float32)


@patch("hybrid_rag.ui.app.get_dense_embedding_model", return_value=MockDenseModel())
@patch("hybrid_rag.ui.app.get_cross_encoder_model", return_value=MockCrossEncoder())
def test_app_initialization_and_empty_state(mock_ce, mock_dense):
    """Verify that the app loads cleanly without uncaught exceptions in its initial state."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Verify no unhandled crashes
    assert not at.exception

    # Verify key UI components exist
    assert len(at.sidebar.button) >= 2  # Index Files, Load Samples, Clear Index
    assert len(at.text_input) >= 1      # Query text input

    metrics = at.sidebar.metric
    chunk_metric = next(m for m in metrics if m.label == "Total Chunks")
    assert int(chunk_metric.value) == 0

    # Verify initial welcome info box is rendered
    assert len(at.info) >= 1
    assert "Welcome!" in at.info[0].value


@patch("hybrid_rag.ui.app.get_dense_embedding_model", return_value=MockDenseModel())
@patch("hybrid_rag.ui.app.get_cross_encoder_model", return_value=MockCrossEncoder())
def test_app_query_without_index_shows_warning(mock_ce, mock_dense):
    """Verify querying an empty index warns the user instead of crashing."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Enter a question without indexing documents
    at.text_input[0].input("What is Apollo 11?").run()

    # Click Run Pipeline
    run_button = next(b for b in at.button if "Run Pipeline" in b.label)
    run_button.click().run()

    assert not at.exception
    # Should display an error that no documents are indexed
    assert len(at.error) >= 1
    assert "No documents indexed" in at.error[0].value
