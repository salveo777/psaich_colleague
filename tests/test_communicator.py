import pytest
from app.communicator.communicator import Communicator

@pytest.fixture
def communicator():
    """Fixture to create a Communicator instance for testing."""
    return Communicator(model_url="http://localhost:11434/api/generate", 
                        model_name="mistral",
                        temperature=0.7,
                        max_context_length=100,
                        max_summary_length_percentage=0.75)

def test_add_message(communicator):
    """Test adding a message to the communicator's history."""
    communicator.add_message("user", "Hello, how are you?")
    assert len(communicator.history) == 1
    assert communicator.history[0] == {"role": "user", "content": "Hello, how are you?"}

def test_get_context_under_limit(communicator):
    communicator.add_message("user", 
                             "Hi, can you explain me the concept of quantum computing in simple terms?")
    context = communicator.get_context()
    assert "user: Hi, can you explain me the concept of quantum computing in simple terms?" in context
    assert "Warning" not in context

def test_get_context_over_limit_triggers_summary(communicator, monkeypatch):
    # Patch summarize_history to avoid real LLM call
    communicator.add_message("user", "X" * 150)  # Exceeding the small max_context_length
    monkeypatch.setattr(communicator, "summarize_history", lambda: "summary here")
    context = communicator.get_context()
    assert "Warning" in context
    assert "summary here" in context

def test_reset_session(communicator):
    communicator.add_message("user", "test")
    communicator.reset_session()
    assert communicator.history == []
    assert communicator.summary is None