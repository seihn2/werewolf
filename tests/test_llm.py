import pytest

from wolfplay.llm import ChatModelConfig, _extract_json


def test_extract_json_from_fenced_or_reasoning_response():
    assert _extract_json('```json\n{"answer": 1}\n```') == {"answer": 1}
    assert _extract_json('thinking first\n{"answer": 2}\nfinished') == {"answer": 2}


def test_chat_model_config_validates_endpoint():
    with pytest.raises(ValueError, match="http"):
        ChatModelConfig(base_url="localhost:8000", api_key="key", model="model")
