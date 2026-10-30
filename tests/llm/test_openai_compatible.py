import json
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import MagicMock, patch

import pytest

from llm.base import LLMProviderError
from llm.providers.openai_compatible import OpenAICompatibleClient


def test_generate_omits_temperature_override_for_reasoning_models():
    client = OpenAICompatibleClient("test-key", model="gpt-6.1-terra")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = (
        b'{"choices":[{"message":{"content":"Grounded answer"}}]}'
    )

    with patch(
        "llm.providers.openai_compatible.urlopen", return_value=response
    ) as urlopen:
        result = client.generate("system", "user")

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)
    assert "temperature" not in payload
    assert result == "Grounded answer"


def test_generate_includes_provider_message_on_http_error():
    client = OpenAICompatibleClient("test-key", model="gpt-6.1-terra")
    body = json.dumps(
        {
            "error": {
                "message": "Unsupported parameter: temperature",
                "code": "unsupported_parameter",
                "param": "temperature",
            }
        }
    ).encode()
    error = HTTPError(
        "https://api.openai.com/v1/chat/completions",
        400,
        "Bad Request",
        {},
        BytesIO(body),
    )

    with patch("llm.providers.openai_compatible.urlopen", side_effect=error):
        with pytest.raises(LLMProviderError, match="Unsupported parameter: temperature"):
            client.generate("system", "user")
