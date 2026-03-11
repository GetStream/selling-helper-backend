"""Tests for the Google Search integration."""

from __future__ import annotations

import importlib
import os
import sys
from unittest import SkipTest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


class FakeGoogleSearch:
    def __init__(self) -> None:
        pass


class FakeTool:
    def __init__(self, *, google_search: FakeGoogleSearch) -> None:
        self.google_search = google_search


class FakeJsonSchema:
    def __init__(self, *, schema: dict) -> None:
        self.schema = schema


class FakeGenerateContentConfig:
    def __init__(self, *, tools: list[FakeTool], response_json_schema: FakeJsonSchema) -> None:
        self.tools = tools
        self.response_json_schema = response_json_schema


def _load_google_search_module():
    fake_google = ModuleType("google")
    fake_genai = ModuleType("google.genai")
    fake_genai.Client = Mock
    fake_genai.types = SimpleNamespace(
        Tool=FakeTool,
        GoogleSearch=FakeGoogleSearch,
        GenerateContentConfig=FakeGenerateContentConfig,
        JsonSchema=FakeJsonSchema,
    )
    fake_google.genai = fake_genai

    with patch.dict(
        sys.modules,
        {
            "google": fake_google,
            "google.genai": fake_genai,
        },
    ):
        sys.modules.pop("integrations.google_search", None)
        return importlib.import_module("integrations.google_search")


def test_perform_google_search_calls_sdk_and_returns_text() -> None:
    google_search_module = _load_google_search_module()

    mock_response = Mock()
    mock_response.text = "Product details"
    mock_response.model_dump.return_value = {}

    mock_generate_content = Mock(return_value=mock_response)
    mock_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=mock_generate_content)
    )

    with patch.object(google_search_module.genai, "Client", return_value=mock_client):
        search_client = google_search_module.GoogleSearchClient(event_emitter=Mock())

    result = search_client.perform_google_search(query="wireless headphones")

    assert result == "Product details"
    mock_generate_content.assert_called_once_with(
        model="gemini-3.1-flash-image-preview",
        contents="wireless headphones",
        config=search_client._config,
    )
    assert len(search_client._config.tools) == 1
    assert search_client._config.tools[0].google_search is not None
    assert search_client._config.response_json_schema.schema["properties"] == {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "specifications": {"type": "string"},
        "pricing": {"type": "string"},
    }


def test_perform_google_search_with_real_api() -> None:
    if os.getenv("RUN_REAL_GOOGLE_SEARCH_TEST") != "1":
        raise SkipTest("Set RUN_REAL_GOOGLE_SEARCH_TEST=1 to execute the real API test.")

    if not os.getenv("GEMINI_API_KEY"):
        raise SkipTest("Set GEMINI_API_KEY to execute the real API test.")

    google_search_module = importlib.import_module("integrations.google_search")
    search_client = google_search_module.GoogleSearchClient(event_emitter=Mock())

    result = search_client.perform_google_search(
        query="Find product details for Apple AirPods Pro 2 and summarize them."
    )

    assert isinstance(result, str)
    assert result.strip()
