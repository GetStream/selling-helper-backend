from google import genai
from google.genai import types

from workflow.events import EventEmitter


class GoogleSearchClient:
    """Direct SDK client for Google Search API calls."""

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
        event_emitter: EventEmitter,
    ) -> None:
        self._event_emitter = event_emitter
        self._model = model
        self._client = genai.Client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        self._config = types.GenerateContentConfig(
            tools=[grounding_tool],
            response_json_schema=types.JsonSchema(
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "specifications": {"type": "string"},
                        "pricing": {"type": "string"},
                    },
                }
            ),
        )

    def perform_google_search(self, *, query: str) -> str:
        """Perform a Google Search and return the results."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=query,
            config=self._config,
        )
        print(response.model_dump(mode="json", exclude_none=False))
        return response.text
