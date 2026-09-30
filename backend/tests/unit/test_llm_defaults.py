import json

import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src import ai, config as config_module
from src.config import Config, set_config_override


def _isolate_llm_env(monkeypatch):
    monkeypatch.setattr(config_module, "get_cached_setting", lambda _name: None)
    for key in ("LLM", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_anthropic_key_alone_defaults_to_a_current_claude_model(monkeypatch):
    _isolate_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert Config().llm == "anthropic:claude-sonnet-5"


def test_openai_key_defaults_to_the_low_cost_openai_model(monkeypatch):
    _isolate_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # e.g. for YouTube metadata

    assert Config().llm == "openai:gpt-6-luna"


def test_missing_keys_point_at_openai(monkeypatch):
    _isolate_llm_env(monkeypatch)

    assert Config().llm == "openai:gpt-6-luna"


def test_transcript_agent_raises_output_budget_only_for_anthropic(monkeypatch):
    _isolate_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(ai, "_transcript_agent", None)
    monkeypatch.setattr(ai, "_transcript_agent_signature", None)
    runtime_config = Config()
    set_config_override(runtime_config)
    try:
        runtime_config.llm = "anthropic:claude-sonnet-5"
        assert ai.get_transcript_agent().model_settings == {
            "max_tokens": ai.ANTHROPIC_ANALYSIS_MAX_TOKENS
        }

        runtime_config.llm = "google-gla:gemini-3-flash-preview"
        assert ai.get_transcript_agent().model_settings is None
    finally:
        set_config_override(None)


async def test_openai_clip_selection_asks_for_json_schema_output_not_a_function_call(
    monkeypatch,
):
    # gpt-6-* models reject function calls in Chat Completions unless their
    # reasoning is switched off, so OpenAI must get a response format instead.
    requests = []

    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        analysis = {"most_relevant_segments": [], "summary": "Sleep tips.", "key_topics": []}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(analysis)},
                    }
                ],
            },
        )

    _isolate_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ai, "_transcript_agent", None)
    monkeypatch.setattr(ai, "_transcript_agent_signature", None)
    monkeypatch.setattr(
        ai,
        "_build_transcript_model",
        lambda _config: OpenAIChatModel(
            "gpt-6-luna",
            provider=OpenAIProvider(
                api_key="test-key",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
            ),
        ),
    )
    set_config_override(Config())
    try:
        result = await ai.get_transcript_agent().run("[00:00 - 00:20] Sleep tips.")
    finally:
        set_config_override(None)

    assert result.output.summary == "Sleep tips."
    assert "tools" not in requests[0]
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "reasoning_effort" not in requests[0]
