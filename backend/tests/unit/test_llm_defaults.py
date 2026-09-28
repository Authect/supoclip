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
