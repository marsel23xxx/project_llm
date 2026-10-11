from app.core.config import Settings
from app.services.llm.router import LLMRouter


def _router(mode: str, key: str | None) -> LLMRouter:
    return LLMRouter(Settings(llm_mode=mode, anthropic_api_key=key))


def test_local_mode_never_uses_claude():
    router = _router("local", "sk-test")
    assert [p.name for p in router.providers_for("complex")] == ["local"]


def test_hybrid_routes_by_task_kind():
    router = _router("hybrid", "sk-test")
    assert [p.name for p in router.providers_for("simple")] == ["local", "claude"]
    assert [p.name for p in router.providers_for("complex")] == ["claude", "local"]


def test_missing_api_key_falls_back_to_local():
    router = _router("api", None)
    assert [p.name for p in router.providers_for("complex")] == ["local"]
