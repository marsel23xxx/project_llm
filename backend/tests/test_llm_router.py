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


def _remote_router(mode: str) -> LLMRouter:
    return LLMRouter(Settings(llm_mode=mode, anthropic_api_key=None, remote_llm_api_key="sk-remote"))


def test_remote_provider_is_the_hosted_model_when_claude_is_not_configured():
    router = _remote_router("api")
    assert [p.name for p in router.providers_for("complex")] == ["remote", "local"]
    assert router.has_strong_model and router.hosted_model == "deepseek-flash"
    assert [p.name for p in _remote_router("hybrid").providers_for("simple")] == ["local", "remote"]
    assert [p.name for p in _remote_router("local").providers_for("complex")] == ["local"]


def test_claude_takes_precedence_when_both_keys_are_set():
    router = LLMRouter(Settings(llm_mode="api", anthropic_api_key="sk-ant", remote_llm_api_key="sk-remote"))
    assert [p.name for p in router.providers_for("complex")] == ["claude", "local"]


async def test_remote_request_shape_and_fallback_on_empty_balance():
    import json

    import httpx

    from app.services.llm.base import LLMError
    from app.services.llm.remote import RemoteProvider

    seen = {}

    def ok(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "SELECT 1", "reasoning_content": None}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    provider = RemoteProvider("https://api.deepseek.com/", "sk-remote", "deepseek-flash")
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(ok), base_url="https://api.deepseek.com", headers=provider._client.headers)
    result = await provider.complete("sistem", [{"role": "user", "content": "halo"}], 100)
    assert (result.text, result.provider, result.input_tokens, result.output_tokens) == ("SELECT 1", "remote", 12, 3)
    assert seen["url"] == "https://api.deepseek.com/chat/completions" and seen["auth"] == "Bearer sk-remote"
    assert seen["body"]["thinking"] == {"type": "disabled"} and seen["body"]["messages"][0]["role"] == "system"

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(402, text="Insufficient Balance")), base_url="https://api.deepseek.com")
    try:
        await provider.complete("s", [{"role": "user", "content": "x"}], 10)
        raise AssertionError("harus gagal")
    except LLMError as exc:
        assert "Saldo" in str(exc)
