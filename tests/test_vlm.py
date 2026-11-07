from loratrain.vlm import (
    OLLAMA,
    OPENAI,
    VlmConfig,
    build_ollama_payload,
    build_openai_payload,
    error_message,
    extract_text,
    parse_response,
)


def test_parse_json_inside_prose_and_fences():
    reply = 'Sure, here it is:\n```json\n{"caption": "a cat", "tags": ["fur"]}\n```'

    data = parse_response(reply)

    assert data["caption"] == "a cat"
    assert data["tags"] == ["fur"]


def test_parse_plain_text_becomes_the_caption():
    assert parse_response("  a red bicycle  ")["caption"] == "a red bicycle"


def test_parse_missing_caption_key_is_empty():
    assert parse_response('{"tags": ["x"]}')["caption"] == ""


def test_endpoint_accepts_base_or_full_url():
    assert VlmConfig("http://host:1/v1", "m").endpoint() == "http://host:1/v1/chat/completions"
    assert (
        VlmConfig("http://host:1/v1/chat/completions", "m").endpoint()
        == "http://host:1/v1/chat/completions"
    )


def test_bare_host_gets_the_openai_v1_prefix():
    assert VlmConfig("http://host:1", "m").endpoint() == "http://host:1/v1/chat/completions"


def test_ollama_flavor_uses_the_native_chat_route():
    config = VlmConfig("http://localhost:11434", "qwen2.5vl:7b", flavor="ollama")

    assert config.endpoint() == "http://localhost:11434/api/chat"


def test_ollama_flavor_strips_a_v1_suffix():
    config = VlmConfig("http://localhost:11434/v1", "m", flavor=OLLAMA)

    assert config.endpoint() == "http://localhost:11434/api/chat"


def test_payload_carries_the_image_as_a_data_uri():
    payload = build_openai_payload("model-x", "describe", "data:image/png;base64,AAAA")

    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert payload["temperature"] == 0


def test_ollama_payload_sends_raw_base64_and_json_mode():
    payload = build_ollama_payload("m", "describe", "QUJD")

    assert payload["messages"][0]["images"] == ["QUJD"]
    assert payload["format"] == "json"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "30m"


def test_extract_text_from_an_ollama_reply():
    payload = {"message": {"content": '{"caption": "a cat"}'}, "done": True}

    assert extract_text(payload, OLLAMA) == '{"caption": "a cat"}'


def test_extract_text_from_openai_content_parts():
    payload = {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]}

    assert extract_text(payload, OPENAI) == "hi"


def test_error_message_unwraps_the_json_body():
    assert "not found" in error_message('{"error": "model x not found"}')
