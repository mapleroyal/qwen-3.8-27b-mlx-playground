from __future__ import annotations

import base64

import pytest

from server.app import (
    BackendManager,
    QwenThinkingFilter,
    _bounded_generation_tokens,
    _count_text_tokens,
    _context_budget_from_error,
    _has_images,
    _normalized_metrics,
    _reasoning_control,
    _valid_session_id,
    _validate_inline_images,
)


def test_session_ids_are_safe_for_cache_affinity_headers():
    assert _valid_session_id("session-123:abc")
    assert not _valid_session_id("")
    assert not _valid_session_id("contains a space")
    assert not _valid_session_id("contains\na newline")
    assert not _valid_session_id("x" * 129)


def test_full_context_generation_is_bounded_by_the_formatted_prompt():
    assert _bounded_generation_tokens(262_144, 18, 262_144) == 262_126
    assert _bounded_generation_tokens(262_144, 27, 262_144) == 262_117
    assert _context_budget_from_error(
        "Request needs 262162 context tokens "
        "(18 prompt + 262144 max generation), but MAX_KV_SIZE is 262144."
    ) == (18, 262_126)

    with pytest.raises(ValueError, match="leaving no room"):
        _bounded_generation_tokens(262_144, 262_144, 262_144)


def test_thinking_filter_handles_prefilled_opener_and_split_close_marker():
    stream = QwenThinkingFilter(thinking=True)

    first = stream.feed("Work through the problem.</thi")
    second = stream.feed("nk>\n\nThe answer is 42.")

    reasoning = "".join(
        value
        for field, value in [*first, *second]
        if field == "reasoning_content"
    )
    assert reasoning == "Work through the problem."
    assert ("content", "The answer is 42.") in second


def test_thinking_filter_passes_direct_mode_without_buffering():
    stream = QwenThinkingFilter(thinking=False)
    assert stream.feed("Direct answer") == [("content", "Direct answer")]


def test_normalizes_dspark_and_mtp_metrics():
    dspark = _normalized_metrics(
        {
            "x_mlx_dspark": {
                "tokens_per_sec": 51.2,
                "accept_len": 3.4,
                "cap": 4,
                "target_forwards": 25,
            }
        },
        "dspark",
    )
    mtp = _normalized_metrics(
        {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "timings": {
                "predicted_ms": 1000,
                "predicted_per_second": 50,
                "prompt_ms": 250,
                "prompt_per_second": 400,
                "prompt_n": 20,
                "cache_n": 80,
                "draft_kind": "mtp",
                "draft_rounds": 20,
                "draft_n": 60,
                "draft_n_accepted": 44,
            },
        },
        "mtp",
    )

    assert dspark["accept_length"] == 3.4
    assert dspark["decode_tokens_per_second"] == 51.2
    assert mtp["reused_prompt_tokens"] == 80
    assert mtp["accepted_tokens"] == 44


def test_counts_visible_thinking_tokens_with_the_target_tokenizer_shape():
    class Encoding:
        ids = [1, 2, 3, 4]

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "private reasoning"
            assert add_special_tokens is False
            return Encoding()

    assert _count_text_tokens(Tokenizer(), "private reasoning") == 4
    assert _count_text_tokens(Tokenizer(), "") == 0


def test_normalizes_qwen_reasoning_effort_and_off_mode():
    assert _reasoning_control({})[:2] == (True, "medium")
    assert _reasoning_control({"reasoning_effort": "medium"})[:2] == (
        True,
        "medium",
    )
    assert _reasoning_control(
        {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "LOW",
            }
        }
    )[:2] == (True, "low")
    assert _reasoning_control(
        {"enable_thinking": False, "reasoning_effort": "xhigh"}
    )[:2] == (False, None)

    with pytest.raises(ValueError, match="low, medium, or xhigh"):
        _reasoning_control({"reasoning_effort": "high"})


def test_image_detection_and_size_validation():
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nsmall image").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        }
    ]

    assert _has_images(messages)
    _validate_inline_images(messages, 1024)

    with pytest.raises(ValueError, match="no larger"):
        _validate_inline_images(messages, 2)

    messages[0]["content"][1]["image_url"]["url"] = "https://example.com/image.png"
    with pytest.raises(ValueError, match="base64 JPEG or PNG"):
        _validate_inline_images(messages, 1024)

    mismatched = base64.b64encode(b"not really a png").decode()
    messages[0]["content"][1]["image_url"]["url"] = (
        f"data:image/png;base64,{mismatched}"
    )
    with pytest.raises(ValueError, match="valid signature"):
        _validate_inline_images(messages, 1024)


def test_backend_commands_share_the_target_and_keep_dspark_text_only(tmp_path):
    manager = BackendManager(
        python=tmp_path / "python",
        target_model=tmp_path / "target",
        mtp_model=tmp_path / "mtp",
        dspark_model=tmp_path / "dspark",
        backend_host="127.0.0.1",
        backend_port=3940,
        max_context=262_144,
        default_backend="dspark",
    )

    dspark, _ = manager._command("dspark")
    mtp, mtp_environment = manager._command("mtp")

    assert str(tmp_path / "target") in dspark
    assert str(tmp_path / "target") in mtp
    assert "--no-lookup-drafts" in dspark
    assert dspark[dspark.index("--max-tokens-cap") + 1] == "262144"
    assert "--draft-block-size" in mtp
    assert mtp[mtp.index("--max-kv-size") + 1] == "262144"
    assert mtp_environment["APC_ENABLED"] == "1"
    assert mtp_environment["APC_NUM_BLOCKS"] == "16384"
