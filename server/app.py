from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from starlette.requests import Request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / ".runtime"
FRONTEND_ROOT = PROJECT_ROOT / "build" / "client"
BACKEND_PID_PATH = RUNTIME_ROOT / "backend.pid"
MODEL_ID = "qwen3.8-27b"
VALID_BACKENDS = frozenset({"dspark", "mtp"})
VALID_REASONING_EFFORTS = frozenset({"low", "medium", "xhigh"})
INLINE_IMAGE_PREFIXES = {
    "image/jpeg": "data:image/jpeg;base64,",
    "image/png": "data:image/png;base64,",
}
CONTEXT_BUDGET_PATTERN = re.compile(
    r"\((?P<prompt>\d+) prompt \+ (?P<generation>\d+) max generation\).*"
    r"MAX_KV_SIZE is (?P<limit>\d+)",
    re.DOTALL,
)


def _device_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "Apple silicon"


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _required_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"{label} is missing: {resolved}")
    if not (resolved / "config.json").is_file():
        raise SystemExit(f"{label} is incomplete: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.8-27B local playground")
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--mtp-model", type=Path, required=True)
    parser.add_argument("--dspark-model", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3939)
    parser.add_argument("--backend-port", type=int, default=3940)
    parser.add_argument("--default-backend", choices=sorted(VALID_BACKENDS), default="mtp")
    parser.add_argument("--max-context", type=int, default=262_144)
    parser.add_argument("--max-image-bytes", type=int, default=20 * 1024 * 1024)
    return parser


@dataclass(frozen=True)
class BackendSpec:
    id: str
    label: str
    description: str
    vision: bool
    artifact: str
    drafter: str


BACKEND_SPECS = {
    "dspark": BackendSpec(
        id="dspark",
        label="DSpark",
        description="Fast text and agent mode with an external speculative drafter.",
        vision=False,
        artifact="8-bit target · 4-bit DSpark drafter · Text",
        drafter="RadixArk Qwen3.8 DSpark",
    ),
    "mtp": BackendSpec(
        id="mtp",
        label="MTP",
        description="Qwen-native multi-token prediction with image support.",
        vision=True,
        artifact="8-bit target · 8-bit native MTP · Vision",
        drafter="Qwen3.8 native MTP",
    ),
}


class BackendBusyError(RuntimeError):
    pass


class BackendManager:
    def __init__(
        self,
        *,
        python: Path,
        target_model: Path,
        mtp_model: Path,
        dspark_model: Path,
        backend_host: str,
        backend_port: int,
        max_context: int,
        default_backend: str,
    ) -> None:
        self.python = python
        self.target_model = target_model
        self.mtp_model = mtp_model
        self.dspark_model = dspark_model
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.max_context = max_context
        self.default_backend = default_backend
        self.active_backend: str | None = None
        self.requested_backend: str = default_backend
        self.state = "starting"
        self.error: str | None = None
        self.load_time_ms: int | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.transition_task: asyncio.Task[None] | None = None
        self.transition_lock = asyncio.Lock()
        self.active_streams: dict[str, Any] = {}
        self.selected_backend_path = RUNTIME_ROOT / "selected-backend"
        self.logger = logging.getLogger("qwen_playground.backend")
        self.started_at: float | None = None

    @property
    def backend_url(self) -> str:
        return f"http://{self.backend_host}:{self.backend_port}"

    def persisted_backend(self) -> str:
        try:
            selected = self.selected_backend_path.read_text(encoding="utf-8").strip()
        except OSError:
            return self.default_backend
        return selected if selected in VALID_BACKENDS else self.default_backend

    async def start(self) -> None:
        await self.request_switch(self.persisted_backend())

    async def close(self) -> None:
        task = self.transition_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.cancel_all_streams()
        await self._stop_process()

    async def request_switch(self, backend: str) -> dict[str, Any]:
        if backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend: {backend}")
        async with self.transition_lock:
            if self.transition_task and not self.transition_task.done():
                if backend == self.requested_backend:
                    return self.snapshot()
                raise BackendBusyError(
                    f"{BACKEND_SPECS[self.requested_backend].label} is still loading."
                )
            if backend == self.active_backend and self.state == "ready":
                return self.snapshot()
            self.requested_backend = backend
            self.state = f"loading-{backend}"
            self.error = None
            self.load_time_ms = None
            self.transition_task = asyncio.create_task(self._transition(backend))
            return self.snapshot()

    async def _transition(self, backend: str) -> None:
        started = time.perf_counter()
        try:
            await self.cancel_all_streams()
            await self._stop_process()
            self.active_backend = None
            self.started_at = time.time()
            command, environment = self._command(backend)
            self.logger.info("Loading %s backend", BACKEND_SPECS[backend].label)
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(PROJECT_ROOT),
                env=environment,
            )
            BACKEND_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
            BACKEND_PID_PATH.write_text(f"{self.process.pid}\n", encoding="utf-8")
            await self._wait_until_ready(backend)
            self.active_backend = backend
            self.state = "ready"
            self.load_time_ms = round((time.perf_counter() - started) * 1000)
            self.selected_backend_path.parent.mkdir(parents=True, exist_ok=True)
            self.selected_backend_path.write_text(f"{backend}\n", encoding="utf-8")
            self.logger.info(
                "%s is ready after %.1f s",
                BACKEND_SPECS[backend].label,
                self.load_time_ms / 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - surface child startup failures
            self.logger.exception("Unable to start the %s backend", backend)
            self.state = "error"
            self.error = str(error)
            await self._stop_process()

    def _command(self, backend: str) -> tuple[list[str], dict[str, str]]:
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(RUNTIME_ROOT / "cache" / "huggingface"),
                "HF_HUB_CACHE": str(RUNTIME_ROOT / "cache" / "huggingface" / "hub"),
                "HF_XET_CACHE": str(RUNTIME_ROOT / "cache" / "huggingface" / "xet"),
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        common = [
            "--host",
            self.backend_host,
            "--port",
            str(self.backend_port),
        ]
        if backend == "dspark":
            return (
                [
                    str(self.python),
                    "-m",
                    "mlx_dspark",
                    "serve",
                    "--mode",
                    "dspark",
                    "--model",
                    str(self.target_model),
                    "--drafter",
                    str(self.dspark_model),
                    "--context-window",
                    str(self.max_context),
                    "--max-tokens-cap",
                    str(self.max_context),
                    "--default-max-tokens",
                    "4096",
                    "--default-temperature",
                    "1.0",
                    "--default-top-p",
                    "0.95",
                    "--default-top-k",
                    "20",
                    "--prefix-cache-slots",
                    "4",
                    "--prefix-cache-rungs",
                    "8192",
                    "--no-lookup-drafts",
                    *common,
                ],
                environment,
            )

        environment.update(
            {
                "APC_ENABLED": "1",
                "APC_NUM_BLOCKS": "16384",
                "APC_BLOCK_SIZE": "16",
                "MLX_VLM_MAX_NUM_SEQS": "1",
            }
        )
        return (
            [
                str(self.python),
                "-m",
                "mlx_vlm.server",
                "--model",
                str(self.target_model),
                "--draft-model",
                str(self.mtp_model),
                "--draft-kind",
                "mtp",
                "--draft-block-size",
                "3",
                "--max-tokens",
                "4096",
                "--max-kv-size",
                str(self.max_context),
                "--max-num-seqs",
                "1",
                "--vision-cache-size",
                "8",
                "--enable-thinking",
                *common,
            ],
            environment,
        )

    async def _wait_until_ready(self, backend: str) -> None:
        import httpx

        deadline = time.monotonic() + 20 * 60
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            while time.monotonic() < deadline:
                process = self.process
                if process is None:
                    raise RuntimeError("The backend process disappeared during startup.")
                if process.returncode is not None:
                    raise RuntimeError(
                        f"{BACKEND_SPECS[backend].label} exited with status {process.returncode}."
                    )
                try:
                    response = await client.get(f"{self.backend_url}/health")
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        raise RuntimeError(f"Timed out while loading {BACKEND_SPECS[backend].label}.")

    async def _stop_process(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            BACKEND_PID_PATH.unlink(missing_ok=True)
            return
        try:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                self.logger.warning("Backend did not stop cleanly; killing it now")
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        finally:
            try:
                recorded_pid = BACKEND_PID_PATH.read_text(encoding="utf-8").strip()
            except OSError:
                recorded_pid = ""
            if not recorded_pid or recorded_pid == str(process.pid):
                BACKEND_PID_PATH.unlink(missing_ok=True)

    def refresh_process_state(self) -> None:
        if self.state == "ready" and self.process and self.process.returncode is not None:
            self.state = "error"
            self.error = f"Backend exited with status {self.process.returncode}."
            self.active_backend = None

    def snapshot(self) -> dict[str, Any]:
        self.refresh_process_state()
        backend = self.active_backend or self.requested_backend
        spec = BACKEND_SPECS[backend]
        return {
            "state": self.state,
            "activeBackend": backend,
            "requestedBackend": self.requested_backend,
            "error": self.error,
            "loadTimeMs": self.load_time_ms,
            "backend": f"MLX · {spec.label}",
            "artifact": spec.artifact,
            "drafter": spec.drafter,
            "supports": {
                "vision": spec.vision and self.state == "ready",
                "warmResume": True,
                "prefixCache": True,
                "thinking": True,
                "tools": True,
                "backendSwitch": True,
            },
            "backends": [
                {
                    "id": item.id,
                    "label": item.label,
                    "description": item.description,
                    "vision": item.vision,
                    "artifact": item.artifact,
                }
                for item in BACKEND_SPECS.values()
            ],
        }

    async def register_stream(self, session_id: str, response: Any) -> None:
        previous = self.active_streams.get(session_id)
        if previous is not None:
            await previous.aclose()
        self.active_streams[session_id] = response

    def unregister_stream(self, session_id: str, response: Any) -> None:
        if self.active_streams.get(session_id) is response:
            self.active_streams.pop(session_id, None)

    async def cancel_stream(self, session_id: str | None) -> bool:
        if session_id is None:
            streams = list(self.active_streams.values())
        else:
            response = self.active_streams.get(session_id)
            streams = [response] if response is not None else []
        for response in streams:
            try:
                await response.aclose()
            except Exception:  # noqa: BLE001 - cancellation is best effort
                pass
        return bool(streams)

    async def cancel_all_streams(self) -> None:
        await self.cancel_stream(None)
        self.active_streams.clear()


class QwenThinkingFilter:
    """Split Qwen's pre-opened <think> stream without leaking its close tag."""

    close_marker = "</think>"
    open_marker = "<think>"

    def __init__(self, thinking: bool) -> None:
        self.phase = "thinking" if thinking else "content"
        self.buffer = ""

    def feed(self, text: str, *, final: bool = False) -> list[tuple[str, str]]:
        if not text and not final:
            return []
        if self.phase == "content":
            return [("content", text)] if text else []

        self.buffer += text
        if self.buffer.startswith(self.open_marker):
            self.buffer = self.buffer[len(self.open_marker) :].lstrip("\n")

        marker_index = self.buffer.find(self.close_marker)
        if marker_index >= 0:
            reasoning = self.buffer[:marker_index]
            content = self.buffer[marker_index + len(self.close_marker) :].lstrip()
            self.buffer = ""
            self.phase = "content"
            output = []
            if reasoning:
                output.append(("reasoning_content", reasoning))
            if content:
                output.append(("content", content))
            return output

        hold = 0 if final else len(self.close_marker) - 1
        safe = max(0, len(self.buffer) - hold)
        reasoning, self.buffer = self.buffer[:safe], self.buffer[safe:]
        return [("reasoning_content", reasoning)] if reasoning else []


async def _iter_sse_data(response: Any) -> AsyncIterator[str]:
    buffer = ""
    data_lines: list[str] = []
    async for text in response.aiter_text():
        buffer += text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
    if buffer.startswith("data:"):
        data_lines.append(buffer[5:].removeprefix(" ").removesuffix("\r"))
    if data_lines:
        yield "\n".join(data_lines)


def _sse(payload: str | dict[str, Any]) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


def _event_with_delta(event: dict[str, Any], field: str, value: str) -> dict[str, Any]:
    copied = dict(event)
    choices = [dict(choice) for choice in event.get("choices", [])]
    copied["choices"] = choices
    if choices:
        delta = dict(choices[0].get("delta") or {})
        delta.pop("content", None)
        delta.pop("reasoning", None)
        delta.pop("reasoning_content", None)
        delta[field] = value
        choices[0]["delta"] = delta
    return copied


def _normalized_metrics(event: dict[str, Any], backend: str) -> dict[str, Any] | None:
    if backend == "dspark" and isinstance(event.get("x_mlx_dspark"), dict):
        stats = event["x_mlx_dspark"]
        return {
            "backend": "dspark",
            "decode_tokens_per_second": stats.get("tokens_per_sec"),
            "accept_length": stats.get("accept_len"),
            "draft_cap": stats.get("cap"),
            "target_forwards": stats.get("target_forwards"),
            "lookup_rounds": stats.get("lookup_rounds"),
        }
    timings = event.get("timings")
    if backend == "mtp" and event.get("usage") and isinstance(timings, dict):
        return {
            "backend": "mtp",
            "decode_ms": timings.get("predicted_ms"),
            "decode_tokens_per_second": timings.get("predicted_per_second"),
            "prefill_ms": timings.get("prompt_ms"),
            "prefill_tokens_per_second": timings.get("prompt_per_second"),
            "prefilled_prompt_tokens": timings.get("prompt_n"),
            "reused_prompt_tokens": timings.get("cache_n"),
            "peak_memory_gb": timings.get("peak_memory"),
            "draft_kind": timings.get("draft_kind"),
            "draft_rounds": timings.get("draft_rounds"),
            "drafted_tokens": timings.get("draft_n"),
            "accepted_tokens": timings.get("draft_n_accepted"),
        }
    return None


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def _count_chat_prompt_tokens(
    tokenizer: Any,
    messages: list[Any],
    *,
    thinking: bool,
    reasoning_effort: str | None,
) -> int:
    template_kwargs: dict[str, Any] = {"enable_thinking": thinking}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **template_kwargs,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        return sum(len(row) for row in input_ids)
    return len(input_ids)


def _bounded_generation_tokens(
    requested_tokens: int,
    prompt_tokens: int,
    context_limit: int,
) -> int:
    available_tokens = context_limit - prompt_tokens
    if available_tokens < 1:
        raise ValueError(
            f"The formatted prompt uses {prompt_tokens} tokens, leaving no room "
            f"in the {context_limit}-token context."
        )
    return min(requested_tokens, available_tokens)


def _context_budget_from_error(detail: Any) -> tuple[int, int] | None:
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("detail") or detail.get("error")
    if not isinstance(detail, str):
        return None
    match = CONTEXT_BUDGET_PATTERN.search(detail)
    if match is None:
        return None
    prompt_tokens = int(match.group("prompt"))
    context_limit = int(match.group("limit"))
    return prompt_tokens, max(0, context_limit - prompt_tokens)


def _valid_session_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and all(0x21 <= ord(character) <= 0x7E for character in value)
    )


def _backend_error_detail(body: bytes) -> Any:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace") or "Backend request failed."
    if not isinstance(payload, dict):
        return payload
    return payload.get("detail") or payload.get("error") or payload


def _reasoning_control(payload: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
    raw_template_kwargs = payload.get("chat_template_kwargs")
    if raw_template_kwargs is not None and not isinstance(raw_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object.")
    template_kwargs = dict(raw_template_kwargs or {})

    enabled = payload.get(
        "enable_thinking",
        template_kwargs.get("enable_thinking", True),
    )
    if not isinstance(enabled, bool):
        raise ValueError("enable_thinking must be a boolean.")
    if not enabled:
        return False, None, template_kwargs

    effort = payload.get(
        "reasoning_effort",
        template_kwargs.get("reasoning_effort", "medium"),
    )
    if not isinstance(effort, str) or effort.lower() not in VALID_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or xhigh.")
    return True, effort.lower(), template_kwargs


def _has_images(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict)
            and part.get("type") in {"image_url", "input_image"}
            for part in content
        ):
            return True
    return False


def _validate_inline_images(messages: Any, max_image_bytes: int) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "input_image":
                raise ValueError("Attached images must use OpenAI image_url parts.")
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str):
                raise ValueError("image_url.url must be a data URL string.")
            image_type = next(
                (
                    mime_type
                    for mime_type, prefix in INLINE_IMAGE_PREFIXES.items()
                    if url.startswith(prefix)
                ),
                None,
            )
            if image_type is None:
                raise ValueError(
                    "Attached images must be base64 JPEG or PNG data URLs."
                )
            encoded = url[len(INLINE_IMAGE_PREFIXES[image_type]) :]
            estimated = len(encoded) * 3 // 4
            if estimated > max_image_bytes + 2:
                limit_mb = max_image_bytes // (1024 * 1024)
                raise ValueError(
                    f"Attached images must be no larger than {limit_mb} MB."
                )
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("An attached image is not valid base64 data.") from error
            if len(decoded) > max_image_bytes:
                limit_mb = max_image_bytes // (1024 * 1024)
                raise ValueError(
                    f"Attached images must be no larger than {limit_mb} MB."
                )
            png_signature = b"\x89PNG\r\n\x1a\n"
            if image_type == "image/png" and not decoded.startswith(png_signature):
                raise ValueError(
                    "The attached PNG data does not have a valid signature."
                )
            if image_type == "image/jpeg" and not decoded.startswith(b"\xff\xd8\xff"):
                raise ValueError(
                    "The attached JPEG data does not have a valid signature."
                )


def create_app(cli: argparse.Namespace):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    import httpx

    target_model = _required_directory(cli.target_model, "Qwen target model")
    mtp_model = _required_directory(cli.mtp_model, "Qwen MTP model")
    dspark_model = _required_directory(cli.dspark_model, "Qwen DSpark model")
    python = cli.python.expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if not python.is_file():
        raise SystemExit(f"Project Python is missing: {python}")
    if not (FRONTEND_ROOT / "index.html").is_file():
        raise SystemExit("The browser app is not built. Run scripts/bootstrap.sh first.")
    tokenizer_path = target_model / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise SystemExit(f"Qwen tokenizer is missing: {tokenizer_path}")
    completion_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        target_model,
        local_files_only=True,
        trust_remote_code=False,
    )

    manager = BackendManager(
        python=python,
        target_model=target_model,
        mtp_model=mtp_model,
        dspark_model=dspark_model,
        backend_host="127.0.0.1",
        backend_port=cli.backend_port,
        max_context=cli.max_context,
        default_backend=cli.default_backend,
    )
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=None, write=60, pool=5),
        trust_env=False,
    )
    model_bytes = _directory_size(target_model)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.close()
            await client.aclose()

    app = FastAPI(title="Qwen3.8-27B Playground", lifespan=lifespan)

    @app.get("/health")
    async def health():
        manager.refresh_process_state()
        return {"status": "ok", "runtime": manager.state}

    @app.get("/api/runtime")
    async def runtime_info():
        snapshot = manager.snapshot()
        return {
            **snapshot,
            "model": "Qwen3.8 27B",
            "modelId": MODEL_ID,
            "device": _device_name(),
            "contextLength": cli.max_context,
            "maxOutputTokens": cli.max_context,
            "maxImageBytes": cli.max_image_bytes,
            "modelBytes": model_bytes,
            "precision": "8-bit",
            "defaults": {
                "temperature": 1.0,
                "topP": 0.95,
                "topK": 20,
                "presencePenalty": 0.0,
                "maxTokens": cli.max_context,
                "enableThinking": True,
                "reasoningEffort": "medium",
            },
        }

    @app.post("/api/backend", status_code=202)
    async def switch_backend(payload: dict[str, Any] = Body(...)):
        if set(payload) != {"backend"}:
            raise HTTPException(status_code=422, detail="Only backend is accepted.")
        backend = payload.get("backend")
        if backend not in VALID_BACKENDS:
            raise HTTPException(status_code=422, detail="Backend must be dspark or mtp.")
        try:
            return await manager.request_switch(backend)
        except BackendBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/cancel")
    async def cancel_generation(payload: dict[str, Any] | None = Body(default=None)):
        body = payload or {}
        if set(body) - {"session_id"}:
            raise HTTPException(status_code=422, detail="Only session_id is accepted.")
        session_id = body.get("session_id")
        if session_id is not None and not _valid_session_id(session_id):
            raise HTTPException(
                status_code=422,
                detail="session_id must be 1-128 printable ASCII characters.",
            )
        cancelled = await manager.cancel_stream(session_id)
        return {"cancelled": cancelled, "session_id": session_id}

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        if not _valid_session_id(session_id):
            raise HTTPException(status_code=422, detail="Invalid session_id.")
        await manager.cancel_stream(session_id)
        return {"deleted": True, "session_id": session_id}

    @app.post("/v1/sessions/{session_id}/reset")
    async def reset_session(session_id: str):
        if not _valid_session_id(session_id):
            raise HTTPException(status_code=422, detail="Invalid session_id.")
        await manager.cancel_stream(session_id)
        return {"reset": True, "session_id": session_id}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        snapshot = manager.snapshot()
        if snapshot["state"] != "ready" or manager.active_backend is None:
            raise HTTPException(
                status_code=503,
                detail=snapshot.get("error")
                or f"{BACKEND_SPECS[manager.requested_backend].label} is still loading.",
            )

        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

        backend = manager.active_backend
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=422, detail="messages must be a non-empty list.")
        if backend == "dspark" and _has_images(messages):
            raise HTTPException(
                status_code=409,
                detail="DSpark mode is text-only. Switch to MTP to use images.",
            )
        try:
            _validate_inline_images(messages, cli.max_image_bytes)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        session_id = payload.pop("session_id", None) or f"request-{time.time_ns()}"
        if not _valid_session_id(session_id):
            raise HTTPException(status_code=422, detail="Invalid session_id.")
        payload["model"] = str(target_model)
        payload["stream"] = True
        payload.pop("reasoning_strength", None)
        try:
            thinking, reasoning_effort, template_kwargs = _reasoning_control(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        payload["enable_thinking"] = thinking
        if backend == "dspark":
            payload.pop("reasoning_effort", None)
            template_kwargs["enable_thinking"] = thinking
            if reasoning_effort is None:
                template_kwargs.pop("reasoning_effort", None)
            else:
                template_kwargs["reasoning_effort"] = reasoning_effort
            payload["chat_template_kwargs"] = template_kwargs
        else:
            payload.pop("chat_template_kwargs", None)
            if reasoning_effort is None:
                payload.pop("reasoning_effort", None)
            else:
                payload["reasoning_effort"] = reasoning_effort

        requested_max_tokens = payload.pop("max_completion_tokens", None)
        legacy_max_tokens = payload.pop("max_tokens", None)
        if requested_max_tokens is None:
            requested_max_tokens = legacy_max_tokens
        if requested_max_tokens is not None:
            if (
                not isinstance(requested_max_tokens, int)
                or isinstance(requested_max_tokens, bool)
                or not 1 <= requested_max_tokens <= cli.max_context
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "max_tokens must be an integer between 1 and "
                        f"{cli.max_context}."
                    ),
                )
            # Normalize both OpenAI spellings before prompt counting. If the
            # local tokenizer cannot estimate this request, the backend still
            # receives the requested cap and its exact context check remains
            # available as the fallback.
            payload["max_tokens"] = requested_max_tokens
            try:
                prompt_tokens = _count_chat_prompt_tokens(
                    prompt_tokenizer,
                    messages,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                )
                payload["max_tokens"] = _bounded_generation_tokens(
                    requested_max_tokens,
                    prompt_tokens,
                    cli.max_context,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            except Exception:  # noqa: BLE001 - backend retry remains the exact fallback
                logging.getLogger("qwen_playground.context").warning(
                    "Unable to estimate the formatted prompt size",
                    exc_info=True,
                )

        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if backend == "mtp":
            headers["X-APC-Tenant"] = session_id

        stream_context = None
        upstream = None
        for attempt in range(2):
            stream_context = client.stream(
                "POST",
                f"{manager.backend_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            try:
                upstream = await stream_context.__aenter__()
            except httpx.HTTPError as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"Backend connection failed: {error}",
                ) from error
            if upstream.is_success:
                break

            status_code = upstream.status_code
            body = await upstream.aread()
            await stream_context.__aexit__(None, None, None)
            detail = _backend_error_detail(body)
            exact_budget = _context_budget_from_error(detail)
            current_max_tokens = payload.get("max_tokens")
            if (
                attempt == 0
                and exact_budget is not None
                and exact_budget[1] >= 1
                and isinstance(current_max_tokens, int)
                and exact_budget[1] < current_max_tokens
            ):
                payload["max_tokens"] = exact_budget[1]
                continue
            raise HTTPException(status_code=status_code, detail=detail)

        if stream_context is None or upstream is None:
            raise HTTPException(status_code=503, detail="Backend connection failed.")

        await manager.register_stream(session_id, upstream)

        async def proxy_stream() -> AsyncIterator[str]:
            thinking_filter = QwenThinkingFilter(thinking and backend == "dspark")
            last_event: dict[str, Any] | None = None
            pending_metrics: dict[str, Any] | None = None
            reasoning_parts: list[str] = []
            try:
                async for data in _iter_sse_data(upstream):
                    if data == "[DONE]":
                        for field, value in thinking_filter.feed("", final=True):
                            if last_event is not None:
                                if field == "reasoning_content":
                                    reasoning_parts.append(value)
                                yield _sse(_event_with_delta(last_event, field, value))
                        if pending_metrics is not None:
                            if thinking:
                                pending_metrics["thinking_tokens"] = _count_text_tokens(
                                    completion_tokenizer,
                                    "".join(reasoning_parts),
                                )
                            yield _sse({"qwen_metrics": pending_metrics})
                        yield _sse("[DONE]")
                        return
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        yield _sse({"error": {"message": "Backend sent malformed SSE data."}})
                        continue
                    if not isinstance(event, dict):
                        continue
                    last_event = event
                    choice = (event.get("choices") or [None])[0]
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if backend == "dspark" and isinstance(content, str) and content:
                        for field, value in thinking_filter.feed(content):
                            if field == "reasoning_content":
                                reasoning_parts.append(value)
                            yield _sse(_event_with_delta(event, field, value))
                    else:
                        if isinstance(delta, dict):
                            reasoning_delta = delta.get("reasoning_content")
                            if not isinstance(reasoning_delta, str):
                                reasoning_delta = delta.get("reasoning")
                            if isinstance(reasoning_delta, str) and reasoning_delta:
                                reasoning_parts.append(reasoning_delta)
                        yield _sse(event)
                    metrics = _normalized_metrics(event, backend)
                    if metrics:
                        pending_metrics = metrics
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError:
                if not upstream.is_closed:
                    yield _sse({"error": {"message": "Generation was interrupted."}})
            finally:
                manager.unregister_stream(session_id, upstream)
                await stream_context.__aexit__(None, None, None)

        return StreamingResponse(
            proxy_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    app.mount("/", StaticFiles(directory=FRONTEND_ROOT, html=True), name="frontend")
    return app


def main() -> None:
    cli = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvicorn

    app = create_app(cli)
    uvicorn.run(app, host=cli.host, port=cli.port, log_level="info")


if __name__ == "__main__":
    main()
