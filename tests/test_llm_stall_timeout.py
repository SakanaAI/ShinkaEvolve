import json
import os
import subprocess
import sys
import textwrap


def test_stalled_local_openai_query_respects_configured_retry_bounds():
    code = textwrap.dedent(
        """
        from __future__ import annotations

        import asyncio
        import http.server
        import json
        import socketserver
        import threading
        import time


        class StallingHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _stall(self) -> None:
                self.server.request_count += 1
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length:
                    self.rfile.read(length)
                while not getattr(self.server, "_shutting_down", False):
                    time.sleep(0.05)

            do_GET = _stall
            do_POST = _stall

            def log_message(self, *_args) -> None:
                pass


        class StallingServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            request_count = 0


        server = StallingServer(("127.0.0.1", 0), StallingHandler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        from shinka.llm.client import get_async_client_llm
        from shinka.llm.constants import (
            BACKOFF_MAX_TIME,
            BACKOFF_MAX_TRIES,
            MAX_RETRIES,
            OPENAI_MAX_RETRIES,
            TIMEOUT,
        )
        from shinka.llm.providers.local_openai import query_local_openai_async


        async def main() -> None:
            client, model, provider = get_async_client_llm(
                f"local/stalled@http://127.0.0.1:{port}/v1?api_key_env=CUSTOM_LOCAL_KEY"
            )
            started = time.monotonic()
            try:
                await query_local_openai_async(
                    client,
                    model,
                    "hello",
                    "system",
                    [],
                    None,
                    max_tokens=1,
                )
                exception = None
            except Exception as exc:
                exception = type(exc).__name__
            elapsed = time.monotonic() - started
            server._shutting_down = True
            server.shutdown()
            print(
                json.dumps(
                    {
                        "provider": provider,
                        "elapsed": elapsed,
                        "exception": exception,
                        "request_count": server.request_count,
                        "timeout": TIMEOUT,
                        "max_retries": MAX_RETRIES,
                        "openai_max_retries": OPENAI_MAX_RETRIES,
                        "backoff_max_tries": BACKOFF_MAX_TRIES,
                        "backoff_max_time": BACKOFF_MAX_TIME,
                    },
                    sort_keys=True,
                )
            )


        asyncio.run(main())
        """
    )
    env = {
        **os.environ,
        "CUSTOM_LOCAL_KEY": "test-local-key",
        "SHINKA_LLM_TIMEOUT": "1",
        "SHINKA_LLM_MAX_RETRIES": "1",
        "SHINKA_OPENAI_MAX_RETRIES": "0",
        "SHINKA_LLM_BACKOFF_MAX_TRIES": "1",
        "SHINKA_LLM_BACKOFF_MAX_TIME": "1",
        "PYTHONUNBUFFERED": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    payload = json.loads(result.stdout.splitlines()[-1])

    assert payload["provider"] == "local_openai"
    assert payload["exception"] == "APITimeoutError"
    assert payload["request_count"] == 1
    assert payload["timeout"] == 1
    assert payload["max_retries"] == 1
    assert payload["openai_max_retries"] == 0
    assert payload["backoff_max_tries"] == 1
    assert payload["backoff_max_time"] == 1
    assert payload["elapsed"] < 3
