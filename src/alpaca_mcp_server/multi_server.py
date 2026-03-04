import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route


@dataclass
class AccountRoute:
    index: int
    external_path: str
    internal_path: str
    port: int
    env: Dict[str, str]


def _normalize_path(path: str, fallback: str) -> str:
    raw = (path or fallback).strip()
    if not raw:
        raw = fallback
    if not raw.startswith("/"):
        raw = f"/{raw}"
    if len(raw) > 1 and raw.endswith("/"):
        raw = raw[:-1]
    return raw


def _suffix_value(name: str, index: int) -> Optional[str]:
    value = os.getenv(f"{name}_{index}")
    if value in [None, "", "None"]:
        return None
    return value


def _resolve_setting(base_name: str, index: int) -> Optional[str]:
    return _suffix_value(base_name, index) or os.getenv(base_name)


def _build_account_routes(base_port: int = 8101) -> List[AccountRoute]:
    routes: List[AccountRoute] = []
    index = 1
    while True:
        api_key = _suffix_value("ALPACA_API_KEY", index)
        secret_key = _suffix_value("ALPACA_SECRET_KEY", index)
        if not api_key and not secret_key:
            break
        if not api_key or not secret_key:
            raise ValueError(
                f"Account {index} is incomplete. Set both ALPACA_API_KEY_{index} and ALPACA_SECRET_KEY_{index}."
            )

        external_default = "/mcp" if index == 1 else f"/mcp{index}"
        external_path = _normalize_path(
            _resolve_setting("ALPACA_MCP_PATH", index) or external_default,
            external_default,
        )

        env = {
            "ALPACA_API_KEY": api_key,
            "ALPACA_SECRET_KEY": secret_key,
            "ALPACA_PAPER_TRADE": _resolve_setting("ALPACA_PAPER_TRADE", index) or "True",
            "TRADE_API_URL": _resolve_setting("TRADE_API_URL", index) or "",
            "TRADE_API_WSS": _resolve_setting("TRADE_API_WSS", index) or "",
            "DATA_API_URL": _resolve_setting("DATA_API_URL", index) or "",
            "STREAM_DATA_WSS": _resolve_setting("STREAM_DATA_WSS", index) or "",
            "DEBUG": os.getenv("DEBUG", "False"),
        }
        routes.append(
            AccountRoute(
                index=index,
                external_path=external_path,
                internal_path="/mcp",
                port=base_port + (index - 1),
                env=env,
            )
        )
        index += 1

    if len(routes) < 2:
        raise ValueError(
            "Multi-account mode requires at least 2 accounts. "
            "Set ALPACA_API_KEY_1/ALPACA_SECRET_KEY_1 and ALPACA_API_KEY_2/ALPACA_SECRET_KEY_2."
        )

    # Validate unique external paths
    seen: Dict[str, int] = {}
    for route in routes:
        if route.external_path in seen:
            raise ValueError(
                f"Duplicate external path '{route.external_path}' for accounts {seen[route.external_path]} and {route.index}."
            )
        seen[route.external_path] = route.index

    return routes


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _start_workers(routes: List[AccountRoute]) -> List[subprocess.Popen[str]]:
    processes: List[subprocess.Popen[str]] = []
    for route in routes:
        child_env = os.environ.copy()
        child_env.update(route.env)
        # Worker processes are internal-only and should not inherit external host policy.
        child_env.pop("ALLOWED_HOSTS", None)
        child_env["HOST"] = "127.0.0.1"
        child_env["PORT"] = str(route.port)
        cmd = [
            sys.executable,
            "-m",
            "alpaca_mcp_server.server",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(route.port),
        ]
        processes.append(
            subprocess.Popen(
                cmd,
                env=child_env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
        )

    # Wait for workers to bind their ports
    for route, process in zip(routes, processes):
        ready = False
        for _ in range(60):
            if process.poll() is not None:
                break
            if _is_port_open("127.0.0.1", route.port):
                ready = True
                break
            time.sleep(0.1)
        if not ready:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Account {route.index} worker exited early with code {exit_code} (port {route.port})."
                )
            raise RuntimeError(f"Account {route.index} worker failed to start on port {route.port}.")

    return processes


def _stop_workers(processes: List[subprocess.Popen[str]]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _resolve_target(path: str, routes: List[AccountRoute]) -> Optional[Tuple[AccountRoute, str]]:
    # Match longest path first to avoid prefix ambiguities.
    for route in sorted(routes, key=lambda r: len(r.external_path), reverse=True):
        if path == route.external_path or path.startswith(f"{route.external_path}/"):
            suffix = path[len(route.external_path):]
            internal_path = route.internal_path if not suffix else f"{route.internal_path}{suffix}"
            return route, internal_path
    return None


def _filtered_response_headers(headers: httpx.Headers) -> Dict[str, str]:
    blocked = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}
    return {k: v for k, v in headers.items() if k.lower() not in blocked}


def run_multi_account_server(host: str, port: int, allowed_hosts: str = "") -> None:
    routes = _build_account_routes()
    workers = _start_workers(routes)

    print("Multi-account MCP mode enabled:", file=sys.stderr)
    for route in routes:
        print(
            f"  account_{route.index}: {route.external_path} -> http://127.0.0.1:{route.port}{route.internal_path}",
            file=sys.stderr,
        )

    async def startup() -> None:
        app.state.client = httpx.AsyncClient(timeout=None)

    async def shutdown() -> None:
        client: httpx.AsyncClient = app.state.client
        await client.aclose()
        _stop_workers(workers)

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "accounts": len(routes)})

    async def root(_: Request) -> Response:
        mapped = [route.external_path for route in routes]
        return PlainTextResponse(
            "Alpaca MCP multi-account server is running.\n"
            f"Available endpoints: {', '.join(mapped)}\n"
        )

    async def proxy(request: Request) -> Response:
        path = request.url.path
        target = _resolve_target(path, routes)
        if target is None:
            mapped = ", ".join(route.external_path for route in routes)
            return JSONResponse(
                {"error": f"Unknown path '{path}'. Use one of: {mapped}"},
                status_code=404,
            )

        route, internal_path = target
        upstream_url = f"http://127.0.0.1:{route.port}{internal_path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        client: httpx.AsyncClient = app.state.client
        upstream_request = client.build_request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            content=await request.body(),
        )

        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"Upstream MCP worker unavailable: {e}"}, status_code=502)

        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=_filtered_response_headers(upstream_response.headers),
            background=BackgroundTask(upstream_response.aclose),
        )

    middleware: List[Middleware] = []
    if allowed_hosts:
        host_rules = [h.strip() for h in allowed_hosts.split(",") if h.strip()]
        host_rules.extend(["127.0.0.1", "localhost", "[::1]"])
        middleware.append(Middleware(TrustedHostMiddleware, allowed_hosts=host_rules))

    app = Starlette(
        routes=[
            Route("/", endpoint=root, methods=["GET"]),
            Route("/healthz", endpoint=health, methods=["GET"]),
            Route("/{path:path}", endpoint=proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]),
        ],
        middleware=middleware,
        on_startup=[startup],
        on_shutdown=[shutdown],
    )

    # Best-effort clean shutdown on SIGTERM/SIGINT in container runtimes.
    def _signal_handler(_: int, __: Any) -> None:
        _stop_workers(workers)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        _stop_workers(workers)
