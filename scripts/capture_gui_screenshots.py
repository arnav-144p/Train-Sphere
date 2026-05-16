from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import struct
import subprocess
import time
import urllib.request
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "screenshots"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
URL = "http://localhost:5173"
DEBUG_URL = "http://127.0.0.1:9222/json"


class CDP:
    def __init__(self, ws_url: str) -> None:
        self.sock = self._connect(ws_url)
        self.next_id = 1

    def _connect(self, ws_url: str) -> socket.socket:
        assert ws_url.startswith("ws://")
        host_port, path = ws_url[5:].split("/", 1)
        host, port = host_port.split(":", 1)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        sock = socket.create_connection((host, int(port)), timeout=10)
        req = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(req.encode("ascii"))
        response = sock.recv(4096)
        if b"101" not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(response.decode("utf-8", errors="replace"))
        return sock

    def _send_frame(self, text: str) -> None:
        payload = text.encode("utf-8")
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = secrets.token_bytes(4)
        header.extend(mask)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(header + masked)

    def _recv_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise RuntimeError("WebSocket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _recv_frame(self) -> str:
        first, second = self._recv_exact(2)
        
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        if second & 0x80:
            mask = self._recv_exact(4)
            payload = self._recv_exact(length)
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        else:
            payload = self._recv_exact(length)
        if opcode == 8:
            raise RuntimeError("WebSocket closed by Chrome")
        if opcode not in (1, 0):
            return self._recv_frame()
        return payload.decode("utf-8", errors="replace")

    def call(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        msg_id = self.next_id
        self.next_id += 1
        self._send_frame(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = json.loads(self._recv_frame())
            if data.get("id") == msg_id:
                if "error" in data:
                    raise RuntimeError(data["error"])
                return data.get("result", {})
        raise TimeoutError(method)

    def eval(self, expression: str, await_promise: bool = True, timeout: float = 60):
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                
                "awaitPromise": await_promise,
                "returnByValue": True,
                "userGesture": True,
            },
            timeout=timeout,
        )
        return result.get("result", {}).get("value")

    def eval_retry(self, expression: str, await_promise: bool = True, timeout: float = 60, tries: int = 5):
        last_error: Exception | None = None
        
        for _ in range(tries):
            try:
                return self.eval(expression, await_promise=await_promise, timeout=timeout)
            except RuntimeError as exc:
                last_error = exc
                if "Execution context was destroyed" not in str(exc):
                    raise
                time.sleep(1)
        raise last_error or RuntimeError("Runtime.evaluate failed")

    def screenshot(self, path: Path) -> None:
        result = self.call(
            "Page.captureScreenshot",
            {"format": "png", "fromSurface": True, "captureBeyondViewport": False},
            timeout=30,
        )
        path.write_bytes(base64.b64decode(result["data"]))


def wait_http(url: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


def get_ws_url() -> str:
    pages = json.loads(urllib.request.urlopen(DEBUG_URL, timeout=5).read().decode("utf-8"))
    for page in pages:
        if page.get("type") == "page":
            return page["webSocketDebuggerUrl"]
    raise RuntimeError("No Chrome page target found")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wait_http(URL)

    profile = ROOT / ".tmp" / "train-sphere-chrome-profile"
    profile.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--remote-debugging-port=9222",
            f"--user-data-dir={profile}",
            "--window-size=1848,1100",
            URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_http(DEBUG_URL)
        cdp = CDP(get_ws_url())
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1848, "height": 1100, "deviceScaleFactor": 1, "mobile": False},
        )
        cdp.call("Page.navigate", {"url": URL})
        time.sleep(2)
        cdp.eval_retry(
            """
            new Promise((resolve) => {
              const done = () => document.querySelector('button') ? resolve(true) : setTimeout(done, 100);
              done();
            })
            """,
            timeout=30,
        )

        cdp.eval(
            """
            (async () => {
              const train = [...document.querySelectorAll('button')].find((button) => button.textContent.includes('Train model'));
              train.click();
              while (!document.body.textContent.includes('Model performance')) {
                await new Promise((resolve) => setTimeout(resolve, 250));
              }
              await new Promise((resolve) => setTimeout(resolve, 900));
              return document.body.textContent.includes('93.33%');
            })()
            """,
            timeout=90,
        )
        cdp.screenshot(OUT_DIR / "train-sphere-single-results.png")

        cdp.eval(
            """
            (async () => {
              const compareTab = [...document.querySelectorAll('button')].find((button) => button.textContent.trim() === 'Compare');
              compareTab.click();
              await new Promise((resolve) => setTimeout(resolve, 500));
              const compare = [...document.querySelectorAll('button')].find((button) => button.textContent.includes('Run comparison'));
              compare.click();
              while (!document.body.textContent.includes('Best on this run')) {
                await new Promise((resolve) => setTimeout(resolve, 250));
              }
              await new Promise((resolve) => setTimeout(resolve, 900));
              return true;
            })()
            """,
            timeout=90,
        )
        cdp.screenshot(OUT_DIR / "train-sphere-model-comparison.png")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
