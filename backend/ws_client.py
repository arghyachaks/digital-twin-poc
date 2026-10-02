"""
Minimal, dependency-free WebSocket client (RFC 6455, text frames, no compression).
Used inside Unreal's embedded Python (which has no `websockets` package) and by simctl.py.
Non-blocking: call poll() regularly; it returns the list of decoded JSON/text messages received.
"""
import base64
import json
import os
import socket
import struct


class WSClient:
    def __init__(self, host="127.0.0.1", port=8765, path="/", timeout=3.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("handshake closed")
            resp += chunk
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError(head.decode(errors="replace"))
        self.sock.setblocking(False)
        self.closed = False

    def _frame(self, opcode, payload):
        mask = os.urandom(4)
        n = len(payload)
        hdr = bytes([0x80 | opcode])
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.setblocking(True)
        try:
            self.sock.sendall(hdr + mask + masked)
        finally:
            self.sock.setblocking(False)

    def send(self, obj):
        data = obj if isinstance(obj, str) else json.dumps(obj)
        self._frame(0x1, data.encode())

    def poll(self, as_json=True):
        """Read everything available and return complete messages."""
        while True:
            try:
                chunk = self.sock.recv(65536)
                if not chunk:
                    self.closed = True
                    break
                self.buf += chunk
            except (BlockingIOError, socket.timeout):
                break
            except OSError:
                self.closed = True
                break
        out, frag = [], b""
        while len(self.buf) >= 2:
            b0, b1 = self.buf[0], self.buf[1]
            n, i = b1 & 0x7F, 2
            if n == 126:
                if len(self.buf) < 4:
                    break
                n, i = struct.unpack(">H", self.buf[2:4])[0], 4
            elif n == 127:
                if len(self.buf) < 10:
                    break
                n, i = struct.unpack(">Q", self.buf[2:10])[0], 10
            if len(self.buf) < i + n:
                break
            payload, self.buf = self.buf[i:i + n], self.buf[i + n:]
            op = b0 & 0x0F
            if op in (0x1, 0x0):
                frag += payload
                if b0 & 0x80:
                    txt = frag.decode("utf-8", errors="replace")
                    frag = b""
                    try:
                        out.append(json.loads(txt) if as_json else txt)
                    except ValueError:
                        out.append(txt)
            elif op == 0x8:
                self.closed = True
            elif op == 0x9:
                self._frame(0xA, payload)
        return out

    def close(self):
        try:
            self._frame(0x8, b"")
            self.sock.close()
        except OSError:
            pass
        self.closed = True
