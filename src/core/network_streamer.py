import socket
import json
from typing import Union


class UDPStreamer:
    # Port changed to 8080 to match Unity
    def __init__(self, ip: str = "127.0.0.1", port: int = 8080):
        self.ip = ip
        self.port = port
        self.address = (self.ip, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # macOS caps outgoing datagrams at the send buffer size (default 9216 bytes), which
        # silently drops frame-carrying packets; raise it to the UDP maximum.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)
        # Non-blocking so Unity's control messages (see poll_messages) can be read without stalling
        self.sock.setblocking(False)

    def send(self, data: Union[bytes, str, dict]) -> bool:
        """
        Send a byte payload, string, or dictionary over UDP.
        Returns True on success, False if network transport fails.
        """
        try:
            if isinstance(data, dict):
                # No newline '\n' required for UDP datagrams
                data_bytes = json.dumps(data).encode("utf-8")
            elif isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data

            self.sock.sendto(data_bytes, self.address)

            # Debug: Verify outbound traffic
            # print(f"[UDP OUT] Sent {len(data_bytes)} bytes to {self.address}")
            return True

        except Exception as e:
            print(f"[UDPStreamer] Network error: {e}")
            return False

    def poll_messages(self) -> list:
        """Return any control messages Unity sent back to this socket (non-blocking)."""
        messages = []
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except (BlockingIOError, OSError):
                # Nothing waiting, or the socket isn't bound yet because nothing was sent
                break
            messages.append(data.decode("utf-8", errors="ignore").strip())
        return messages

    def close(self) -> None:
        """Safely release the UDP socket."""
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()