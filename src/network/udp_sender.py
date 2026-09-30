import socket

class UDPSender:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload_string: str):
        """Encodes and broadcasts the JSON string over the UDP socket."""
        try:
            self.sock.sendto(payload_string.encode('utf-8'), (self.ip, self.port))
        except Exception as e:
            print(f"UDP transmission error: {e}")

    def close(self):
        """Releases the socket when the tracker shuts down."""
        self.sock.close()