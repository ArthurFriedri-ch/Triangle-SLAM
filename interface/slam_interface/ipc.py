# interface/slam_interface/ipc.py
import socket
import struct
from .record import Record

_HDR = struct.Struct(">I")   # 4-byte big-endian length prefix per message


def _recv_exact(sock, n: int) -> bytes:
    chunks = []
    while n > 0:
        b = sock.recv(n)
        if not b:
            raise ConnectionError("peer closed mid-message")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def _send_msg(sock, payload: bytes):
    sock.sendall(_HDR.pack(len(payload)) + payload)


def _recv_msg(sock) -> bytes:
    (length,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    return _recv_exact(sock, length)


class RecordSender:
    """Tracker side (live): connect to the mapper and stream records."""
    def __init__(self, host="127.0.0.1", port=55555):
        self.sock = socket.create_connection((host, port))

    def send(self, record: Record, validate: bool = False):
        if validate:
            record.validate()
        _send_msg(self.sock, record.to_bytes())

    def close(self):
        self.sock.close()


class RecordReceiver:
    """Mapper side (live): listen, accept the tracker, yield records."""
    def __init__(self, host="0.0.0.0", port=55555):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.conn = None

    def recv(self):
        if self.conn is None:
            self.conn, _ = self._srv.accept()
        try:
            return Record.from_bytes(_recv_msg(self.conn))
        except ConnectionError:
            return None            # stream ended

    def __iter__(self):
        while True:
            r = self.recv()
            if r is None:
                break
            yield r

    def close(self):
        if self.conn:
            self.conn.close()
        self._srv.close()