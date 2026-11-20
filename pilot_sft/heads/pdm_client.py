"""Client of ``pdm_server.py`` for the RL trainer (pilot env; plain Python, no NAVSIM import).

    client = PDMClient()                       # the server listens on 127.0.0.1:47411
    results = client.score([(token, [path_1, path_2, ...]), ...])   # paths: 8 x (x, y) in metres
    rewards = [[r["score"] for r in per_scene] for per_scene in results]
"""

from __future__ import annotations

from multiprocessing.connection import Client
from typing import List, Sequence, Tuple

AUTHKEY = b"heads-pdm"  # pdm_server.AUTHKEY


class PDMClient:
    def __init__(self, port: int = 47411, host: str = "127.0.0.1"):
        self.conn = Client((host, port), authkey=AUTHKEY)
        self.last_seconds = 0.0

    def score(self, items: Sequence[Tuple[str, Sequence]]) -> List[List[dict]]:
        """One result dict per candidate path, in order: ``score`` and the sub-scores, or ``score`` 0 and
        ``error`` when NAVSIM could not score it."""
        self.conn.send({"op": "score", "items": [(t, [[list(map(float, p)) for p in path] for path in paths])
                                                  for t, paths in items]})
        reply = self.conn.recv()
        self.last_seconds = reply["seconds"]
        return reply["results"]

    def close(self) -> None:
        try:
            self.conn.send({"op": "close"})
        finally:
            self.conn.close()
