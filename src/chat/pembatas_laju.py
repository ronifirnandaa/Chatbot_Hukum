"""Pembatas laju sederhana (jendela geser), aman dipakai lintas thread (sesi Streamlit berjalan di thread berbeda)."""
from __future__ import annotations

import threading
import time
from collections import deque


class PembatasLaju:
    def __init__(self, maks: int, jendela_detik: float, jam=time.monotonic):
        self.maks, self.jendela, self._jam = maks, float(jendela_detik), jam
        self._waktu: deque = deque()
        self._kunci = threading.Lock()

    def coba(self) -> tuple:
        """Return (diizinkan, detik_tunggu). Bila diizinkan, satu slot langsung terpakai."""
        with self._kunci:
            sekarang = self._jam()
            while self._waktu and sekarang - self._waktu[0] >= self.jendela:
                self._waktu.popleft()
            if len(self._waktu) < self.maks:
                self._waktu.append(sekarang)
                return True, 0.0
            if not self._waktu:  # maks == 0: selalu ditolak
                return False, self.jendela
            return False, max(0.0, self.jendela - (sekarang - self._waktu[0]))
