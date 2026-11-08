"""Pemeriksa kutipan hukum: apakah pasal/peraturan/putusan yang disebut jawaban ADA di konteks yang diberikan ke model?

Ini pemeriksaan KASAR berbasis pencocokan teks, bukan penilaian kebenaran hukum:
* Hanya memeriksa nomor pasal, nomor-tahun peraturan, dan nomor putusan MK.
* Nomor pasal dicocokkan tanpa memandang undang-undangnya (Pasal 3 UU A dianggap cocok dengan Pasal 3 UU B).
* Kutipan yang ditemukan di konteks belum tentu dipakai dengan benar; kutipan yang TIDAK ditemukan adalah
  kandidat karangan model (atau pengetahuan bawaan model) dan layak diperiksa manusia.
Bukan bagian dari metrik evaluasi penelitian (RAGAS di notebook).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_RE_PASAL = re.compile(r"\bPasal\s+(\d+[A-Za-z]?)\b", re.I)
_RE_PERATURAN = re.compile(
    r"\b(?:UU|PP|Perpres|Perppu|Perma|Undang-?undang|Peraturan\s+Pemerintah)\s*(?:No\.?|Nomor)?\s*(\d+)"
    r"\s*(?:/|\s+Tahun\s+)\s*(\d{4})\b",
    re.I,
)
_RE_PUTUSAN = re.compile(r"\b(\d+/PUU-[IVXLC]+/\d{4})\b", re.I)

JENIS = ("pasal", "peraturan", "putusan")


def ekstrak(teks) -> dict:
    """Himpunan kutipan ternormalisasi: {'pasal': {'3','2'}, 'peraturan': {'31/1999'}, 'putusan': {'25/PUU-XIV/2016'}}."""
    t = str(teks or "")
    return {
        "pasal": {m.upper() for m in _RE_PASAL.findall(t)},
        "peraturan": {"%d/%s" % (int(n), y) for n, y in _RE_PERATURAN.findall(t)},
        "putusan": {m.upper() for m in _RE_PUTUSAN.findall(t)},
    }


@dataclass
class HasilPeriksa:
    dikutip: dict
    tidak_ditemukan: dict

    @property
    def n_dikutip(self) -> int:
        return sum(len(v) for v in self.dikutip.values())

    @property
    def n_tidak_ditemukan(self) -> int:
        return sum(len(v) for v in self.tidak_ditemukan.values())

    @property
    def semua_ditemukan(self) -> bool:
        return self.n_tidak_ditemukan == 0

    def ringkas(self) -> str:
        if not self.n_tidak_ditemukan:
            return "semua %d kutipan ditemukan di konteks" % self.n_dikutip
        bagian = []
        for j in JENIS:
            if self.tidak_ditemukan[j]:
                bagian.append("%s %s" % (j, ", ".join(sorted(self.tidak_ditemukan[j]))))
        return "%d dari %d kutipan TIDAK ada di konteks: %s" % (self.n_tidak_ditemukan, self.n_dikutip, "; ".join(bagian))


def periksa_kutipan(jawaban: str, konteks: str) -> HasilPeriksa:
    dikutip, ada = ekstrak(jawaban), ekstrak(konteks)
    return HasilPeriksa(dikutip, {j: dikutip[j] - ada[j] for j in JENIS})
