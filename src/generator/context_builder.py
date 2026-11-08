"""Menyusun konteks hukum untuk prompt generator dari hasil retriever.

Format penuh (`susun_konteks_penuh`) SAMA PERSIS dengan penyusunan `context_text` di notebook retriever:
    [Konteks N]\\nNode ID: <node_id>\\n<text>      (blok digabung dengan "\\n\\n")

Dua mode pemotongan (karena prompt model dibatasi panjangnya):
* "eksperimen"     : salinan fungsi `potong_konteks` notebook generator (per blok 2.500 karakter dari awal,
                     total 12.000). Kelemahan: [PASAL TERKAIT] berada di akhir blok sehingga umumnya terbuang.
* "disempurnakan"  : DEFAULT website. Setiap blok = awal ulasan (maks 2.000 karakter) + daftar pasal (maks 8
                     entri, urutan dari database) yang SELALU disertakan. Total maks 14.000; blok yang tidak
                     muat dibuang utuh (tidak dipotong di tengah). Jumlah pasal yang tidak ditampilkan dicatat.

Tampilan sumber hukum di UI memakai hasil retriever LENGKAP, bukan teks yang dipotong ini.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_PER_DOC,
    PASAL_MAKS_DISEMPURNAKAN,
    TOTAL_MAKS_DISEMPURNAKAN,
    ULASAN_MAKS_DISEMPURNAKAN,
    get_mode_potong,
)

PENANDA_PASAL = "[PASAL TERKAIT]:"


@dataclass
class KonteksPrompt:
    teks: str
    mode: str
    n_dokumen_total: int
    n_dokumen_dipakai: int
    dipotong: bool
    panjang_penuh: int
    panjang_akhir: int
    n_pasal_tidak_ditampilkan: int = 0


# ------------------------------------------------------------------------------ format penuh
def susun_konteks_penuh(items: list) -> str:
    """Sama persis dengan notebook retriever (sel 8): blok '[Konteks N]\\nNode ID: x\\n<text>' digabung '\\n\\n'."""
    blocks = []
    for rank, res in enumerate(items, 1):
        teks_konten = res.get("text", "")
        blocks.append(f"[Konteks {rank}]\nNode ID: {res.get('node_id')}\n{teks_konten}")
    return "\n\n".join(blocks)


# ------------------------------------------------------------------------------ mode eksperimen
def potong_konteks_eksperimen(konteks, max_total=0, max_per_doc=0):
    """SALINAN PERSIS `potong_konteks` dari notebook generator (diverifikasi oleh uji kesetaraan)."""
    konteks = str(konteks)
    if max_per_doc and max_per_doc > 0:
        blocks = re.split(r"(?=\[Konteks\s+\d+\])", konteks)
        blocks = [b for b in blocks if b.strip()]
        if len(blocks) > 1:
            hasil = []
            for b in blocks:
                if len(b) > max_per_doc:
                    b = b[:max_per_doc] + "\n... [dokumen dipotong]"
                hasil.append(b)
            konteks = "\n".join(hasil)
    if max_total and len(konteks) > max_total:
        konteks = konteks[:max_total] + "\n... [konteks dipotong: melebihi batas total]"
    return konteks


# ------------------------------------------------------------------------------ mode disempurnakan
def _pisah_ulasan(item: dict) -> str:
    """Bagian ulasan dari `text` item (tanpa blok [PASAL TERKAIT])."""
    teks = item.get("text") or ""
    if not item.get("pasal_terkait"):
        return teks
    if teks.startswith(PENANDA_PASAL):
        return ""
    idx = teks.rfind("\n\n" + PENANDA_PASAL)
    return teks[:idx] if idx != -1 else teks


def _blok_disempurnakan(rank, item, ulasan_maks, pasal_maks):
    ulasan = _pisah_ulasan(item)
    ulasan_dipotong = len(ulasan) > ulasan_maks
    if ulasan_dipotong:
        ulasan = ulasan[:ulasan_maks].rstrip() + "\n... [ulasan dipotong]"

    labels = [p["label"] for p in (item.get("pasal_terkait") or [])]
    tampil = labels[:pasal_maks]
    sisa = len(labels) - len(tampil)

    bagian = []
    if ulasan.strip():
        bagian.append(ulasan)
    if tampil:
        baris = ["- " + l for l in tampil]
        if sisa:
            baris.append("- ... dan %d pasal terkait lainnya (tidak ditampilkan)" % sisa)
        bagian.append(PENANDA_PASAL + "\n" + "\n".join(baris))
    isi = "\n\n".join(bagian)
    return f"[Konteks {rank}]\nNode ID: {item.get('node_id')}\n{isi}", ulasan_dipotong, sisa


def potong_konteks_disempurnakan(
    items: list,
    ulasan_maks: int = ULASAN_MAKS_DISEMPURNAKAN,
    pasal_maks: int = PASAL_MAKS_DISEMPURNAKAN,
    total_maks: int = TOTAL_MAKS_DISEMPURNAKAN,
):
    """Return (teks, n_dipakai, ada_pemotongan, n_pasal_tidak_ditampilkan)."""
    blocks, ada_potong, pasal_hilang = [], False, 0
    total = 0
    for item in items:
        blok, ul_dipotong, sisa = _blok_disempurnakan(len(blocks) + 1, item, ulasan_maks, pasal_maks)
        tambahan = len(blok) + (2 if blocks else 0)
        if blocks and total + tambahan > total_maks:
            ada_potong = True  # blok ini dan sesudahnya dibuang utuh
            break
        blocks.append(blok)
        total += tambahan
        ada_potong = ada_potong or ul_dipotong or sisa > 0
        pasal_hilang += sisa
    if len(blocks) < len(items):
        ada_potong = True
    return "\n\n".join(blocks), len(blocks), ada_potong, pasal_hilang


# ------------------------------------------------------------------------------ antarmuka utama
def siapkan_konteks(items: list, mode: str | None = None) -> KonteksPrompt:
    """Bangun konteks akhir untuk prompt. `mode`: 'disempurnakan' | 'eksperimen' (default dari konfigurasi)."""
    mode = mode or get_mode_potong()
    penuh = susun_konteks_penuh(items)

    if mode == "eksperimen":
        teks = potong_konteks_eksperimen(penuh, MAX_CONTEXT_CHARS, MAX_CONTEXT_CHARS_PER_DOC)
        dipakai = len(re.findall(r"\[Konteks\s+\d+\]", teks))
        return KonteksPrompt(teks, mode, len(items), dipakai, teks != penuh, len(penuh), len(teks))
    if mode == "disempurnakan":
        teks, dipakai, potong, pasal_hilang = potong_konteks_disempurnakan(items)
        return KonteksPrompt(teks, mode, len(items), dipakai, potong, len(penuh), len(teks), pasal_hilang)
    raise ValueError("Mode pemotongan tidak dikenal: %r" % (mode,))
