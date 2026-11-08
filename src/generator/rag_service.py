"""Menyambungkan hasil retriever ke generator: konteks -> prompt -> model terpilih -> jawaban."""
from __future__ import annotations

import time
from dataclasses import dataclass

from src.config import MAX_NEW_TOKENS, PESAN_KONTEKS_KOSONG, TEMPERATURE
from src.generator.context_builder import KonteksPrompt, siapkan_konteks
from src.generator.gaya_jawaban import GAYA_NARATIF, bangun_messages_gaya


@dataclass
class JawabanRAG:
    jawaban: str
    model_id: str
    konteks: KonteksPrompt
    waktu_generasi: float
    selesai_normal: bool = True
    memanggil_llm: bool = True
    gaya: str = GAYA_NARATIF
    token_masuk: int | None = None
    token_keluar: int | None = None


def hasilkan_jawaban(pertanyaan: str, items: list, model_id: str, llm, mode_potong: str | None = None,
                     max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE,
                     gaya: str = GAYA_NARATIF) -> JawabanRAG:
    """`items`: hasil `HybridGraphRAGRetriever.cari_dengan_info(...).items`. `llm`: objek dengan `.generate(...)`.
    `gaya`: 'naratif' (prompt penelitian, default) atau 'ringkas' (varian website).

    Bila konteks kosong, jawaban baku dari aturan nomor 6 system prompt dikembalikan TANPA memanggil model
    (menghemat kuota gratis; hasilnya sama dengan yang diperintahkan prompt)."""
    if not isinstance(pertanyaan, str) or not pertanyaan.strip():
        raise ValueError("Pertanyaan tidak boleh kosong.")
    if not model_id:
        raise ValueError("Model generator belum dipilih.")

    konteks = siapkan_konteks(items, mode_potong)
    if not items or not konteks.teks.strip():
        return JawabanRAG(PESAN_KONTEKS_KOSONG, model_id, konteks, 0.0, True, False, gaya)

    messages = bangun_messages_gaya(pertanyaan.strip(), konteks.teks, gaya)
    mulai = time.perf_counter()
    hasil = llm.generate(model_id, messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return JawabanRAG(hasil.teks, model_id, konteks, time.perf_counter() - mulai, hasil.selesai_normal, True, gaya,
                      getattr(hasil, "token_masuk", None), getattr(hasil, "token_keluar", None))
