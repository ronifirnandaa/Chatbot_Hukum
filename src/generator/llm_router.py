"""Router penyedia LLM: meneruskan permintaan ke penyedia yang tepat berdasarkan kunci model "penyedia:id".

Bagian lain aplikasi (rag_service, chat_service, halaman) hanya mengenal satu objek `llm` dengan `daftar_model()` dan
`generate(model_key, messages, ...)`, sehingga tidak perlu tahu model berasal dari Groq atau Google.
"""
from __future__ import annotations

from src.generator.llm_client import GenerasiError, ModelTidakTersediaError


class LLMRouter:
    def __init__(self, penyedia: dict, penyedia_dilewati: dict | None = None):
        self.penyedia = dict(penyedia)               # {"groq": GroqLLM, "google": GeminiLLM}
        self.penyedia_dilewati = dict(penyedia_dilewati or {})  # {"google": "alasan"} (mis. paket belum terpasang)
        self.galat_penyedia: dict = {}               # galat pada pemeriksaan daftar model terakhir

    def peringatan_penyedia(self) -> dict:
        """Masalah penyedia yang perlu ditampilkan (dilewati saat dibuat + gagal saat memeriksa daftar model)."""
        return {**self.penyedia_dilewati, **self.galat_penyedia}

    def daftar_model(self) -> set:
        """Kunci model ("penyedia:id") yang terdaftar. Gagalnya satu penyedia tidak menyembunyikan penyedia lain;
        bila SEMUA gagal, galat pertama dilempar."""
        ada, galat, detail = set(), {}, []
        for nama, llm in self.penyedia.items():
            try:
                ada |= {"%s:%s" % (nama, i) for i in llm.daftar_model()}
            except GenerasiError as e:
                galat[nama] = str(e)
                detail.append("%s: %s" % (nama, e.detail_teknis or e))
        self.galat_penyedia = galat
        if galat and len(galat) == len(self.penyedia):
            raise GenerasiError(next(iter(galat.values())), " | ".join(detail))
        return ada

    def generate(self, model_key: str, messages: list, **kw):
        nama, pemisah, model_id = model_key.partition(":")
        if not pemisah or nama not in self.penyedia or not model_id:
            raise ModelTidakTersediaError("Model yang dipilih tidak tersedia saat ini. Pilih model lain.",
                                          "kunci model tidak dikenal: %r" % (model_key,))
        return self.penyedia[nama].generate(model_id, messages, **kw)
