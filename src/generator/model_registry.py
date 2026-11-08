"""Daftar model generator yang ditawarkan di chatbot.

Daftar berasal dari konfigurasi (`GENERATOR_MODELS` atau default penyedia aktif), sehingga model dapat ditambah atau
dihapus tanpa mengubah kode. Model dikenali dengan kunci "penyedia:id". Hanya model yang BENAR-BENAR terdaftar di API
penyedianya yang ditampilkan. `dievaluasi` = model punya hasil evaluasi RAGAS di notebook (saat ini hanya qwen3.5:4b via
Ollama, yang tidak dilayani website ini), sehingga model di chatbot diberi label "belum dievaluasi".
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import MODEL_DIEVALUASI, get_model_ids, pecah_kunci_model

NAMA_PENYEDIA = {"groq": "Groq", "google": "Google"}
LABEL_RAMAH = {
    "groq:qwen/qwen3.8-27b": "Qwen 3.8 27B (Groq)",
    "groq:openai/gpt-oss-120b": "GPT-OSS 120B (Groq)",
    "groq:openai/gpt-oss-20b": "GPT-OSS 20B (Groq)",
    "google:gemma-4-31b-it": "Gemma 4 31B (Google)",
    "google:gemma-4-26b-a4b-it": "Gemma 4 26B-A4B (Google)",
}


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    dievaluasi: bool
    provider: str = "google"

    @property
    def kunci(self) -> str:
        """Pengenal unik "penyedia:id"; inilah yang diteruskan ke router LLM."""
        return "%s:%s" % (self.provider, self.id)

    @property
    def label_lengkap(self) -> str:
        return self.label


def model_terkonfigurasi(ids: list | None = None) -> list:
    ids = ids if ids is not None else get_model_ids()
    hasil = []
    for entri in ids:
        penyedia, model_id = pecah_kunci_model(entri)
        label = LABEL_RAMAH.get("%s:%s" % (penyedia, model_id), "%s (%s)" % (model_id, NAMA_PENYEDIA[penyedia]))
        hasil.append(ModelInfo(model_id, label, model_id == MODEL_DIEVALUASI, penyedia))
    return hasil


def model_tersedia(llm, terkonfigurasi: list | None = None) -> list:
    """Saring daftar terkonfigurasi: hanya yang terdaftar di API. Melempar GenerasiError bila semua penyedia gagal."""
    terkonfigurasi = terkonfigurasi if terkonfigurasi is not None else model_terkonfigurasi()
    ada = llm.daftar_model()  # himpunan kunci "penyedia:id"
    return [m for m in terkonfigurasi if m.kunci in ada]
