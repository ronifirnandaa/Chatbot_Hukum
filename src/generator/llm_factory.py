"""Membuat objek `llm` (router) dari konfigurasi: penyedia aktif = yang kunci API-nya terisi."""
from __future__ import annotations

from src.config import ConfigError, get_google_api_key, get_groq_api_key, get_penyedia_aktif
from src.generator.groq_client import GroqLLM
from src.generator.llm_client import GeminiLLM, buat_klien_google
from src.generator.llm_router import LLMRouter


def buat_llm_dari_konfigurasi(http=None) -> LLMRouter:
    aktif = get_penyedia_aktif()
    if not aktif:
        raise ConfigError(
            "Konfigurasi generator belum lengkap. Variabel yang belum diisi: GROQ_API_KEY dan/atau GOOGLE_API_KEY. "
            "Isi salah satu di Streamlit Secrets / environment / berkas .env."
        )
    penyedia, dilewati = {}, {}
    if "groq" in aktif:
        penyedia["groq"] = GroqLLM(get_groq_api_key(), http=http)
    if "google" in aktif:
        kunci = get_google_api_key()
        try:
            penyedia["google"] = GeminiLLM(buat_klien_google(kunci), rahasia=(kunci,))
        except ImportError:
            if not penyedia:
                raise
            dilewati["google"] = "Paket 'google-genai' belum terpasang; model Google tidak ditampilkan."
    return LLMRouter(penyedia, dilewati)
