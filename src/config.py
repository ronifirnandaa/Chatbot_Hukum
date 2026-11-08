"""Konfigurasi terpusat aplikasi.

Prinsip:
* Kredensial TIDAK pernah ditulis di source code. Sumber (berurutan):
  1. environment variable,
  2. Streamlit Secrets (.streamlit/secrets.toml lokal / Settings > Secrets di Hugging Face Spaces),
  3. berkas `.env` di folder project (untuk pengembangan lokal tanpa Streamlit).
* Modul ini tidak membutuhkan Streamlit untuk diimpor, sehingga dapat diuji mandiri.
* Nilai rahasia tidak pernah muncul di pesan error maupun repr().
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --------------------------------------------------------------------------------------
# Konstanta retriever (nilai diambil APA ADANYA dari notebook retriever)
# --------------------------------------------------------------------------------------
MODEL_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM_EMBEDDING = 384
INDEKS_VEKTOR_WAJIB = ("legalqa_vector", "law_vector")

K_DEFAULT = 5               # K untuk chatbot (keputusan: selaras dengan batas konteks generator)
K_EVALUASI = (5, 10, 20)    # nilai K pada evaluasi di notebook

# Tanda tangan KG final (dari output notebook retriever, instance yang dikonfirmasi final).
# Dipakai untuk memperingatkan bila database yang tersambung berbeda dari yang dievaluasi.
TANDA_TANGAN_KG_FINAL = {"LegalQA": 1010, "Law": 9066, "REFERENCES": 32063}

# --------------------------------------------------------------------------------------
# Rahasia
# --------------------------------------------------------------------------------------
NEO4J_KEYS = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
SKEMA_URI_VALID = ("neo4j", "neo4j+s", "neo4j+ssc", "bolt", "bolt+s", "bolt+ssc")


class ConfigError(Exception):
    """Konfigurasi tidak lengkap/tidak valid. Pesan aman ditampilkan ke pengguna (tanpa nilai rahasia)."""


_DOTENV_SUDAH_DIMUAT = False


def muat_dotenv(path: Path | None = None, override: bool = False) -> bool:
    """Muat `.env` sederhana (KEY=VALUE) ke os.environ. Tidak menimpa variabel yang sudah ada."""
    global _DOTENV_SUDAH_DIMUAT
    f = Path(path) if path else BASE_DIR / ".env"
    _DOTENV_SUDAH_DIMUAT = True
    if not f.is_file():
        return False
    for baris in f.read_text(encoding="utf-8").splitlines():
        baris = baris.strip()
        if not baris or baris.startswith("#") or "=" not in baris:
            continue
        k, _, v = baris.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and (override or k not in os.environ):
            os.environ[k] = v
    return True


def _dari_streamlit_secrets(nama: str):
    try:
        import streamlit as st  # impor malas: modul ini tetap bisa dipakai tanpa Streamlit

        return st.secrets.get(nama)
    except Exception:  # tidak ada Streamlit / tidak ada berkas secrets
        return None


def get_secret(nama: str, default: str | None = None) -> str | None:
    """Ambil nilai rahasia: environment -> Streamlit Secrets -> .env."""
    nilai = os.environ.get(nama)
    if not (nilai and str(nilai).strip()):
        nilai = _dari_streamlit_secrets(nama)
    if not (nilai and str(nilai).strip()) and not _DOTENV_SUDAH_DIMUAT:
        muat_dotenv()
        nilai = os.environ.get(nama)
    if nilai is None or not str(nilai).strip():
        return default
    return str(nilai).strip()


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    username: str
    password: str
    database: str = "neo4j"

    def __repr__(self) -> str:  # jangan pernah membocorkan password
        return "Neo4jSettings(uri=<disembunyikan>, username=<disembunyikan>, password=<disembunyikan>, database=%r)" % (
            self.database,
        )

    __str__ = __repr__


def load_neo4j_settings() -> Neo4jSettings:
    """Muat kredensial Neo4j. Gagal dengan ConfigError yang hanya menyebut NAMA variabel yang kurang."""
    nilai = {k: get_secret(k) for k in NEO4J_KEYS}
    kurang = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not nilai[k]]
    if kurang:
        raise ConfigError(
            "Konfigurasi database belum lengkap. Variabel yang belum diisi: %s. "
            "Isi di Streamlit Secrets / environment / berkas .env (lihat secrets.toml.example)." % ", ".join(kurang)
        )
    skema = urlparse(nilai["NEO4J_URI"]).scheme
    if skema not in SKEMA_URI_VALID:
        raise ConfigError(
            "NEO4J_URI tidak valid: skema harus salah satu dari %s." % ", ".join(SKEMA_URI_VALID)
        )
    return Neo4jSettings(
        uri=nilai["NEO4J_URI"],
        username=nilai["NEO4J_USERNAME"],
        password=nilai["NEO4J_PASSWORD"],
        database=nilai["NEO4J_DATABASE"] or "neo4j",
    )


# --------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------
# Parameter generation: sama dengan eksperimen di notebook generator (do_sample=False -> temperature 0,
# MAX_NEW_TOKENS=1024). Catatan: pada notebook, num_ctx=8192 dan think=False adalah pengaturan Ollama;
# keduanya tidak berlaku untuk API Google.
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TIMEOUT_GENERASI_DETIK = 120
# Mode berpikir Gemma 4 dimatikan ("minimal"), setara dengan think=False pada Qwen di notebook. Token berpikir ikut
# dihitung dalam batas MAX_NEW_TOKENS, sehingga jika dinyalakan jawaban bisa kosong (finish_reason=MAX_TOKENS).
THINKING_LEVEL = "minimal"

# Pemotongan konteks. Mode "eksperimen" = persis notebook. Mode "disempurnakan" = keputusan untuk website
# (daftar [PASAL TERKAIT] tidak ikut terpotong). Mode dapat diganti lewat secret/env MODE_POTONG.
MODE_POTONG_VALID = ("disempurnakan", "eksperimen")
MODE_POTONG_DEFAULT = "disempurnakan"
MAX_CONTEXT_CHARS = 12000           # eksperimen: batas total
MAX_CONTEXT_CHARS_PER_DOC = 2500    # eksperimen: batas per blok [Konteks N]
ULASAN_MAKS_DISEMPURNAKAN = 2000    # disempurnakan: karakter awal ulasan per blok
PASAL_MAKS_DISEMPURNAKAN = 8        # disempurnakan: jumlah entri pasal per blok
TOTAL_MAKS_DISEMPURNAKAN = 14000    # disempurnakan: batas total

# Pesan baku dari system prompt hybrid (aturan nomor 6) bila konteks kosong.
PESAN_KONTEKS_KOSONG = "Maaf, informasi hukum tersebut tidak tersedia dalam basis data saya."

# --- Penyedia model dan daftar model ---------------------------------------------------------------------------------
# Model dikenali dengan kunci "penyedia:id" (mis. "groq:qwen/qwen3.8-27b", "google:gemma-4-26b-a4b-it"). Entri tanpa
# awalan dianggap model Google (kompatibel dengan konfigurasi lama). Penyedia AKTIF = yang kunci API-nya terisi.
# Urutan = urutan di dropdown; yang pertama menjadi default. Groq diletakkan lebih dulu.
PENYEDIA_VALID = ("groq", "google")
MODEL_DEFAULT_PER_PENYEDIA = {
    "groq": ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"),
    "google": ("gemma-4-26b-a4b-it", "gemma-4-31b-it"),
}
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Model GPT-OSS di Groq tidak dapat mematikan penalaran (hanya low/medium/high); token penalaran ikut dihitung dalam
# batas keluaran. Batas keluaran ditambah agar jawaban tidak kosong/terpotong akibat penalaran.
GROQ_HEADROOM_REASONING = 512
# Satu-satunya model yang punya hasil evaluasi RAGAS di notebook (dijalankan lewat Ollama, bukan di website).
MODEL_DIEVALUASI = "qwen3.5:4b"


def get_mode_potong() -> str:
    mode = (get_secret("MODE_POTONG", MODE_POTONG_DEFAULT) or MODE_POTONG_DEFAULT).lower()
    if mode not in MODE_POTONG_VALID:
        raise ConfigError("MODE_POTONG tidak valid: harus salah satu dari %s." % ", ".join(MODE_POTONG_VALID))
    return mode


def get_google_api_key() -> str:
    kunci = get_secret("GOOGLE_API_KEY")
    if not kunci:
        raise ConfigError(
            "Konfigurasi generator belum lengkap. Variabel yang belum diisi: GOOGLE_API_KEY. "
            "Isi di Streamlit Secrets / environment / berkas .env."
        )
    return kunci


def get_groq_api_key() -> str:
    kunci = get_secret("GROQ_API_KEY")
    if not kunci:
        raise ConfigError(
            "Konfigurasi generator belum lengkap. Variabel yang belum diisi: GROQ_API_KEY. "
            "Isi di Streamlit Secrets / environment / berkas .env."
        )
    return kunci


def get_penyedia_aktif() -> list[str]:
    """Penyedia yang kunci API-nya terisi, berurutan sesuai PENYEDIA_VALID."""
    kunci = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY"}
    return [p for p in PENYEDIA_VALID if get_secret(kunci[p])]


def pecah_kunci_model(kunci: str) -> tuple:
    """'groq:qwen/qwen3.8-27b' -> ('groq', 'qwen/qwen3.8-27b'). Tanpa awalan penyedia valid -> ('google', kunci)."""
    awalan, pemisah, sisa = kunci.partition(":")
    if pemisah and awalan in PENYEDIA_VALID and sisa:
        return awalan, sisa
    return "google", kunci


def get_model_ids() -> list[str]:
    """Daftar kunci model terkonfigurasi ("penyedia:id"). Dari GENERATOR_MODELS bila diisi; jika tidak, default
    semua penyedia aktif. Tanpa duplikat, urutan dipertahankan."""
    mentah = get_secret("GENERATOR_MODELS")
    if mentah:
        entri = [m.strip() for m in mentah.split(",") if m.strip()]
        kunci = ["%s:%s" % pecah_kunci_model(e) for e in entri]
    else:
        kunci = ["%s:%s" % (p, m) for p in get_penyedia_aktif() for m in MODEL_DEFAULT_PER_PENYEDIA[p]]
    return list(dict.fromkeys(kunci))


# --------------------------------------------------------------------------------------
# Chatbot
# --------------------------------------------------------------------------------------
MAKS_PANJANG_PERTANYAAN = 2000
# Kuota API gratis dipakai bersama semua pengunjung. Pembatas per sesi hanya mengurangi penyalahgunaan sederhana
# (sesi baru melewatinya); pembatas GLOBAL yang melindungi kuota. Dapat diganti lewat secret/env.
RATE_SESI_MAKS_DEFAULT = 5
RATE_SESI_JENDELA_DETIK = 600
RATE_GLOBAL_MAKS_DEFAULT = 8
RATE_GLOBAL_JENDELA_DETIK = 60


def _int_secret(nama: str, default: int) -> int:
    nilai = get_secret(nama)
    try:
        return max(0, int(nilai)) if nilai is not None else default
    except ValueError:
        raise ConfigError("%s harus berupa bilangan bulat." % nama) from None


def get_batas_laju() -> dict:
    """{'sesi': (maks, jendela_detik), 'global': (maks, jendela_detik)}."""
    return {
        "sesi": (_int_secret("RATE_SESI_MAKS", RATE_SESI_MAKS_DEFAULT), RATE_SESI_JENDELA_DETIK),
        "global": (_int_secret("RATE_GLOBAL_MAKS", RATE_GLOBAL_MAKS_DEFAULT), RATE_GLOBAL_JENDELA_DETIK),
    }


def get_peringatan_kutipan() -> bool:
    """Tampilkan peringatan bila jawaban mengutip pasal/peraturan yang tidak ada di konteks (default: ya)."""
    return (get_secret("PERINGATAN_KUTIPAN", "true") or "true").lower() not in ("0", "false", "tidak", "no", "off")


# --------------------------------------------------------------------------------------
# Gaya jawaban
# --------------------------------------------------------------------------------------
# "naratif" = prompt PENELITIAN (persis notebook). "ringkas" = varian website yang langsung ke inti (bukan dari
# notebook, belum dievaluasi dengan RAGAS). Default dapat diganti lewat secret/env GAYA_JAWABAN.
GAYA_VALID = ("ringkas", "naratif")
GAYA_DEFAULT = "ringkas"


def get_gaya_default() -> str:
    gaya = (get_secret("GAYA_JAWABAN", GAYA_DEFAULT) or GAYA_DEFAULT).lower()
    if gaya not in GAYA_VALID:
        raise ConfigError("GAYA_JAWABAN tidak valid: harus salah satu dari %s." % ", ".join(GAYA_VALID))
    return gaya
