"""Adapter ke Google Gemini API (tier gratis; model Gemma 4) untuk generator.

Isi prompt (system prompt, few-shot, pertanyaan+konteks) berasal dari `prompt_template.bangun_messages`
dan TIDAK diubah di sini. Adapter hanya menerjemahkan format pesan notebook (system/user/assistant) ke format API:
  * system  -> `system_instruction`; bila model menolaknya (HTTP 400), system prompt digabung ke pesan user pertama,
  * user    -> role "user", assistant -> role "model",
  * parameter: temperature 0 dan max_output_tokens 1024 (sama dengan eksperimen),
  * mode berpikir dimatikan (`thinking_level="minimal"`), setara `think=False` pada Qwen di notebook. Tanpa ini,
    token berpikir menghabiskan batas 1024 dan jawaban kosong (finish_reason=MAX_TOKENS). Bila suatu model menolak
    `thinking_config` (HTTP 400), pengaturan itu dilepas untuk model tersebut.

Pesan galat untuk pengguna tidak memuat kunci API maupun detail internal; detail ada di `detail_teknis`.
SDK `google-genai` diimpor secara malas, dan klien dapat disuntikkan sehingga modul ini dapat diuji offline.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from src.config import MAX_NEW_TOKENS, TEMPERATURE, THINKING_LEVEL, TIMEOUT_GENERASI_DETIK

log = logging.getLogger(__name__)


class GenerasiError(RuntimeError):
    """Galat generasi. `str(e)` aman ditampilkan ke pengguna; `detail_teknis` hanya untuk log."""

    def __init__(self, pesan_pengguna: str, detail_teknis: str = ""):
        super().__init__(pesan_pengguna)
        self.detail_teknis = detail_teknis


class KuotaHabisError(GenerasiError):
    """Kuota/laju API gratis terlampaui (HTTP 429)."""


class ModelTidakTersediaError(GenerasiError):
    """Model tidak ditemukan / tidak dapat diakses (HTTP 404)."""


@dataclass
class HasilLLM:
    teks: str
    selesai_normal: bool          # False bila berhenti karena batas panjang (MAX_TOKENS) dsb.
    finish_reason: str | None = None
    token_masuk: int | None = None    # token prompt menurut API (bila tersedia)
    token_keluar: int | None = None   # token keluaran menurut API (bila tersedia)


def buat_klien_google(api_key: str, timeout_detik: float = TIMEOUT_GENERASI_DETIK):
    """Buat klien Google GenAI (impor malas). Timeout dalam milidetik pada http_options."""
    from google import genai

    return genai.Client(api_key=api_key, http_options={"timeout": int(timeout_detik * 1000)})


# ---------------------------------------------------------------------------------- util galat
def _kode_http(e: Exception):
    kode = getattr(e, "code", None)
    if isinstance(kode, int):
        return kode
    m = re.match(r"\s*(\d{3})\b", str(e))
    return int(m.group(1)) if m else None


def _transien(e: Exception) -> bool:
    kode = _kode_http(e)
    if kode is not None and kode >= 500:
        return True
    cek = (type(e).__name__ + " " + str(e)).lower()
    return any(t in cek for t in ("timeout", "timed out", "connecterror", "connection", "readerror", "unavailable"))


def _pesan_api(e: Exception) -> str:
    return str(getattr(e, "message", None) or e)


def _bersihkan(teks, rahasia, maks=300) -> str:
    t = str(teks)
    for r in rahasia:
        if r and len(r) >= 6:
            t = t.replace(r, "***")
    return t[:maks]


def _terjemahkan(e: Exception, rahasia=()) -> GenerasiError:
    kode = _kode_http(e)
    detail = _bersihkan("%s [%s]: %s" % (type(e).__name__, kode, _pesan_api(e)), rahasia)
    if kode == 429:
        return KuotaHabisError(
            "Kuota layanan model gratis sedang habis atau terlalu banyak permintaan. Coba lagi beberapa saat.", detail
        )
    if kode in (401, 403):
        return GenerasiError("Layanan model menolak akses. Hubungi pengelola website.", detail)
    if kode == 404:
        return ModelTidakTersediaError("Model yang dipilih tidak tersedia saat ini. Pilih model lain.", detail)
    if kode == 400:
        return GenerasiError("Permintaan ke model tidak dapat diproses.", detail)
    if _transien(e):
        return GenerasiError("Layanan model sedang sibuk atau tidak merespons. Coba lagi, atau pilih model lain.", detail)
    return GenerasiError("Terjadi kesalahan saat menghasilkan jawaban.", detail)


# ---------------------------------------------------------------------------------- klien
class GeminiLLM:
    def __init__(self, sdk_client, sleep=time.sleep, rahasia: tuple = ()):
        self.client = sdk_client
        self._sleep = sleep
        self._rahasia = tuple(rahasia)
        self._gabung_system: set = set()  # model yang menolak system_instruction -> system digabung ke user
        self._tanpa_thinking: set = set()  # model yang menolak thinking_config -> dikirim tanpa pengaturan itu

    # ------------------------------------------------------------ daftar model
    def daftar_model(self) -> set:
        """ID model yang terdaftar di API dan mendukung generateContent (tanpa awalan 'models/')."""
        try:
            hasil = set()
            for m in self.client.models.list():
                aksi = getattr(m, "supported_actions", None)
                if aksi and "generateContent" not in aksi:
                    continue
                hasil.add(str(m.name).replace("models/", "", 1))
            return hasil
        except Exception as e:  # noqa: BLE001
            raise _terjemahkan(e, self._rahasia) from None

    # ------------------------------------------------------------ generasi
    @staticmethod
    def _pecah(messages: list):
        system = None
        sisa = list(messages)
        if sisa and sisa[0].get("role") == "system":
            system = sisa.pop(0)["content"]
        return system, sisa

    @staticmethod
    def _contents(pesan: list, gabung_system: str | None) -> list:
        contents = []
        for i, m in enumerate(pesan):
            teks = m["content"]
            if i == 0 and gabung_system:
                teks = gabung_system + "\n\n" + teks
            contents.append({"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": teks}]})
        return contents

    def _panggil(self, model_id, system, pesan, max_new_tokens, temperature, gabung, thinking=None):
        config = {"temperature": temperature, "max_output_tokens": max_new_tokens}
        if thinking:
            config["thinking_config"] = {"thinking_level": thinking}
        if system and not gabung:
            config["system_instruction"] = system
        contents = self._contents(pesan, system if gabung else None)
        return self.client.models.generate_content(model=model_id, contents=contents, config=config)

    @staticmethod
    def _ekstrak(resp) -> HasilLLM:
        kandidat = (getattr(resp, "candidates", None) or [None])[0]
        alasan = getattr(kandidat, "finish_reason", None) if kandidat is not None else None
        alasan = str(getattr(alasan, "name", alasan)) if alasan is not None else None
        bagian = getattr(getattr(kandidat, "content", None), "parts", None) or []
        teks = "".join(
            p.text for p in bagian if getattr(p, "text", None) and not getattr(p, "thought", False)
        ).strip()
        if not teks:
            n_thought = sum(1 for p in bagian if getattr(p, "thought", False))
            token_pikir = getattr(getattr(resp, "usage_metadata", None), "thoughts_token_count", None)
            raise GenerasiError(
                "Model tidak menghasilkan jawaban. Coba ulangi atau pilih model lain.",
                "finish_reason=%s; bagian_thought=%d; thoughts_token_count=%s" % (alasan, n_thought, token_pikir),
            )
        normal = alasan in (None, "STOP", "FinishReason.STOP")
        pakai = getattr(resp, "usage_metadata", None)
        masuk, keluar = getattr(pakai, "prompt_token_count", None), getattr(pakai, "candidates_token_count", None)
        return HasilLLM(teks=teks, selesai_normal=normal, finish_reason=alasan,
                        token_masuk=masuk if isinstance(masuk, int) else None,
                        token_keluar=keluar if isinstance(keluar, int) else None)

    def generate(self, model_id: str, messages: list, max_new_tokens: int = MAX_NEW_TOKENS,
                 temperature: float = TEMPERATURE, percobaan: int = 3,
                 thinking_level: str | None = THINKING_LEVEL) -> HasilLLM:
        system, pesan = self._pecah(messages)
        gabung = model_id in self._gabung_system
        i = 0
        while True:
            thinking = thinking_level if model_id not in self._tanpa_thinking else None
            try:
                resp = self._panggil(model_id, system, pesan, max_new_tokens, temperature, gabung, thinking)
                return self._ekstrak(resp)
            except GenerasiError:
                raise
            except Exception as e:  # noqa: BLE001
                # Model yang tidak mendukung system_instruction: gabungkan ke pesan user pertama, ulangi sekali.
                if (not gabung and system and _kode_http(e) == 400
                        and "instruction" in _pesan_api(e).lower()):
                    log.warning("model %s menolak system_instruction; system digabung ke pesan pertama", model_id)
                    self._gabung_system.add(model_id)
                    gabung = True
                    continue
                # Model yang tidak mengenal thinking_config: lepas pengaturan itu, ulangi sekali.
                if thinking and _kode_http(e) == 400 and "thinking" in _pesan_api(e).lower():
                    log.warning("model %s menolak thinking_config; dikirim tanpa pengaturan itu", model_id)
                    self._tanpa_thinking.add(model_id)
                    continue
                if _transien(e) and i < percobaan - 1:
                    i += 1
                    log.warning("retry %d/%d untuk %s", i, percobaan, model_id)
                    self._sleep(2.0 * i)
                    continue
                raise _terjemahkan(e, self._rahasia) from None
