"""Adapter ke Groq (API kompatibel OpenAI: POST /chat/completions, GET /models).

Format pesan notebook (system/user/assistant) dikirim APA ADANYA; isi prompt tidak diubah di sini.
Parameter: temperature 0 dan batas keluaran 1024 (sama dengan eksperimen).

Penalaran (menurut dokumentasi Groq):
  * qwen/*          : `reasoning_effort="none"` mematikan penalaran,
  * openai/gpt-oss* : penalaran tidak bisa dimatikan (hanya low/medium/high) -> `low` + `include_reasoning=false`,
                      dan batas keluaran ditambah GROQ_HEADROOM_REASONING karena token penalaran ikut terhitung.
  Nilai di luar himpunan yang didukung model ditolak HTTP 400; bila galat 400 menyebut "reasoning", pengaturan itu
  dilepas untuk model tersebut dan permintaan diulang sekali.

HTTP diinjeksikan (default: requests.Session) agar dapat diuji offline. Pesan galat untuk pengguna tidak memuat kunci API.
"""
from __future__ import annotations

import logging
import math
import re
import time

from src.config import GROQ_BASE_URL, GROQ_HEADROOM_REASONING, MAX_NEW_TOKENS, TEMPERATURE, TIMEOUT_GENERASI_DETIK
from src.generator.llm_client import (
    GenerasiError,
    HasilLLM,
    KuotaHabisError,
    ModelTidakTersediaError,
    _bersihkan,
)

log = logging.getLogger(__name__)
_RE_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
PESAN_SIBUK = "Layanan model sedang sibuk atau tidak merespons. Coba lagi, atau pilih model lain."


def pengaturan_reasoning(model_id: str) -> dict:
    if model_id.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    if model_id.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    return {}


def _pesan_api(resp) -> str:
    try:
        j = resp.json()
        return str(((j.get("error") or {}).get("message")) or j)[:300]
    except Exception:  # noqa: BLE001
        return (getattr(resp, "text", "") or "")[:300]


class GroqLLM:
    def __init__(self, api_key: str, http=None, base_url: str = GROQ_BASE_URL, sleep=time.sleep,
                 timeout: float = TIMEOUT_GENERASI_DETIK):
        self._rahasia = (api_key,)
        self._headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        self.base_url, self._sleep, self.timeout = base_url.rstrip("/"), sleep, timeout
        if http is None:
            import requests  # impor malas

            http = requests.Session()
        self.http = http
        self._tanpa_reasoning: set = set()  # model yang menolak parameter penalaran

    # ------------------------------------------------------------ galat
    def _retry_after(self, resp):
        try:
            nilai = float((getattr(resp, "headers", None) or {}).get("retry-after"))
            return nilai if nilai > 0 else None
        except (TypeError, ValueError):
            return None

    def _dari_http(self, resp) -> GenerasiError:
        kode = resp.status_code
        detail = _bersihkan("HTTP %s: %s" % (kode, _pesan_api(resp)), self._rahasia)
        if kode == 429:
            tunggu = self._retry_after(resp)
            akhir = "Coba lagi dalam sekitar %d detik." % math.ceil(tunggu) if tunggu else "Coba lagi beberapa saat."
            return KuotaHabisError(
                "Kuota layanan model gratis sedang habis atau terlalu banyak permintaan. " + akhir, detail)
        if kode == 413:
            return GenerasiError(
                "Permintaan terlalu besar untuk batas token model ini. Coba lagi sebentar lagi atau pilih model lain.",
                detail)
        if kode in (401, 403):
            return GenerasiError("Layanan model menolak akses. Hubungi pengelola website.", detail)
        if kode == 404:
            return ModelTidakTersediaError("Model yang dipilih tidak tersedia saat ini. Pilih model lain.", detail)
        if kode == 400:
            return GenerasiError("Permintaan ke model tidak dapat diproses.", detail)
        if kode >= 500:
            return GenerasiError(PESAN_SIBUK, detail)
        return GenerasiError("Terjadi kesalahan saat menghasilkan jawaban.", detail)

    def _dari_exc(self, e: Exception) -> GenerasiError:
        detail = _bersihkan("%s: %s" % (type(e).__name__, e), self._rahasia)
        if self._transien_exc(e):
            return GenerasiError(PESAN_SIBUK, detail)
        return GenerasiError("Terjadi kesalahan saat menghubungi layanan model.", detail)

    @staticmethod
    def _transien_exc(e: Exception) -> bool:
        cek = (type(e).__name__ + " " + str(e)).lower()
        return any(t in cek for t in ("timeout", "timed out", "connection", "connect", "readerror", "unavailable"))

    # ------------------------------------------------------------ daftar model
    def daftar_model(self) -> set:
        """ID model aktif di akun Groq."""
        try:
            resp = self.http.get(self.base_url + "/models", headers=self._headers, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            raise self._dari_exc(e) from None
        if resp.status_code != 200:
            raise self._dari_http(resp)
        try:
            data = resp.json().get("data", [])
        except Exception:  # noqa: BLE001
            raise GenerasiError("Terjadi kesalahan saat membaca daftar model.", "respons /models bukan JSON") from None
        return {m["id"] for m in data if m.get("id") and m.get("active", True)}

    # ------------------------------------------------------------ generasi
    def _payload(self, model_id, messages, max_new_tokens, temperature, tanpa_reasoning) -> dict:
        batas = max_new_tokens + (GROQ_HEADROOM_REASONING if model_id.startswith("openai/gpt-oss") else 0)
        p = {"model": model_id, "messages": messages, "temperature": temperature,
             "max_completion_tokens": batas, "stream": False}
        if not tanpa_reasoning:
            p.update(pengaturan_reasoning(model_id))
        return p

    def _ekstrak(self, resp) -> HasilLLM:
        try:
            data = resp.json()
            pilihan = data["choices"][0]
        except Exception:  # noqa: BLE001
            raise GenerasiError("Respons model tidak dapat dibaca.", "respons bukan JSON chat completion") from None
        pesan = pilihan.get("message") or {}
        teks = _RE_THINK.sub("", pesan.get("content") or "").strip()
        alasan = pilihan.get("finish_reason")
        pakai = data.get("usage") or {}
        if not teks:
            reasoning = (pakai.get("completion_tokens_details") or {}).get("reasoning_tokens")
            raise GenerasiError(
                "Model tidak menghasilkan jawaban. Coba ulangi atau pilih model lain.",
                "finish_reason=%s; completion_tokens=%s; reasoning_tokens=%s"
                % (alasan, pakai.get("completion_tokens"), reasoning),
            )
        masuk, keluar = pakai.get("prompt_tokens"), pakai.get("completion_tokens")
        return HasilLLM(teks=teks, selesai_normal=alasan in (None, "stop"), finish_reason=alasan,
                        token_masuk=masuk if isinstance(masuk, int) else None,
                        token_keluar=keluar if isinstance(keluar, int) else None)

    def generate(self, model_id: str, messages: list, max_new_tokens: int = MAX_NEW_TOKENS,
                 temperature: float = TEMPERATURE, percobaan: int = 3) -> HasilLLM:
        i = 0
        while True:
            payload = self._payload(model_id, messages, max_new_tokens, temperature, model_id in self._tanpa_reasoning)
            try:
                resp = self.http.post(self.base_url + "/chat/completions", headers=self._headers, json=payload,
                                      timeout=self.timeout)
            except Exception as e:  # noqa: BLE001  (jaringan)
                if self._transien_exc(e) and i < percobaan - 1:
                    i += 1
                    self._sleep(2.0 * i)
                    continue
                raise self._dari_exc(e) from None
            if resp.status_code == 200:
                return self._ekstrak(resp)
            # Model menolak parameter penalaran: lepas, ulangi sekali.
            if (resp.status_code == 400 and model_id not in self._tanpa_reasoning and pengaturan_reasoning(model_id)
                    and "reasoning" in _pesan_api(resp).lower()):
                log.warning("model %s menolak parameter penalaran; dikirim tanpa pengaturan itu", model_id)
                self._tanpa_reasoning.add(model_id)
                continue
            if resp.status_code >= 500 and i < percobaan - 1:
                i += 1
                log.warning("retry %d/%d untuk %s", i, percobaan, model_id)
                self._sleep(2.0 * i)
                continue
            raise self._dari_http(resp)
