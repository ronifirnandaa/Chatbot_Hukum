"""Satu pertanyaan -> retrieval -> generasi -> pemeriksaan kutipan, dengan penanganan galat per tahap.

Tidak bergantung pada Streamlit sehingga dapat diuji mandiri. Setiap pertanyaan diproses MANDIRI (tanpa riwayat).
Bila generasi gagal, hasil retrieval tetap dikembalikan agar sumber hukum masih dapat ditampilkan.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from src.config import GAYA_VALID, K_DEFAULT, MAKS_PANJANG_PERTANYAAN, ConfigError, get_gaya_default
from src.generator.llm_client import GenerasiError, KuotaHabisError
from src.generator.pemeriksa_kutipan import HasilPeriksa, periksa_kutipan
from src.generator.rag_service import JawabanRAG, hasilkan_jawaban
from src.retriever.hybrid_retriever import HasilRetrieval, RetrievalError
from src.retriever.neo4j_connection import KoneksiNeo4jError

log = logging.getLogger(__name__)


@dataclass
class HasilChat:
    pertanyaan: str
    model_id: str
    model_label: str = ""
    gaya: str = ""
    retrieval: HasilRetrieval | None = None
    jawaban: JawabanRAG | None = None
    kutipan: HasilPeriksa | None = None
    galat: str | None = None
    jenis_galat: str | None = None  # 'validasi' | 'konfigurasi' | 'retrieval' | 'kuota' | 'generasi' | 'tak_terduga'
    waktu_total: float = 0.0


class RetrieverTerkunci:
    """Membungkus retriever bersama agar satu pencarian berjalan pada satu waktu (retriever menyimpan peringatan
    terakhir sebagai state instans, dan pemakaian encoder lintas thread tidak dijamin aman). Pencarian ~0,5 detik."""

    def __init__(self, retriever):
        self._retriever, self._kunci = retriever, threading.Lock()

    def cari_dengan_info(self, pertanyaan, top_k=K_DEFAULT):
        with self._kunci:
            return self._retriever.cari_dengan_info(pertanyaan, top_k)


def proses_pertanyaan(pertanyaan: str, model_id: str, retriever, llm, mode_potong: str | None = None,
                      top_k: int = K_DEFAULT, periksa: bool = True, model_label: str = "",
                      gaya: str | None = None) -> HasilChat:
    mulai = time.perf_counter()
    hasil = HasilChat(pertanyaan=(pertanyaan or "").strip() if isinstance(pertanyaan, str) else "",
                      model_id=model_id, model_label=model_label)

    def selesai(galat=None, jenis=None):
        hasil.galat, hasil.jenis_galat = galat, jenis
        hasil.waktu_total = time.perf_counter() - mulai
        return hasil

    try:
        hasil.gaya = gaya or get_gaya_default()
    except ConfigError as e:
        return selesai(str(e), "konfigurasi")
    if hasil.gaya not in GAYA_VALID:
        return selesai("Gaya jawaban tidak dikenal.", "validasi")
    if not hasil.pertanyaan:
        return selesai("Pertanyaan tidak boleh kosong.", "validasi")
    if len(hasil.pertanyaan) > MAKS_PANJANG_PERTANYAAN:
        return selesai("Pertanyaan terlalu panjang (maksimal %d karakter)." % MAKS_PANJANG_PERTANYAAN, "validasi")
    if not model_id:
        return selesai("Model generator belum dipilih.", "validasi")

    try:
        hasil.retrieval = retriever.cari_dengan_info(hasil.pertanyaan, top_k)
    except (KoneksiNeo4jError, RetrievalError, ValueError) as e:
        return selesai("Gagal mengambil konteks hukum: %s" % e, "retrieval")
    except Exception:  # noqa: BLE001
        log.exception("galat tak terduga pada retrieval")
        return selesai("Terjadi kesalahan tak terduga saat mengambil konteks hukum.", "tak_terduga")

    try:
        hasil.jawaban = hasilkan_jawaban(hasil.pertanyaan, hasil.retrieval.items, model_id, llm, mode_potong,
                                         gaya=hasil.gaya)
    except ConfigError as e:  # mis. MODE_POTONG tidak valid; pesan hanya menyebut nama variabel
        return selesai(str(e), "konfigurasi")
    except ValueError as e:  # mis. gaya tidak dikenal
        return selesai(str(e), "validasi")
    except KuotaHabisError as e:
        return selesai(str(e), "kuota")
    except GenerasiError as e:
        return selesai(str(e), "generasi")
    except Exception:  # noqa: BLE001
        log.exception("galat tak terduga pada generasi")
        return selesai("Terjadi kesalahan tak terduga saat menyusun jawaban.", "tak_terduga")

    if periksa and hasil.jawaban.memanggil_llm:
        hasil.kutipan = periksa_kutipan(hasil.jawaban.jawaban, hasil.jawaban.konteks.teks)
    return selesai()
