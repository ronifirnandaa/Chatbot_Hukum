"""Koneksi Neo4j untuk retriever.

Sumber dari notebook retriever:
* pembuatan driver (+ fallback bila driver lama tidak mengenal `notifications_disabled_classifications`),
* `panggil_dengan_retry` (Neo4j Aura Free kadang memutus koneksi / membatasi laju),
* pemeriksaan isi database (indeks vektor, jumlah LegalQA/Law/REFERENCES).

Tambahan untuk website:
* pesan error yang aman ditampilkan ke pengguna (tanpa URI, username, atau password),
* pembandingan dengan tanda tangan KG final.

Impor paket `neo4j` dilakukan saat dibutuhkan, sehingga modul ini dapat diuji tanpa database.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from urllib.parse import urlparse

from src.config import (
    INDEKS_VEKTOR_WAJIB,
    TANDA_TANGAN_KG_FINAL,
    Neo4jSettings,
)

log = logging.getLogger(__name__)

# Sama dengan notebook, ditambah pencocokan nama kelas pengecualian.
TANDA_TRANSIEN = ("ServiceUnavailable", "SessionExpired", "TransientError", "Connection", "Timeout", "TooManyRequests", "429")


class KoneksiNeo4jError(RuntimeError):
    """Galat koneksi/query. `str(e)` aman ditampilkan ke pengguna; `detail_teknis` hanya untuk log."""

    def __init__(self, pesan_pengguna: str, detail_teknis: str = ""):
        super().__init__(pesan_pengguna)
        self.detail_teknis = detail_teknis


class KGTidakSesuaiError(KoneksiNeo4jError):
    """Database tersambung tetapi isinya bukan KG yang diharapkan."""


# --------------------------------------------------------------------------------------
# Utilitas
# --------------------------------------------------------------------------------------
def _bersihkan(teks, settings: Neo4jSettings, maks: int = 300) -> str:
    """Hapus password / URI / host / username dari teks sebelum dicatat."""
    t = str(teks)
    host = urlparse(settings.uri).hostname or ""
    for rahasia, pengganti in (
        (settings.password, "***"),
        (settings.uri, "<uri>"),
        (host, "<host>"),
        (settings.username, "<user>"),
    ):
        if rahasia and len(rahasia) >= 4:
            t = t.replace(rahasia, pengganti)
    return t[:maks]


def _transien(e: Exception) -> bool:
    cek = type(e).__name__ + " " + str(e)
    return any(t in cek for t in TANDA_TRANSIEN)


def panggil_dengan_retry(fn, attempts: int = 3, tunggu: float = 2.0, sleep=time.sleep):
    """Jalankan `fn`; ulangi untuk galat sementara (koneksi putus, batas laju). Galat lain langsung dilempar."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not _transien(e) or i == attempts - 1:
                raise
            log.warning("retry %d/%d setelah galat sementara: %s", i + 1, attempts, type(e).__name__)
            sleep(tunggu * (i + 1))
    return None


def _pesan_pengguna(e: Exception) -> str:
    nama = type(e).__name__
    if nama == "AuthError":
        return "Database menolak kredensial. Periksa NEO4J_USERNAME dan NEO4J_PASSWORD di Secrets."
    if _transien(e) or nama in ("ConfigurationError", "DriverError"):
        return (
            "Tidak dapat menjangkau database. Instance Neo4j mungkin sedang tidak aktif/dijeda, "
            "atau jaringan bermasalah. Coba lagi beberapa saat."
        )
    return "Terjadi kesalahan saat mengakses database."


def _driver_default(uri: str, auth: tuple):
    from neo4j import GraphDatabase  # impor malas

    try:
        return GraphDatabase.driver(uri, auth=auth, notifications_disabled_classifications=["DEPRECATION"])
    except Exception:  # driver versi lama tidak mengenal argumen di atas
        return GraphDatabase.driver(uri, auth=auth)


# --------------------------------------------------------------------------------------
# Koneksi
# --------------------------------------------------------------------------------------
class Neo4jConnection:
    """Pembungkus driver Neo4j. Dipakai bersama oleh retriever (satu koneksi untuk seluruh aplikasi)."""

    def __init__(self, settings: Neo4jSettings, driver_factory=None, sleep=time.sleep):
        self.settings = settings
        self._sleep = sleep
        factory = driver_factory or _driver_default
        try:
            self.driver = factory(settings.uri, (settings.username, settings.password))
        except Exception as e:  # noqa: BLE001
            raise KoneksiNeo4jError(_pesan_pengguna(e), _bersihkan("%s: %s" % (type(e).__name__, e), settings)) from None
        try:
            panggil_dengan_retry(self.driver.verify_connectivity, sleep=sleep)
        except Exception as e:  # noqa: BLE001
            self.close()
            raise KoneksiNeo4jError(_pesan_pengguna(e), _bersihkan("%s: %s" % (type(e).__name__, e), settings)) from None

    # ---------------------------------------------------------------- sesi & query
    @contextmanager
    def session(self):
        s = self.driver.session(database=self.settings.database)
        try:
            yield s
        finally:
            s.close()

    def jalankan(self, query: str, **params) -> list[dict]:
        """Jalankan satu query dengan retry; kembalikan list[dict]."""

        def _run():
            with self.session() as s:
                return s.run(query, **params).data()

        try:
            return panggil_dengan_retry(_run, sleep=self._sleep) or []
        except Exception as e:  # noqa: BLE001
            raise KoneksiNeo4jError(
                _pesan_pengguna(e), _bersihkan("%s: %s" % (type(e).__name__, e), self.settings)
            ) from None

    def close(self) -> None:
        drv = getattr(self, "driver", None)
        if drv is not None:
            try:
                drv.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    # ---------------------------------------------------------------- verifikasi KG
    def cek_kg(self) -> dict:
        """Periksa isi database (query sama dengan sel koneksi notebook retriever).

        Return dict: indeks, n_legalqa, n_ulasan, n_law, n_references, galat[], peringatan[].
        `galat` = database tidak bisa dipakai retriever. `peringatan` = bisa dipakai, tetapi berbeda dari KG evaluasi.
        """
        indeks = sorted(r["name"] for r in self.jalankan("SHOW VECTOR INDEXES YIELD name RETURN name"))
        jml = self.jalankan(
            "MATCH (q:LegalQA) WITH count(q) AS c_q, count(q.ulasan) AS c_ulasan "
            "MATCH (l:Law) RETURN c_q, c_ulasan, count(l) AS c_law"
        )
        jml = jml[0] if jml else {"c_q": 0, "c_ulasan": 0, "c_law": 0}
        ref = self.jalankan("MATCH (:LegalQA)-[:REFERENCES]->(:Law) RETURN count(*) AS c")
        n_ref = ref[0]["c"] if ref else 0

        hasil = {
            "indeks": indeks,
            "n_legalqa": jml["c_q"],
            "n_ulasan": jml["c_ulasan"],
            "n_law": jml["c_law"],
            "n_references": n_ref,
            "galat": [],
            "peringatan": [],
        }
        if not hasil["n_legalqa"]:
            hasil["galat"].append("Database tidak memiliki node :LegalQA (mungkin menunjuk instance/database lain).")
        for nama in INDEKS_VEKTOR_WAJIB:
            if nama not in indeks:
                hasil["galat"].append("Indeks vektor '%s' tidak ada." % nama)
        if hasil["n_legalqa"] and hasil["n_ulasan"] != hasil["n_legalqa"]:
            hasil["galat"].append(
                "Sebagian LegalQA tidak punya ulasan (%d dari %d)." % (hasil["n_ulasan"], hasil["n_legalqa"])
            )
        if not n_ref:
            hasil["peringatan"].append("Tidak ada relasi REFERENCES: sumber hukum (pasal) tidak akan tersedia.")

        sekarang = {"LegalQA": hasil["n_legalqa"], "Law": hasil["n_law"], "REFERENCES": n_ref}
        beda = {k: (sekarang[k], v) for k, v in TANDA_TANGAN_KG_FINAL.items() if sekarang[k] != v}
        if beda:
            hasil["peringatan"].append(
                "Isi database berbeda dari KG yang dievaluasi di notebook (sekarang, seharusnya): %s. "
                "Hasil retrieval mungkin tidak sebanding dengan notebook." % beda
            )
        return hasil

    def pastikan_kg_valid(self) -> dict:
        """Seperti cek_kg(), tetapi melempar KGTidakSesuaiError bila ada `galat`."""
        h = self.cek_kg()
        if h["galat"]:
            raise KGTidakSesuaiError("Database bukan KG yang diharapkan: " + " ".join(h["galat"]))
        return h
