"""Uji layanan chat, pembatas laju, dan pengaman thread (offline)."""
import os
import threading
import time
import unittest
from unittest import mock

from src import config
from src.chat.chat_service import RetrieverTerkunci, proses_pertanyaan
from src.chat.pembatas_laju import PembatasLaju
from src.config import MAKS_PANJANG_PERTANYAAN, PESAN_KONTEKS_KOSONG
from src.generator.llm_client import GenerasiError, HasilLLM, KuotaHabisError
from src.retriever.hybrid_retriever import HasilRetrieval, RetrievalError
from src.retriever.neo4j_connection import KoneksiNeo4jError

try:
    from test_hybrid_retriever import KGTiruan, buat_retriever
except ImportError:
    from tests.test_hybrid_retriever import KGTiruan, buat_retriever


class LLM:
    def __init__(self, teks="Dijerat Pasal 3 UU 31/1999.", galat=None, normal=True):
        self.teks, self.galat, self.normal, self.calls, self.messages = teks, galat, normal, [], []

    def generate(self, model_id, messages, max_new_tokens, temperature):
        self.calls.append(model_id)
        self.messages.append(messages)
        if self.galat:
            raise self.galat
        return HasilLLM(self.teks, self.normal, "STOP" if self.normal else "MAX_TOKENS")


class RetrieverGalat:
    def __init__(self, galat):
        self.galat = galat

    def cari_dengan_info(self, p, k):
        raise self.galat


def proses(pertanyaan="Apa sanksinya?", retriever=None, llm=None, **kw):
    return proses_pertanyaan(pertanyaan, kw.pop("model_id", "m1"), retriever or buat_retriever(KGTiruan()),
                             llm or LLM(), "disempurnakan", **kw)


class TestProsesPertanyaan(unittest.TestCase):
    def test_alur_sukses(self):
        llm = LLM()
        h = proses(llm=llm, model_label="Model Satu")
        self.assertIsNone(h.galat)
        self.assertGreater(len(h.retrieval.items), 0)
        self.assertEqual(h.jawaban.jawaban, "Dijerat Pasal 3 UU 31/1999.")
        self.assertEqual((h.model_id, h.model_label, llm.calls), ("m1", "Model Satu", ["m1"]))
        self.assertGreater(h.waktu_total, 0)

    def test_validasi_input_tanpa_memanggil_apa_pun(self):
        llm = LLM()
        for p in ("", "   ", None):
            h = proses(p, llm=llm)
            self.assertEqual(h.jenis_galat, "validasi")
        self.assertEqual(proses("x" * (MAKS_PANJANG_PERTANYAAN + 1), llm=llm).jenis_galat, "validasi")
        self.assertEqual(proses(model_id="", llm=llm).jenis_galat, "validasi")
        self.assertEqual(llm.calls, [])

    def test_galat_retrieval_dibungkus_aman(self):
        for galat in (KoneksiNeo4jError("Tidak dapat menjangkau database."), RetrievalError("Pencarian gagal."), ValueError("x")):
            h = proses(retriever=RetrieverGalat(galat), llm=LLM())
            self.assertEqual(h.jenis_galat, "retrieval")
            self.assertTrue(h.galat.startswith("Gagal mengambil konteks hukum"))
            self.assertIsNone(h.jawaban)

    def test_galat_tak_terduga_tidak_membocorkan_detail(self):
        h = proses(retriever=RetrieverGalat(OSError("C:\\rahasia\\path password=abc")))
        self.assertEqual(h.jenis_galat, "tak_terduga")
        self.assertNotIn("rahasia", h.galat)
        h2 = proses(llm=LLM(galat=RuntimeError("kunci=AIza-rahasia")))
        self.assertEqual(h2.jenis_galat, "tak_terduga")
        self.assertNotIn("AIza", h2.galat)

    def test_galat_generasi_tetap_menyimpan_sumber_hukum(self):
        h = proses(llm=LLM(galat=GenerasiError("Layanan model sedang sibuk.")))
        self.assertEqual((h.jenis_galat, h.galat), ("generasi", "Layanan model sedang sibuk."))
        self.assertGreater(len(h.retrieval.items), 0)  # sumber tetap bisa ditampilkan
        self.assertIsNone(h.jawaban)

    def test_galat_konfigurasi_dibedakan_dan_aman(self):
        h = proses(llm=LLM(galat=config.ConfigError("MODE_POTONG tidak valid.")))
        self.assertEqual((h.jenis_galat, h.galat), ("konfigurasi", "MODE_POTONG tidak valid."))
        self.assertGreater(len(h.retrieval.items), 0)

    def test_kuota_dibedakan(self):
        self.assertEqual(proses(llm=LLM(galat=KuotaHabisError("kuota habis"))).jenis_galat, "kuota")

    def test_pemeriksaan_kutipan_menandai_karangan(self):
        h = proses(llm=LLM("Pasal 3 UU 31/1999 dan Pasal 777 serta UU 8/2010."))
        self.assertFalse(h.kutipan.semua_ditemukan)
        self.assertIn("777", h.kutipan.tidak_ditemukan["pasal"])
        self.assertIn("8/2010", h.kutipan.tidak_ditemukan["peraturan"])

    def test_pemeriksaan_dapat_dimatikan(self):
        self.assertIsNone(proses(periksa=False).kutipan)

    def test_konteks_kosong_jawaban_baku_tanpa_llm_dan_tanpa_pemeriksaan(self):
        llm = LLM()
        h = proses(retriever=buat_retriever(KGTiruan(vec={})), llm=llm)
        self.assertEqual(h.jawaban.jawaban, PESAN_KONTEKS_KOSONG)
        self.assertEqual(llm.calls, [])
        self.assertIsNone(h.kutipan)
        self.assertTrue(any("Tidak ada konteks" in p for p in h.retrieval.peringatan))

    def test_jawaban_terpotong_batas_panjang_terbawa(self):
        self.assertFalse(proses(llm=LLM(normal=False)).jawaban.selesai_normal)


class TestPembatasLaju(unittest.TestCase):
    def test_jendela_geser(self):
        t = {"now": 100.0}
        p = PembatasLaju(2, 60, jam=lambda: t["now"])
        self.assertEqual(p.coba(), (True, 0.0))
        t["now"] = 110
        self.assertEqual(p.coba(), (True, 0.0))
        ok, tunggu = p.coba()
        self.assertFalse(ok)
        self.assertAlmostEqual(tunggu, 50.0)  # slot tertua (t=100) bebas pada t=160
        t["now"] = 160
        self.assertEqual(p.coba(), (True, 0.0))

    def test_maks_nol_selalu_ditolak(self):
        ok, tunggu = PembatasLaju(0, 30).coba()
        self.assertEqual((ok, tunggu), (False, 30.0))

    def test_aman_lintas_thread(self):
        p = PembatasLaju(50, 60)
        hasil = []

        def kerja():
            for _ in range(20):
                hasil.append(p.coba()[0])

        ts = [threading.Thread(target=kerja) for _ in range(10)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(sum(hasil), 50)  # tepat 50 diizinkan dari 200 percobaan


class TestRetrieverTerkunci(unittest.TestCase):
    def test_tidak_ada_pencarian_yang_tumpang_tindih(self):
        class Lambat:
            aktif = maks_aktif = 0

            def cari_dengan_info(self, p, k):
                Lambat.aktif += 1
                Lambat.maks_aktif = max(Lambat.maks_aktif, Lambat.aktif)
                time.sleep(0.01)
                Lambat.aktif -= 1
                return HasilRetrieval(items=[], waktu_detik=0)

        r = RetrieverTerkunci(Lambat())
        ts = [threading.Thread(target=lambda: r.cari_dengan_info("q", 5)) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(Lambat.maks_aktif, 1)


class TestKonfigurasiChatbot(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(config, "_DOTENV_SUDAH_DIMUAT", True),
                  mock.patch.object(config, "_dari_streamlit_secrets", lambda n: None)):
            p.start()
            self.addCleanup(p.stop)

    def test_default_dan_override_batas_laju(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.get_batas_laju(), {"sesi": (5, 600), "global": (8, 60)})
        with mock.patch.dict(os.environ, {"RATE_SESI_MAKS": "2", "RATE_GLOBAL_MAKS": "0"}, clear=True):
            self.assertEqual(config.get_batas_laju(), {"sesi": (2, 600), "global": (0, 60)})
        with mock.patch.dict(os.environ, {"RATE_SESI_MAKS": "banyak"}, clear=True):
            with self.assertRaises(config.ConfigError):
                config.get_batas_laju()

    def test_peringatan_kutipan_default_aktif(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(config.get_peringatan_kutipan())
        for nilai in ("false", "FALSE", "0", "off"):
            with mock.patch.dict(os.environ, {"PERINGATAN_KUTIPAN": nilai}, clear=True):
                self.assertFalse(config.get_peringatan_kutipan())


if __name__ == "__main__":
    unittest.main()
