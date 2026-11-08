"""Uji gaya jawaban: prompt ringkas, pemilihan gaya, dan jaminan bahwa gaya naratif (prompt penelitian) tidak berubah."""
import os
import unittest
from unittest import mock

from src import config
from src.config import GAYA_VALID, PESAN_KONTEKS_KOSONG
from src.generator import prompt_template as pt
from src.generator.gaya_jawaban import (
    LABEL_GAYA,
    SYSTEM_PROMPT_RINGKAS,
    bangun_messages_gaya,
    jumlah_kata,
    periksa_format_ringkas,
)
from src.generator.rag_service import hasilkan_jawaban

try:
    from test_chat_service import LLM, proses
    from test_hybrid_retriever import KGTiruan, buat_retriever
except ImportError:
    from tests.test_chat_service import LLM, proses
    from tests.test_hybrid_retriever import KGTiruan, buat_retriever

KONTEKS = "[Konteks 1]\nNode ID: 1\n[ULASAN / JAWABAN]: Pasal 3 UU 31/1999 mengatur ..."


class TestPromptRingkas(unittest.TestCase):
    def test_dua_pesan_tanpa_few_shot(self):
        m = bangun_messages_gaya("Apa sanksinya?", KONTEKS, "ringkas")
        self.assertEqual([x["role"] for x in m], ["system", "user"])
        self.assertEqual(m[0]["content"], SYSTEM_PROMPT_RINGKAS)
        self.assertEqual(m[1]["content"], pt.user_prompt("Apa sanksinya?", KONTEKS))
        self.assertFalse(any(x["content"] == f["content"] for x in m for f in pt.FEW_SHOT_HYBRID))

    def test_aturan_kejujuran_dipertahankan(self):
        sp = SYSTEM_PROMPT_RINGKAS
        self.assertIn("HANYA berdasarkan konteks", sp)
        self.assertIn("Jangan menambahkan pasal, peraturan, putusan, angka, atau penjelasan yang tidak ada dalam konteks", sp)
        self.assertIn(PESAN_KONTEKS_KOSONG, sp)  # pesan baku konteks kosong sama dengan yang dipakai rag_service
        self.assertIn("[PASAL TERKAIT]", sp)
        self.assertIn("[ULASAN / JAWABAN]", sp)

    def test_tidak_lagi_menyuruh_naratif(self):
        sp = SYSTEM_PROMPT_RINGKAS.lower()
        for kata in ("naratif", "komprehensif", "paragraf", "hindari bullet"):
            self.assertNotIn(kata, sp)
        for kata in ("singkat", "langsung", "**jawaban singkat:**", "**dasar hukum:**"):
            self.assertIn(kata, sp)

    def test_gaya_naratif_identik_dengan_prompt_penelitian(self):
        self.assertEqual(bangun_messages_gaya("Q?", KONTEKS, "naratif"), pt.bangun_messages("Q?", KONTEKS))
        self.assertEqual(bangun_messages_gaya("Q?", KONTEKS), pt.bangun_messages("Q?", KONTEKS))  # default = naratif

    def test_gaya_tidak_dikenal(self):
        with self.assertRaises(ValueError):
            bangun_messages_gaya("Q?", KONTEKS, "acak")

    def test_label_lengkap(self):
        self.assertEqual(set(LABEL_GAYA), set(GAYA_VALID))


class TestFormatRingkas(unittest.TestCase):
    BAIK = "**Jawaban singkat:** Ya.\n\n**Dasar hukum:**\n- Pasal 3 UU 31/1999 — sanksi.\n\n**Hal penting:**\n- Syarat."

    def test_sesuai(self):
        self.assertEqual(periksa_format_ringkas(self.BAIK), [])
        self.assertEqual(periksa_format_ringkas("  \n" + self.BAIK), [])

    def test_pembuka_atau_bagian_hilang_ditandai(self):
        self.assertTrue(periksa_format_ringkas("**Analisis Yuridis**\n\n" + self.BAIK))
        self.assertEqual(len(periksa_format_ringkas("**Jawaban singkat:** Ya.")), 1)
        self.assertEqual(len(periksa_format_ringkas("Paragraf naratif biasa tanpa format.")), 3)

    def test_jawaban_baku_konteks_kosong_sah(self):
        self.assertEqual(periksa_format_ringkas(PESAN_KONTEKS_KOSONG), [])

    def test_jumlah_kata(self):
        self.assertEqual((jumlah_kata("satu dua  tiga\ncuatro"), jumlah_kata(""), jumlah_kata(None)), (4, 0, 0))


class TestKonfigurasiGaya(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(config, "_DOTENV_SUDAH_DIMUAT", True),
                  mock.patch.object(config, "_dari_streamlit_secrets", lambda n: None)):
            p.start()
            self.addCleanup(p.stop)

    def test_default_override_dan_tidak_valid(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.get_gaya_default(), "ringkas")
        with mock.patch.dict(os.environ, {"GAYA_JAWABAN": "NARATIF"}, clear=True):
            self.assertEqual(config.get_gaya_default(), "naratif")
        with mock.patch.dict(os.environ, {"GAYA_JAWABAN": "puitis"}, clear=True):
            with self.assertRaises(config.ConfigError):
                config.get_gaya_default()


class TestGayaMengalirKeModel(unittest.TestCase):
    def test_rag_service_default_naratif_dan_gaya_terbawa(self):
        items = buat_retriever(KGTiruan()).cari_dengan_info("q", 5).items
        for gaya, n_pesan in (("naratif", 4), ("ringkas", 2)):
            llm = LLM()
            j = hasilkan_jawaban("Apa sanksinya?", items, "m", llm, "disempurnakan", gaya=gaya)
            self.assertEqual((len(llm.messages[0]), j.gaya), (n_pesan, gaya))
        llm = LLM()
        self.assertEqual(hasilkan_jawaban("q?", items, "m", llm, "disempurnakan").gaya, "naratif")
        self.assertEqual(len(llm.messages[0]), 4)

    def test_konteks_kosong_tetap_membawa_gaya(self):
        j = hasilkan_jawaban("q?", [], "m", LLM(), "disempurnakan", gaya="ringkas")
        self.assertEqual((j.jawaban, j.gaya), (PESAN_KONTEKS_KOSONG, "ringkas"))

    def test_chat_service_memakai_default_konfigurasi_dan_dapat_dioverride(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GAYA_JAWABAN", None)
            llm = LLM()
            h = proses(llm=llm)
            self.assertEqual((h.gaya, len(llm.messages[0])), ("ringkas", 2))
            llm2 = LLM()
            h2 = proses(llm=llm2, gaya="naratif")
            self.assertEqual((h2.gaya, len(llm2.messages[0])), ("naratif", 4))

    def test_gaya_tidak_valid_ditolak_sebelum_retrieval(self):
        class TidakBoleh:
            def cari_dengan_info(self, *a):
                raise AssertionError("retrieval tidak boleh dipanggil")

        h = proses(retriever=TidakBoleh(), gaya="acak")
        self.assertEqual((h.jenis_galat, h.galat), ("validasi", "Gaya jawaban tidak dikenal."))
        with mock.patch.dict(os.environ, {"GAYA_JAWABAN": "puitis"}):
            h2 = proses(retriever=TidakBoleh())
            self.assertEqual(h2.jenis_galat, "konfigurasi")


if __name__ == "__main__":
    unittest.main()
