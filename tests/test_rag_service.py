"""Uji integrasi retriever -> konteks -> prompt -> generator (offline, KG dan LLM tiruan)."""
import unittest

from src.config import PESAN_KONTEKS_KOSONG
from src.generator import prompt_template as pt
from src.generator.llm_client import GenerasiError, HasilLLM
from src.generator.rag_service import hasilkan_jawaban

try:
    from test_hybrid_retriever import KGTiruan, buat_retriever
except ImportError:  # dijalankan sebagai tests.test_rag_service
    from tests.test_hybrid_retriever import KGTiruan, buat_retriever


class LLMTiruan:
    def __init__(self, hasil=None, galat=None):
        self.hasil = hasil or HasilLLM("Jawaban tiruan.", True, "STOP")
        self.galat, self.calls = galat, []

    def generate(self, model_id, messages, max_new_tokens, temperature):
        self.calls.append({"model": model_id, "messages": messages, "max": max_new_tokens, "temp": temperature})
        if self.galat:
            raise self.galat
        return self.hasil


class TestRagService(unittest.TestCase):
    def test_konteks_kosong_tanpa_memanggil_llm(self):
        llm = LLMTiruan()
        j = hasilkan_jawaban("Apa itu hukum?", [], "gemma-4-31b-it", llm, "disempurnakan")
        self.assertEqual(j.jawaban, PESAN_KONTEKS_KOSONG)
        self.assertFalse(j.memanggil_llm)
        self.assertEqual(llm.calls, [])
        self.assertIn("tidak tersedia dalam basis data saya", pt.SYSTEM_PROMPT_HYBRID)  # pesan baku memang dari prompt

    def test_model_terpilih_dipakai_dan_pesan_benar(self):
        items = buat_retriever(KGTiruan()).cari_dengan_info("pertanyaan", 5).items
        llm = LLMTiruan()
        j = hasilkan_jawaban("  Apa sanksinya?  ", items, "gemma-4-26b-a4b-it", llm, "disempurnakan")
        c = llm.calls[0]
        self.assertEqual(c["model"], "gemma-4-26b-a4b-it")
        self.assertEqual((c["max"], c["temp"]), (1024, 0.0))
        self.assertEqual(c["messages"], pt.bangun_messages("Apa sanksinya?", j.konteks.teks))
        self.assertEqual(j.model_id, "gemma-4-26b-a4b-it")
        self.assertEqual(j.jawaban, "Jawaban tiruan.")
        self.assertTrue(j.memanggil_llm and j.selesai_normal)

    def test_keluaran_retriever_masuk_prompt_dengan_label_yang_diharapkan_prompt(self):
        items = buat_retriever(KGTiruan()).cari_dengan_info("pertanyaan", 5).items
        j = hasilkan_jawaban("q", items, "m", LLMTiruan(), "disempurnakan")
        self.assertTrue(j.konteks.teks.startswith("[Konteks 1]\nNode ID: "))
        self.assertIn("[ULASAN / JAWABAN]: ", j.konteks.teks)
        self.assertIn("[PASAL TERKAIT]:\n- Pasal 48 (KUHP) [law_1]", j.konteks.teks)
        # label yang disebut system prompt harus ada di konteks hasil retriever
        for label in ("[ULASAN / JAWABAN]", "[PASAL TERKAIT]"):
            self.assertIn(label, pt.SYSTEM_PROMPT_HYBRID)
            self.assertIn(label, j.konteks.teks)

    def test_kedua_mode_menghasilkan_prompt_valid(self):
        items = buat_retriever(KGTiruan()).cari_dengan_info("pertanyaan", 5).items
        for mode in ("disempurnakan", "eksperimen"):
            with self.subTest(mode=mode):
                llm = LLMTiruan()
                j = hasilkan_jawaban("q", items, "m", llm, mode)
                self.assertEqual(j.konteks.mode, mode)
                self.assertGreaterEqual(j.konteks.n_dokumen_dipakai, 1)

    def test_validasi_input(self):
        for p in ("", "   ", None):
            with self.assertRaises(ValueError):
                hasilkan_jawaban(p, [], "m", LLMTiruan())
        with self.assertRaises(ValueError):
            hasilkan_jawaban("q", [], "", LLMTiruan())

    def test_galat_llm_diteruskan(self):
        items = buat_retriever(KGTiruan()).cari_dengan_info("pertanyaan", 5).items
        with self.assertRaises(GenerasiError):
            hasilkan_jawaban("q", items, "m", LLMTiruan(galat=GenerasiError("gagal")), "disempurnakan")

    def test_jawaban_terpotong_ditandai(self):
        items = buat_retriever(KGTiruan()).cari_dengan_info("pertanyaan", 5).items
        j = hasilkan_jawaban("q", items, "m", LLMTiruan(HasilLLM("terpo", False, "MAX_TOKENS")), "disempurnakan")
        self.assertFalse(j.selesai_normal)


if __name__ == "__main__":
    unittest.main()
