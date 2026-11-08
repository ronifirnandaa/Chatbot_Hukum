"""Uji asap (smoke) halaman Chatbot dengan Streamlit TIRUAN (offline).

Menjalankan skrip pages/1_Chatbot.py apa adanya dengan `streamlit` dan `src.app_resources` palsu. Menangkap salah
nama/salah alur/galat tak tertangani. TIDAK menguji tampilan nyata maupun kecocokan dengan API Streamlit asli:
itu harus dicek langsung dengan `streamlit run app.py`.
"""
import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from src.chat.chat_service import RetrieverTerkunci
from src.chat.pembatas_laju import PembatasLaju
from src.generator.llm_client import GenerasiError, HasilLLM, KuotaHabisError
from src.generator.model_registry import ModelInfo
from src.retriever.neo4j_connection import KoneksiNeo4jError

try:
    from test_chat_service import LLM
    from test_hybrid_retriever import KGTiruan, buat_retriever
except ImportError:
    from tests.test_chat_service import LLM
    from tests.test_hybrid_retriever import KGTiruan, buat_retriever

HALAMAN = Path(__file__).resolve().parent.parent / "pages" / "1_Chatbot.py"
TOMBOL_BERSIHKAN = "🗑️ Bersihkan percakapan"


class _Rerun(Exception):
    pass


class _Stop(Exception):
    pass


class SS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v


class FakeST(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SS()
        self.sidebar = mock.MagicMock()
        self.log, self.keys, self.input_value, self.tekan = [], [], None, set()
        self.pilihan_radio = None

    def _rec(self, nama, *a, **k):
        self.log.append((nama, a, k))
        return mock.MagicMock()

    def __getattr__(self, nama):
        if nama.startswith("__"):
            raise AttributeError(nama)
        return lambda *a, **k: self._rec(nama, *a, **k)

    def selectbox(self, label, options, format_func=str, **k):
        self.log.append(("selectbox", (label,), k))
        for o in options:
            format_func(o)
        return options[0] if options else None

    def radio(self, label, options, index=0, format_func=str, **k):
        self.log.append(("radio", (label,), dict(k, index=index)))
        for o in options:
            format_func(o)
        return self.pilihan_radio if self.pilihan_radio in options else options[index]

    def button(self, label, **k):
        self.log.append(("button", (label,), k))
        return label in self.tekan

    def chat_input(self, label, **k):
        # Sengaja TETAP mengembalikan teks walau disabled (Streamlit asli mengembalikan None) untuk menguji penjaga.
        self.log.append(("chat_input", (label,), k))
        v, self.input_value = self.input_value, None
        return v

    def text_area(self, label, **k):
        self.keys.append(k.get("key"))
        return self._rec("text_area", label, **k)

    def rerun(self):
        raise _Rerun

    def stop(self):
        raise _Stop

    # util uji
    def panggilan(self, nama):
        return [(a, k) for n, a, k in self.log if n == nama]

    def teks(self, nama):
        return [a[0] for a, _ in self.panggilan(nama) if a]


def sumber_default(**override):
    llm = override.pop("llm", None) or LLM()
    res = {
        "get_retriever": lambda: RetrieverTerkunci(buat_retriever(KGTiruan())),
        "get_llm": lambda: llm,
        "get_model_tersedia": lambda: [ModelInfo("m1", "Model Satu", False), ModelInfo("m2", "Model Dua", False)],
        "get_pembatas_global": lambda: PembatasLaju(100, 60),
    }
    res.update(override)
    return res, llm


def jalankan(fake, res, env=None):
    modul = types.ModuleType("src.app_resources")
    for k, v in res.items():
        setattr(modul, k, v)
    fake.keys = []
    with mock.patch.dict(sys.modules, {"streamlit": fake, "src.app_resources": modul}), \
            mock.patch.dict(os.environ, env or {}):
        try:
            runpy.run_path(str(HALAMAN), run_name="halaman_uji")
        except (_Rerun, _Stop):
            pass
    return fake


class TestHalamanChatbot(unittest.TestCase):
    def test_halaman_kosong_tanpa_input(self):
        fake = jalankan(FakeST(), sumber_default()[0])
        self.assertEqual(fake.session_state.riwayat, [])
        self.assertTrue(any("Contoh" in t for t in fake.teks("info")))
        self.assertEqual(fake.panggilan("chat_input")[0][1].get("disabled"), False)
        self.assertEqual(fake.teks("selectbox"), ["Model generator"])

    def test_pertanyaan_diproses_dengan_model_terpilih(self):
        res, llm = sumber_default()
        fake = FakeST()
        fake.input_value = "Apa sanksi korupsi?"
        jalankan(fake, res)
        r = fake.session_state.riwayat
        self.assertEqual([x["peran"] for x in r], ["user", "assistant"])
        hasil = r[1]["hasil"]
        self.assertIsNone(hasil.galat)
        self.assertEqual((llm.calls, hasil.model_id, hasil.model_label), (["google:m1"], "google:m1", "Model Satu"))
        self.assertIn(hasil.jawaban.jawaban, fake.teks("markdown"))
        labels = fake.teks("expander")
        self.assertTrue(any(l.startswith("Sumber hukum yang ditemukan") for l in labels))
        self.assertIn("Konteks yang dikirim ke model", labels)

    def test_riwayat_dirender_ulang_dengan_kunci_unik(self):
        res, _ = sumber_default()
        fake = FakeST()
        fake.input_value = "Pertanyaan pertama?"
        jalankan(fake, res)
        fake.input_value = "Pertanyaan kedua?"
        jalankan(fake, res)
        self.assertEqual(len(fake.session_state.riwayat), 4)
        self.assertEqual(fake.keys, ["ctx_1", "ctx_3"])  # satu per jawaban, tanpa duplikat
        fake.input_value = None
        jalankan(fake, res)  # render ulang riwayat saja
        self.assertEqual(fake.keys, ["ctx_1", "ctx_3"])
        self.assertEqual(len(set(fake.keys)), len(fake.keys))

    def test_galat_generasi_menampilkan_pesan_dan_tetap_menampilkan_sumber(self):
        res, _ = sumber_default(llm=LLM(galat=KuotaHabisError("Kuota layanan model habis.")))
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertIn("Kuota layanan model habis.", fake.teks("error"))
        self.assertTrue(any(l.startswith("Sumber hukum") for l in fake.teks("expander")))
        self.assertNotIn("Konteks yang dikirim ke model", fake.teks("expander"))

    def test_peringatan_kutipan_dan_dapat_dimatikan(self):
        res, _ = sumber_default(llm=LLM("Dijerat Pasal 777 UU 8/2010."))
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertTrue(any(t.startswith("Pemeriksaan otomatis") for t in fake.teks("warning")))
        fake2 = FakeST()
        fake2.input_value = "Apa sanksinya?"
        jalankan(fake2, res, env={"PERINGATAN_KUTIPAN": "false"})
        self.assertFalse(any(t.startswith("Pemeriksaan otomatis") for t in fake2.teks("warning")))

    def test_jawaban_terpotong_diberi_peringatan(self):
        res, _ = sumber_default(llm=LLM(normal=False))
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertTrue(any("terpotong" in t for t in fake.teks("warning")))

    def test_batas_per_sesi(self):
        res, llm = sumber_default()
        fake = FakeST()
        for i in range(2):
            fake.input_value = "Pertanyaan %d?" % i
            jalankan(fake, res, env={"RATE_SESI_MAKS": "1"})
        self.assertEqual(len(fake.session_state.riwayat), 2)  # pertanyaan kedua ditolak
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(any("Batas pertanyaan per sesi" in t for t in fake.teks("warning")))

    def test_batas_global(self):
        res, llm = sumber_default(get_pembatas_global=lambda: PembatasLaju(0, 60))
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertEqual((fake.session_state.riwayat, llm.calls), ([], []))
        self.assertTrue(any("banyak pertanyaan" in t for t in fake.teks("warning")))

    def test_sumber_daya_gagal_tidak_membocorkan_detail(self):
        def gagal_db():
            raise KoneksiNeo4jError("Tidak dapat menjangkau database.")

        def gagal_umum():
            raise OSError("C:\\rahasia\\model.bin tidak ada")

        for fn, harapan in ((gagal_db, "Tidak dapat menjangkau database."), (gagal_umum, "Sistem pencarian belum siap")):
            res, _ = sumber_default(get_retriever=fn)
            fake = FakeST()
            fake.input_value = "Apa sanksinya?"
            jalankan(fake, res)
            self.assertTrue(any(harapan in t for t in fake.teks("error")))
            self.assertFalse(any("rahasia" in t for t in fake.teks("error")))
            self.assertEqual(fake.session_state.riwayat, [])

    def test_tanpa_model_input_dinonaktifkan(self):
        for kasus in (lambda: [], mock.Mock(side_effect=GenerasiError("API tidak terjangkau."))):
            res, llm = sumber_default(get_model_tersedia=kasus)
            fake = FakeST()
            fake.input_value = "Apa sanksinya?"
            jalankan(fake, res)
            self.assertTrue(fake.panggilan("chat_input")[0][1]["disabled"])
            self.assertEqual(fake.panggilan("selectbox"), [])
            self.assertEqual(llm.calls, [])

    def test_gaya_ringkas_default_dan_dapat_diganti_di_sidebar(self):
        res, llm = sumber_default()
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res, env={"GAYA_JAWABAN": ""})
        radio = fake.panggilan("radio")[0]
        self.assertEqual((radio[0][0], radio[1]["index"]), ("Gaya jawaban", 0))  # ringkas = pilihan pertama
        self.assertEqual(len(llm.messages[0]), 2)
        self.assertEqual(fake.session_state.riwayat[1]["hasil"].gaya, "ringkas")
        self.assertTrue(any("gaya: Ringkas" in t for t in fake.teks("caption")))

        res2, llm2 = sumber_default()
        fake2 = FakeST()
        fake2.input_value = "Apa sanksinya?"
        fake2.pilihan_radio = "naratif"
        jalankan(fake2, res2)
        self.assertEqual(len(llm2.messages[0]), 4)
        self.assertEqual(fake2.session_state.riwayat[1]["hasil"].gaya, "naratif")
        self.assertTrue(any("gaya: Naratif" in t for t in fake2.teks("caption")))

    def test_gaya_default_dari_konfigurasi_mengatur_pilihan_awal(self):
        res, llm = sumber_default()
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res, env={"GAYA_JAWABAN": "naratif"})
        self.assertEqual(fake.panggilan("radio")[0][1]["index"], 1)
        self.assertEqual(len(llm.messages[0]), 4)

    def test_gaya_default_salah_ditampilkan_sebagai_galat_tanpa_crash(self):
        res, _ = sumber_default()
        fake = FakeST()
        jalankan(fake, res, env={"GAYA_JAWABAN": "puitis"})
        self.assertTrue(any("GAYA_JAWABAN" in t for t in fake.teks("error")))

    def test_peringatan_penyedia_ditampilkan_tanpa_menyembunyikan_model(self):
        class LLMDenganPeringatan(LLM):
            def peringatan_penyedia(self):
                return {"google": "Paket 'google-genai' belum terpasang."}

        res, _ = sumber_default(llm=LLMDenganPeringatan())
        fake = jalankan(FakeST(), res)
        self.assertTrue(any("Penyedia google" in t for t in fake.teks("warning")))
        self.assertEqual(fake.teks("selectbox"), ["Model generator"])  # dropdown tetap ada

    def test_token_ditampilkan_bila_tersedia(self):
        class LLMToken(LLM):
            def generate(self, model_id, messages, max_new_tokens, temperature):
                super().generate(model_id, messages, max_new_tokens, temperature)
                return HasilLLM("Jawaban.", True, "stop", 4321, 612)

        res, _ = sumber_default(llm=LLMToken())
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertTrue(any("token 4.321 masuk / 612 keluar" in t for t in fake.teks("caption")))

    def test_tanpa_token_keterangan_token_tidak_muncul(self):
        res, _ = sumber_default()
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertFalse(any("token " in t and "masuk" in t for t in fake.teks("caption")))

    def test_model_multi_penyedia_dikirim_dengan_kunci_penyedia(self):
        from src.generator.model_registry import ModelInfo as MI

        res, llm = sumber_default(get_model_tersedia=lambda: [MI("qwen/qwen3.8-27b", "Qwen 3.8 27B (Groq)", False, "groq")])
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertEqual(llm.calls, ["groq:qwen/qwen3.8-27b"])
        self.assertEqual(fake.session_state.riwayat[1]["hasil"].model_label, "Qwen 3.8 27B (Groq)")

    def test_tombol_bersihkan_mengosongkan_riwayat(self):
        res, _ = sumber_default()
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res)
        self.assertEqual(len(fake.session_state.riwayat), 2)
        fake.tekan = {TOMBOL_BERSIHKAN}
        jalankan(fake, res)
        self.assertEqual(fake.session_state.riwayat, [])

    def test_konfigurasi_batas_salah_dihentikan_dengan_pesan(self):
        res, _ = sumber_default()
        fake = FakeST()
        jalankan(fake, res, env={"RATE_SESI_MAKS": "banyak"})
        self.assertTrue(any("RATE_SESI_MAKS" in t for t in fake.teks("error")))

    def test_mode_potong_salah_dilaporkan_sebagai_galat_konfigurasi(self):
        res, llm = sumber_default()
        fake = FakeST()
        fake.input_value = "Apa sanksinya?"
        jalankan(fake, res, env={"MODE_POTONG": "ngawur"})
        self.assertTrue(any("MODE_POTONG" in t for t in fake.teks("error")))
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
