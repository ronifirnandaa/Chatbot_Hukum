"""Uji router penyedia, pabrik LLM, registry model, dan konfigurasi multi-penyedia (offline)."""
import os
import unittest
from unittest import mock

from src import config
from src.generator.llm_client import GenerasiError, HasilLLM, ModelTidakTersediaError
from src.generator.llm_factory import buat_llm_dari_konfigurasi
from src.generator.llm_router import LLMRouter
from src.generator.model_registry import ModelInfo, model_terkonfigurasi, model_tersedia
from src.generator.rag_service import hasilkan_jawaban

try:
    from test_groq_client import FakeHTTP, Resp
    from test_hybrid_retriever import KGTiruan, buat_retriever
except ImportError:
    from tests.test_groq_client import FakeHTTP, Resp
    from tests.test_hybrid_retriever import KGTiruan, buat_retriever

MSG = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class Penyedia:
    def __init__(self, ids=(), galat=None, tokens=(None, None)):
        self.ids, self.galat, self.tokens, self.calls = set(ids), galat, tokens, []

    def daftar_model(self):
        if self.galat:
            raise self.galat
        return set(self.ids)

    def generate(self, model_id, messages, **kw):
        self.calls.append((model_id, messages, kw))
        return HasilLLM("jawaban dari " + model_id, True, "stop", *self.tokens)


class TestRouter(unittest.TestCase):
    def test_daftar_model_berawalan_penyedia(self):
        r = LLMRouter({"groq": Penyedia({"qwen/qwen3.8-27b"}), "google": Penyedia({"gemma-4-26b-a4b-it"})})
        self.assertEqual(r.daftar_model(), {"groq:qwen/qwen3.8-27b", "google:gemma-4-26b-a4b-it"})
        self.assertEqual(r.peringatan_penyedia(), {})

    def test_satu_penyedia_gagal_tidak_menyembunyikan_yang_lain(self):
        r = LLMRouter({"groq": Penyedia({"qwen/qwen3.8-27b"}), "google": Penyedia(galat=GenerasiError("Google tidak terjangkau.", "d"))})
        self.assertEqual(r.daftar_model(), {"groq:qwen/qwen3.8-27b"})
        self.assertEqual(r.peringatan_penyedia(), {"google": "Google tidak terjangkau."})

    def test_peringatan_pulih_pada_pemeriksaan_berikutnya(self):
        google = Penyedia(galat=GenerasiError("gagal"))
        r = LLMRouter({"groq": Penyedia({"a"}), "google": google})
        r.daftar_model()
        self.assertTrue(r.peringatan_penyedia())
        google.galat, google.ids = None, {"g"}
        r.daftar_model()
        self.assertEqual(r.peringatan_penyedia(), {})

    def test_semua_penyedia_gagal_melempar_galat(self):
        r = LLMRouter({"groq": Penyedia(galat=GenerasiError("Groq mati.", "dg")), "google": Penyedia(galat=GenerasiError("Google mati.", "dgg"))})
        with self.assertRaises(GenerasiError) as cm:
            r.daftar_model()
        self.assertEqual(str(cm.exception), "Groq mati.")
        self.assertIn("dgg", cm.exception.detail_teknis)

    def test_peringatan_dilewati_digabung(self):
        r = LLMRouter({"groq": Penyedia({"a"})}, penyedia_dilewati={"google": "paket belum terpasang"})
        r.daftar_model()
        self.assertEqual(r.peringatan_penyedia(), {"google": "paket belum terpasang"})

    def test_generate_diteruskan_ke_penyedia_yang_tepat(self):
        groq, google = Penyedia(), Penyedia()
        r = LLMRouter({"groq": groq, "google": google})
        h = r.generate("groq:qwen/qwen3.8-27b", MSG, max_new_tokens=300, temperature=0.0)
        self.assertEqual(h.teks, "jawaban dari qwen/qwen3.8-27b")
        self.assertEqual(groq.calls, [("qwen/qwen3.8-27b", MSG, {"max_new_tokens": 300, "temperature": 0.0})])
        self.assertEqual(google.calls, [])
        r.generate("google:gemma-4-26b-a4b-it", MSG)
        self.assertEqual(len(google.calls), 1)

    def test_id_dengan_titik_dua_tidak_terpotong(self):
        google = Penyedia()
        LLMRouter({"google": google}).generate("google:model:v2", MSG)
        self.assertEqual(google.calls[0][0], "model:v2")

    def test_kunci_tak_dikenal_ditolak_tanpa_memanggil_penyedia(self):
        groq = Penyedia()
        r = LLMRouter({"groq": groq})
        for kunci in ("google:gemma", "tanpa-awalan", "groq:", "", "xyz:abc"):
            with self.assertRaises(ModelTidakTersediaError):
                r.generate(kunci, MSG)
        self.assertEqual(groq.calls, [])

    def test_alur_penuh_rag_ke_router_membawa_model_dan_token(self):
        groq = Penyedia(tokens=(4321, 612))
        items = buat_retriever(KGTiruan()).cari_dengan_info("q", 5).items
        j = hasilkan_jawaban("Apa sanksinya?", items, "groq:openai/gpt-oss-20b", LLMRouter({"groq": groq}), "disempurnakan")
        self.assertEqual(j.jawaban, "jawaban dari openai/gpt-oss-20b")
        self.assertEqual((j.model_id, j.token_masuk, j.token_keluar), ("groq:openai/gpt-oss-20b", 4321, 612))
        self.assertEqual(groq.calls[0][0], "openai/gpt-oss-20b")


class TestPabrik(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(config, "_DOTENV_SUDAH_DIMUAT", True),
                  mock.patch.object(config, "_dari_streamlit_secrets", lambda n: None)):
            p.start()
            self.addCleanup(p.stop)

    def test_tanpa_kunci_apa_pun(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(config.ConfigError) as cm:
                buat_llm_dari_konfigurasi()
        self.assertIn("GROQ_API_KEY", str(cm.exception))
        self.assertIn("GOOGLE_API_KEY", str(cm.exception))

    def test_hanya_groq(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_abcdefghij"}, clear=True):
            r = buat_llm_dari_konfigurasi(http=FakeHTTP())
        self.assertEqual(list(r.penyedia), ["groq"])
        self.assertEqual(r.penyedia_dilewati, {})

    def test_groq_dan_google_bila_dua_kunci_ada(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_abcdefghij", "GOOGLE_API_KEY": "AIza_abcdefghij"}, clear=True), \
                mock.patch("src.generator.llm_factory.buat_klien_google", return_value=object()):
            r = buat_llm_dari_konfigurasi(http=FakeHTTP())
        self.assertEqual(list(r.penyedia), ["groq", "google"])

    def test_paket_google_tidak_ada_tidak_mematikan_groq(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_abcdefghij", "GOOGLE_API_KEY": "AIza_abcdefghij"}, clear=True), \
                mock.patch("src.generator.llm_factory.buat_klien_google", side_effect=ImportError("google.genai")):
            r = buat_llm_dari_konfigurasi(http=FakeHTTP())
        self.assertEqual(list(r.penyedia), ["groq"])
        self.assertIn("google", r.peringatan_penyedia())

    def test_paket_google_tidak_ada_dan_tanpa_groq_adalah_galat(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "AIza_abcdefghij"}, clear=True), \
                mock.patch("src.generator.llm_factory.buat_klien_google", side_effect=ImportError("google.genai")):
            with self.assertRaises(ImportError):
                buat_llm_dari_konfigurasi()

    def test_klien_groq_mengirim_ke_http_yang_disuntikkan(self):
        http = FakeHTTP(get=Resp(200, {"data": [{"id": "qwen/qwen3.8-27b"}]}))
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_abcdefghij"}, clear=True):
            r = buat_llm_dari_konfigurasi(http=http)
        self.assertEqual(r.daftar_model(), {"groq:qwen/qwen3.8-27b"})


class TestKonfigurasiMultiPenyedia(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(config, "_DOTENV_SUDAH_DIMUAT", True),
                  mock.patch.object(config, "_dari_streamlit_secrets", lambda n: None)):
            p.start()
            self.addCleanup(p.stop)

    def test_penyedia_aktif_berurutan_groq_dulu(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.get_penyedia_aktif(), [])
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "x" * 10, "GROQ_API_KEY": "y" * 10}, clear=True):
            self.assertEqual(config.get_penyedia_aktif(), ["groq", "google"])

    def test_model_default_mengikuti_penyedia_aktif(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.get_model_ids(), [])
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "y" * 10}, clear=True):
            self.assertEqual(config.get_model_ids(), ["groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b"])
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "y" * 10, "GOOGLE_API_KEY": "x" * 10}, clear=True):
            ids = config.get_model_ids()
        self.assertEqual(ids[0], "groq:qwen/qwen3.8-27b")
        self.assertEqual(ids[-2:], ["google:gemma-4-26b-a4b-it", "google:gemma-4-31b-it"])

    def test_override_dengan_awalan_tanpa_awalan_dan_duplikat(self):
        env = {"GENERATOR_MODELS": "groq:openai/gpt-oss-20b, gemma-4-31b-it ,google:gemma-4-31b-it,, groq:openai/gpt-oss-20b"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(config.get_model_ids(), ["groq:openai/gpt-oss-20b", "google:gemma-4-31b-it"])

    def test_pecah_kunci_model(self):
        self.assertEqual(config.pecah_kunci_model("groq:qwen/qwen3.8-27b"), ("groq", "qwen/qwen3.8-27b"))
        self.assertEqual(config.pecah_kunci_model("google:gemma-4-31b-it"), ("google", "gemma-4-31b-it"))
        self.assertEqual(config.pecah_kunci_model("gemma-4-31b-it"), ("google", "gemma-4-31b-it"))
        self.assertEqual(config.pecah_kunci_model("qwen3.5:4b"), ("google", "qwen3.5:4b"))  # bukan penyedia valid

    def test_kunci_groq_wajib_dan_pesan_hanya_nama(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(config.ConfigError) as cm:
                config.get_groq_api_key()
            self.assertIn("GROQ_API_KEY", str(cm.exception))
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_rahasia12345"}, clear=True):
            self.assertEqual(config.get_groq_api_key(), "gsk_rahasia12345")


class TestRegistryMultiPenyedia(unittest.TestCase):
    def test_kunci_dan_label(self):
        m = model_terkonfigurasi(["groq:qwen/qwen3.8-27b", "groq:model/baru", "google:gemma-4-31b-it", "gemma-4-26b-a4b-it"])
        self.assertEqual([x.kunci for x in m], ["groq:qwen/qwen3.8-27b", "groq:model/baru", "google:gemma-4-31b-it", "google:gemma-4-26b-a4b-it"])
        self.assertEqual(m[0].label_lengkap, "Qwen 3.8 27B (Groq)")
        self.assertEqual(m[1].label, "model/baru (Groq)")  # model di luar daftar label tetap tampil
        self.assertEqual(ModelInfo("m", "M", False).kunci, "google:m")  # default penyedia lama

    def test_dua_penyedia_dengan_id_sama_tidak_bertabrakan(self):
        m = model_terkonfigurasi(["groq:sama", "google:sama"])
        self.assertEqual(len({x.kunci for x in m}), 2)

    def test_hanya_model_terdaftar_di_penyedianya(self):
        r = LLMRouter({"groq": Penyedia({"qwen/qwen3.8-27b"}), "google": Penyedia({"gemma-4-31b-it"})})
        ids = ["groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-20b", "google:gemma-4-31b-it", "google:qwen/qwen3.8-27b"]
        self.assertEqual([m.kunci for m in model_tersedia(r, model_terkonfigurasi(ids))],
                         ["groq:qwen/qwen3.8-27b", "google:gemma-4-31b-it"])


if __name__ == "__main__":
    unittest.main()
