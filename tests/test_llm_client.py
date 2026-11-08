"""Uji adapter Gemini + registry model dengan SDK tiruan (offline)."""
import os
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from src import config
from src.generator import prompt_template as pt
from src.generator.llm_client import GenerasiError, GeminiLLM, KuotaHabisError, ModelTidakTersediaError
from src.generator.llm_router import LLMRouter
from src.generator.model_registry import ModelInfo, model_terkonfigurasi, model_tersedia

KUNCI = "AIzaSyKUNCI-RAHASIA-1234567890"


class ApiError(Exception):
    def __init__(self, code, message, status=""):
        super().__init__("%s %s. %s" % (code, status, message))
        self.code, self.message, self.status = code, message, status


def resp(teks="Jawaban.", finish="STOP", thought_teks=None, token_pikir=None):
    parts = []
    if thought_teks:
        parts.append(NS(text=thought_teks, thought=True))
    if teks is not None:
        parts.append(NS(text=teks, thought=None))
    return NS(candidates=[NS(content=NS(parts=parts), finish_reason=NS(name=finish))],
              usage_metadata=NS(thoughts_token_count=token_pikir))


class FakeModels:
    def __init__(self, perilaku, daftar=()):
        self.perilaku, self.calls, self._daftar = list(perilaku), [], daftar

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        x = self.perilaku.pop(0) if len(self.perilaku) > 1 else self.perilaku[0]
        if isinstance(x, Exception):
            raise x
        return x

    def list(self):
        if isinstance(self._daftar, Exception):
            raise self._daftar
        return iter(self._daftar)


def buat(perilaku, daftar=(), rahasia=(KUNCI,)):
    models = FakeModels(perilaku, daftar)
    return GeminiLLM(NS(models=models), sleep=lambda s: None, rahasia=rahasia), models


MSG = pt.bangun_messages("Apa sanksinya?", "[Konteks 1]\nNode ID: 1\nIsi")


class TestGenerate(unittest.TestCase):
    def test_pemetaan_pesan_dan_parameter(self):
        llm, m = buat([resp("Halo")])
        hasil = llm.generate("gemma-4-31b-it", MSG)
        c = m.calls[0]
        self.assertEqual(hasil.teks, "Halo")
        self.assertTrue(hasil.selesai_normal)
        self.assertEqual(c["model"], "gemma-4-31b-it")
        self.assertEqual(c["config"]["system_instruction"], pt.SYSTEM_PROMPT_HYBRID)
        self.assertEqual((c["config"]["temperature"], c["config"]["max_output_tokens"]), (0.0, 1024))
        self.assertEqual(c["config"]["thinking_config"], {"thinking_level": "minimal"})  # mode berpikir dimatikan
        self.assertEqual([x["role"] for x in c["contents"]], ["user", "model", "user"])  # system dipisah
        self.assertEqual(len(c["contents"]), len(MSG) - 1)
        self.assertEqual(c["contents"][-1]["parts"][0]["text"], MSG[-1]["content"])
        self.assertEqual(c["contents"][1]["parts"][0]["text"], pt.FEW_SHOT_HYBRID[1]["content"])

    def test_bagian_thought_dibuang(self):
        llm, _ = buat([resp("Jawaban akhir", thought_teks="pikiran internal")])
        self.assertEqual(llm.generate("m", MSG).teks, "Jawaban akhir")

    def test_jawaban_kosong_adalah_galat(self):
        llm, _ = buat([resp(None, finish="SAFETY")])
        with self.assertRaises(GenerasiError) as cm:
            llm.generate("m", MSG)
        self.assertIn("SAFETY", cm.exception.detail_teknis)

    def test_terpotong_batas_panjang_ditandai(self):
        llm, _ = buat([resp("Jawaban terpo", finish="MAX_TOKENS")])
        h = llm.generate("m", MSG)
        self.assertFalse(h.selesai_normal)
        self.assertEqual(h.finish_reason, "MAX_TOKENS")


class TestGalat(unittest.TestCase):
    def test_429_kuota_tanpa_retry(self):
        llm, m = buat([ApiError(429, "Quota exceeded for %s" % KUNCI, "RESOURCE_EXHAUSTED")])
        with self.assertRaises(KuotaHabisError) as cm:
            llm.generate("m", MSG)
        self.assertEqual(len(m.calls), 1)  # tidak mengulang -> tidak membakar kuota
        self.assertNotIn(KUNCI, str(cm.exception) + cm.exception.detail_teknis)

    def test_401_dan_403_pesan_aman(self):
        for kode in (401, 403):
            llm, _ = buat([ApiError(kode, "API key %s invalid" % KUNCI)])
            with self.assertRaises(GenerasiError) as cm:
                llm.generate("m", MSG)
            self.assertNotIn("kunci", str(cm.exception).lower())
            self.assertNotIn(KUNCI, cm.exception.detail_teknis)

    def test_404_model_tidak_tersedia(self):
        llm, _ = buat([ApiError(404, "model not found")])
        with self.assertRaises(ModelTidakTersediaError):
            llm.generate("m", MSG)

    def test_5xx_dicoba_ulang_lalu_berhasil(self):
        llm, m = buat([ApiError(503, "overloaded"), ApiError(503, "overloaded"), resp("ok")])
        self.assertEqual(llm.generate("m", MSG).teks, "ok")
        self.assertEqual(len(m.calls), 3)

    def test_5xx_terus_menerus_menyerah_setelah_3_kali(self):
        llm, m = buat([ApiError(503, "overloaded")])
        with self.assertRaises(GenerasiError):
            llm.generate("m", MSG)
        self.assertEqual(len(m.calls), 3)

    def test_timeout_dianggap_sementara(self):
        class ReadTimeout(Exception):
            pass

        llm, m = buat([ReadTimeout("timed out"), resp("ok")])
        self.assertEqual(llm.generate("m", MSG).teks, "ok")

    def test_400_lain_tidak_diulang(self):
        llm, m = buat([ApiError(400, "Invalid argument")])
        with self.assertRaises(GenerasiError):
            llm.generate("m", MSG)
        self.assertEqual(len(m.calls), 1)

    def test_galat_tak_dikenal_dibungkus(self):
        llm, _ = buat([ValueError("aneh " + KUNCI)])
        with self.assertRaises(GenerasiError) as cm:
            llm.generate("m", MSG)
        self.assertNotIn(KUNCI, str(cm.exception) + cm.exception.detail_teknis)


class TestFallbackSystemInstruction(unittest.TestCase):
    def test_system_digabung_ke_pesan_pertama_dan_diingat(self):
        tolak = ApiError(400, "Developer instruction is not enabled for models/gemma-4-31b-it")
        llm, m = buat([tolak, resp("ok1"), resp("ok2")])
        self.assertEqual(llm.generate("gemma-4-31b-it", MSG).teks, "ok1")
        pertama, kedua = m.calls[0], m.calls[1]
        self.assertIn("system_instruction", pertama["config"])
        self.assertNotIn("system_instruction", kedua["config"])
        self.assertTrue(kedua["contents"][0]["parts"][0]["text"].startswith(pt.SYSTEM_PROMPT_HYBRID + "\n\n"))
        self.assertIn(pt.FEW_SHOT_HYBRID[0]["content"], kedua["contents"][0]["parts"][0]["text"])
        # panggilan berikutnya langsung memakai mode gabungan (tanpa percobaan yang gagal lagi)
        llm.generate("gemma-4-31b-it", MSG)
        self.assertEqual(len(m.calls), 3)
        self.assertNotIn("system_instruction", m.calls[2]["config"])

    def test_tidak_memengaruhi_model_lain(self):
        llm, m = buat([ApiError(400, "system instruction unsupported"), resp("a"), resp("b")])
        llm.generate("model-a", MSG)
        llm.generate("model-b", MSG)
        self.assertIn("system_instruction", m.calls[2]["config"])


class TestModeBerpikir(unittest.TestCase):
    def test_hanya_thought_dan_max_tokens_didiagnosis(self):
        """Kasus nyata: token berpikir menghabiskan batas, tidak ada teks jawaban."""
        r = NS(candidates=[NS(content=NS(parts=[NS(text="menalar panjang...", thought=True)]),
                              finish_reason=NS(name="MAX_TOKENS"))],
               usage_metadata=NS(thoughts_token_count=1024))
        llm, _ = buat([r])
        with self.assertRaises(GenerasiError) as cm:
            llm.generate("m", MSG)
        d = cm.exception.detail_teknis
        self.assertIn("MAX_TOKENS", d)
        self.assertIn("bagian_thought=1", d)
        self.assertIn("thoughts_token_count=1024", d)

    def test_thinking_level_dapat_diubah_atau_dihilangkan(self):
        llm, m = buat([resp("a"), resp("b")])
        llm.generate("m", MSG, thinking_level="high")
        llm.generate("m", MSG, thinking_level=None)
        self.assertEqual(m.calls[0]["config"]["thinking_config"], {"thinking_level": "high"})
        self.assertNotIn("thinking_config", m.calls[1]["config"])

    def test_model_menolak_thinking_config_dilepas_dan_diingat(self):
        tolak = ApiError(400, "Thinking level is not supported for this model")
        llm, m = buat([tolak, resp("ok1"), resp("ok2")])
        self.assertEqual(llm.generate("model-lama", MSG).teks, "ok1")
        self.assertIn("thinking_config", m.calls[0]["config"])
        self.assertNotIn("thinking_config", m.calls[1]["config"])
        llm.generate("model-lama", MSG)
        self.assertEqual(len(m.calls), 3)  # panggilan berikut langsung tanpa thinking_config
        self.assertNotIn("thinking_config", m.calls[2]["config"])

    def test_penolakan_thinking_tidak_memengaruhi_model_lain(self):
        llm, m = buat([ApiError(400, "thinking not supported"), resp("a"), resp("b")])
        llm.generate("model-a", MSG)
        llm.generate("gemma-4-31b-it", MSG)
        self.assertIn("thinking_config", m.calls[2]["config"])

    def test_penolakan_system_dan_thinking_dapat_terjadi_berurutan(self):
        llm, m = buat([ApiError(400, "Thinking level unsupported"),
                       ApiError(400, "Developer instruction is not enabled"), resp("ok")])
        self.assertEqual(llm.generate("m", MSG).teks, "ok")
        akhir = m.calls[-1]["config"]
        self.assertNotIn("thinking_config", akhir)
        self.assertNotIn("system_instruction", akhir)


class TestDaftarModel(unittest.TestCase):
    DAFTAR = [
        NS(name="models/gemma-4-31b-it", supported_actions=["generateContent"]),
        NS(name="models/gemma-4-26b-a4b-it", supported_actions=["generateContent", "countTokens"]),
        NS(name="models/embedding-001", supported_actions=["embedContent"]),
        NS(name="models/gemini-x", supported_actions=None),
    ]

    def test_daftar_model_menyaring_dan_membuang_awalan(self):
        llm, _ = buat([resp()], daftar=self.DAFTAR)
        self.assertEqual(llm.daftar_model(), {"gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemini-x"})

    def test_hanya_model_terdaftar_yang_ditampilkan(self):
        llm, _ = buat([resp()], daftar=self.DAFTAR[:1])
        tersedia = model_tersedia(LLMRouter({"google": llm}),
                                  model_terkonfigurasi(["gemma-4-31b-it", "gemma-4-26b-a4b-it", "tidak-ada"]))
        self.assertEqual([m.id for m in tersedia], ["gemma-4-31b-it"])
        self.assertEqual(tersedia[0].label_lengkap, "Gemma 4 31B (Google)")

    def test_galat_listing_dipetakan(self):
        llm, _ = buat([resp()], daftar=ApiError(403, "forbidden " + KUNCI))
        with self.assertRaises(GenerasiError) as cm:
            llm.daftar_model()
        self.assertNotIn(KUNCI, str(cm.exception) + cm.exception.detail_teknis)

    def test_model_dievaluasi_tanpa_label_tambahan(self):
        self.assertEqual(ModelInfo(config.MODEL_DIEVALUASI, "Qwen", True).label_lengkap, "Qwen")


class TestKonfigurasiModel(unittest.TestCase):
    def setUp(self):
        for nama, nilai in (("_DOTENV_SUDAH_DIMUAT", True),):
            p = mock.patch.object(config, nama, nilai)
            p.start()
            self.addCleanup(p.stop)
        p2 = mock.patch.object(config, "_dari_streamlit_secrets", lambda n: None)
        p2.start()
        self.addCleanup(p2.stop)

    def test_default_dan_override(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": KUNCI}, clear=True):
            self.assertEqual(config.get_model_ids(), ["google:gemma-4-26b-a4b-it", "google:gemma-4-31b-it"])
        with mock.patch.dict(os.environ, {"GENERATOR_MODELS": " a-model , b-model ,, "}, clear=True):
            self.assertEqual(config.get_model_ids(), ["google:a-model", "google:b-model"])  # tanpa awalan = google

    def test_kunci_api_wajib_dan_tidak_bocor(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(config.ConfigError) as cm:
                config.get_google_api_key()
            self.assertIn("GOOGLE_API_KEY", str(cm.exception))
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": KUNCI}, clear=True):
            self.assertEqual(config.get_google_api_key(), KUNCI)

    def test_mode_potong(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.get_mode_potong(), "disempurnakan")
        with mock.patch.dict(os.environ, {"MODE_POTONG": "EKSPERIMEN"}, clear=True):
            self.assertEqual(config.get_mode_potong(), "eksperimen")
        with mock.patch.dict(os.environ, {"MODE_POTONG": "ngawur"}, clear=True):
            with self.assertRaises(config.ConfigError):
                config.get_mode_potong()


if __name__ == "__main__":
    unittest.main()
