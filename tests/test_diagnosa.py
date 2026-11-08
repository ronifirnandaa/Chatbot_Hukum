"""Uji logika skrip diagnosis (offline): urutan skenario dan kesimpulan."""
import unittest
from types import SimpleNamespace as NS

from src.generator.llm_client import GeminiLLM

try:
    from diagnosa_gemma import SKENARIO, jalankan, ringkas
except ImportError:
    from tests.diagnosa_gemma import SKENARIO, jalankan, ringkas


class ServerError(Exception):
    def __init__(self, code, message):
        super().__init__("%s %s" % (code, message))
        self.code, self.message = code, message


def ok():
    return NS(candidates=[NS(content=NS(parts=[NS(text="Jakarta.", thought=None)]), finish_reason=NS(name="STOP"))],
              usage_metadata=NS(thoughts_token_count=0))


class Models:
    """Gagal 500 hanya bila thinking_config dikirim (meniru dugaan penyebab)."""

    def __init__(self, gagal_bila_thinking=True, selalu_gagal=False):
        self.gagal_thinking, self.selalu, self.calls = gagal_bila_thinking, selalu_gagal, []

    def generate_content(self, model, contents, config):
        self.calls.append(config)
        if self.selalu or (self.gagal_thinking and "thinking_config" in config):
            raise ServerError(500, "Internal error encountered.")
        return ok()


def jalan(models, ids=("m1",)):
    llm = GeminiLLM(NS(models=models), sleep=lambda s: None)
    return jalankan(llm, ids, sleep=lambda s: None, cetak=lambda *a, **k: None)


class TestDiagnosa(unittest.TestCase):
    def test_empat_skenario_per_model_satu_percobaan_tanpa_retry(self):
        m = Models(gagal_bila_thinking=False)
        hasil = jalan(m, ("m1", "m2"))
        self.assertEqual(len(hasil), 2 * len(SKENARIO))
        self.assertEqual(len(m.calls), 2 * len(SKENARIO))  # tidak ada retry tersembunyi

    def test_thinking_memicu_500_disimpulkan(self):
        hasil = jalan(Models(gagal_bila_thinking=True))
        st = {h["kode"]: h["status"] for h in hasil}
        self.assertEqual(st, {"A": "OK", "B": "GAGAL", "C": "GAGAL", "D": "OK"})
        self.assertIn("thinking_level=minimal memicu galat", ringkas(hasil)[0])
        self.assertIn("500", [h for h in hasil if h["kode"] == "B"][0]["info"])

    def test_semua_gagal_disimpulkan_layanan_bermasalah(self):
        self.assertIn("layanan Google", ringkas(jalan(Models(selalu_gagal=True)))[0])

    def test_semua_berhasil(self):
        self.assertIn("kemungkinan sementara", ringkas(jalan(Models(gagal_bila_thinking=False)))[0])


if __name__ == "__main__":
    unittest.main()
