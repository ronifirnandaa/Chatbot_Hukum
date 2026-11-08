"""Uji logika skrip perbandingan gaya (offline)."""
import tempfile
import unittest
from pathlib import Path

from src.generator.llm_client import GenerasiError, HasilLLM, KuotaHabisError

try:
    from bandingkan_gaya import bandingkan_gaya, cetak_tabel
    from test_chat_service import KGTiruan, buat_retriever  # noqa: F401
except ImportError:
    from tests.bandingkan_gaya import bandingkan_gaya, cetak_tabel
    from tests.test_hybrid_retriever import KGTiruan, buat_retriever

RINGKAS = "**Jawaban singkat:** Ya.\n\n**Dasar hukum:**\n- Pasal 48 KUHP — hapusnya pidana."
NARATIF = "**Analisis**\n\nParagraf panjang tentang Pasal 48 KUHP dan Pasal 777."


class LLMGaya:
    """Menjawab sesuai gaya (dibedakan dari jumlah pesan: ringkas=2, naratif=4); bisa 429 sekali di awal."""

    def __init__(self, kuota_dulu=False, galat_naratif=False):
        self.kuota, self.galat_naratif, self.n = kuota_dulu, galat_naratif, 0

    def generate(self, model_id, messages, max_new_tokens, temperature):
        self.n += 1
        if self.kuota:
            self.kuota = False
            raise KuotaHabisError("kuota", "ClientError [429]")
        if len(messages) == 4:
            if self.galat_naratif:
                raise GenerasiError("gagal", "ServerError [503]")
            return HasilLLM(NARATIF, True, "STOP")
        return HasilLLM(RINGKAS, True, "STOP")


def items():
    return buat_retriever(KGTiruan()).cari_dengan_info("q", 5).items


class TestBandingkanGaya(unittest.TestCase):
    def test_dua_gaya_konteks_sama_dan_metrik(self):
        with tempfile.TemporaryDirectory() as d:
            b = bandingkan_gaya(LLMGaya(), "m", items(), "q?", sleep=lambda s: None, folder=Path(d), mode_potong="disempurnakan")
            self.assertEqual([x["gaya"] for x in b], ["ringkas", "naratif"])
            self.assertEqual(b[0]["format"], "sesuai")
            self.assertEqual(b[1]["format"], "(n/a)")
            self.assertEqual(b[0]["kutipan_hilang"], "-")
            self.assertIn("777", b[1]["kutipan_hilang"])  # kutipan karangan pada jawaban naratif terdeteksi
            self.assertLess(b[0]["kata"], b[1]["kata"] + 100)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["naratif.txt", "ringkas.txt"])
            self.assertEqual((Path(d) / "ringkas.txt").read_text(encoding="utf-8"), RINGKAS)

    def test_429_menunggu_lalu_berhasil(self):
        tidur, llm = [], LLMGaya(kuota_dulu=True)
        b = bandingkan_gaya(llm, "m", items(), "q?", sleep=tidur.append, catat=lambda *a: None, mode_potong="disempurnakan")
        self.assertEqual([x["galat"] for x in b], ["", ""])
        self.assertIn(65, tidur)
        self.assertEqual(llm.n, 3)  # ringkas (429) + ringkas (ulang) + naratif

    def test_galat_satu_gaya_tidak_menghentikan_yang_lain(self):
        b = bandingkan_gaya(LLMGaya(galat_naratif=True), "m", items(), "q?", sleep=lambda s: None, mode_potong="disempurnakan")
        self.assertEqual(b[0]["galat"], "")
        self.assertIn("503", b[1]["galat"])

    def test_tabel(self):
        b = bandingkan_gaya(LLMGaya(), "m", items(), "q?", sleep=lambda s: None, mode_potong="disempurnakan")
        keluaran = []
        cetak_tabel(b, cetak=keluaran.append)
        self.assertEqual(len(keluaran), 3)  # header + 2 gaya


if __name__ == "__main__":
    unittest.main()
