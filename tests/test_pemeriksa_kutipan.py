"""Uji pemeriksa kutipan dan logika skrip perbandingan konteks (offline)."""
import tempfile
import unittest
from pathlib import Path

from src.generator.llm_client import GenerasiError, HasilLLM, KuotaHabisError
from src.generator.pemeriksa_kutipan import ekstrak, periksa_kutipan

try:
    from bandingkan_konteks import bandingkan, cetak_tabel, pengaturan_konteks
except ImportError:
    from tests.bandingkan_konteks import bandingkan, cetak_tabel, pengaturan_konteks


class TestEkstrak(unittest.TestCase):
    def test_pasal_peraturan_putusan(self):
        e = ekstrak("Pasal 3 UU 31/1999 jo. Putusan MK No. 25/PUU-XIV/2016, juga Pasal 2 ayat (1) dan Pasal 55A.")
        self.assertEqual(e["pasal"], {"3", "2", "55A"})
        self.assertEqual(e["peraturan"], {"31/1999"})
        self.assertEqual(e["putusan"], {"25/PUU-XIV/2016"})

    def test_bentuk_penulisan_peraturan_dinormalkan(self):
        for t in ("UU 31/1999", "Undang-Undang Nomor 31 Tahun 1999", "undang-undang no. 31 tahun 1999", "UU No. 031/1999"):
            self.assertEqual(ekstrak(t)["peraturan"], {"31/1999"}, t)
        self.assertEqual(ekstrak("Peraturan Pemerintah Nomor 2 Tahun 2018")["peraturan"], {"2/2018"})

    def test_nomor_putusan_tidak_dikira_peraturan(self):
        self.assertEqual(ekstrak("Putusan MK No. 25/PUU-XIV/2016")["peraturan"], set())

    def test_kosong_dan_none(self):
        for t in ("", None):
            self.assertEqual(ekstrak(t), {"pasal": set(), "peraturan": set(), "putusan": set()})


class TestPeriksa(unittest.TestCase):
    KONTEKS = ("[ULASAN / JAWABAN]: Pasal 2 ayat (1) dan Pasal 3 Undang-Undang Nomor 31 Tahun 1999 ... "
               "Putusan MK 25/PUU-XIV/2016.\n[PASAL TERKAIT]:\n- Pasal 3 (UU 31/1999) [law_000003]")

    def test_semua_ditemukan(self):
        h = periksa_kutipan("Dijerat Pasal 3 UU 31/1999 dan Pasal 2 ayat (1) sesuai Putusan MK No. 25/PUU-XIV/2016.", self.KONTEKS)
        self.assertTrue(h.semua_ditemukan)
        self.assertEqual(h.n_dikutip, 4)
        self.assertIn("semua 4 kutipan", h.ringkas())

    def test_kutipan_karangan_ditandai(self):
        h = periksa_kutipan("Pasal 3 UU 31/1999, Pasal 55 KUHP, dan UU 8/2010, Putusan 99/PUU-X/2020.", self.KONTEKS)
        self.assertFalse(h.semua_ditemukan)
        self.assertEqual(h.tidak_ditemukan["pasal"], {"55"})
        self.assertEqual(h.tidak_ditemukan["peraturan"], {"8/2010"})
        self.assertEqual(h.tidak_ditemukan["putusan"], {"99/PUU-X/2020"})
        self.assertEqual(h.n_tidak_ditemukan, 3)
        self.assertIn("3 dari", h.ringkas())

    def test_jawaban_tanpa_kutipan(self):
        h = periksa_kutipan("Maaf, informasi hukum tersebut tidak tersedia dalam basis data saya.", self.KONTEKS)
        self.assertEqual((h.n_dikutip, h.semua_ditemukan), (0, True))


class LLMTiruan:
    def __init__(self):
        self.pesan = []

    def generate(self, model_id, messages):
        self.pesan.append(messages)
        konteks = messages[-1]["content"]
        if "Pasal 999" in konteks:
            raise GenerasiError("gagal", "ServerError [503]")
        return HasilLLM("Dijerat Pasal 3 dan Pasal 777 UU 31/1999.", True, "STOP")


def item(aid, n_ulasan, n_pasal=3):
    labels = ["Pasal %d (KUHP) [law_%06d]" % (i, i) for i in range(1, n_pasal + 1)]
    teks = "[ULASAN / JAWABAN]: " + "Pasal 3 UU 31/1999. " * (n_ulasan // 20) + "\n\n[PASAL TERKAIT]:\n- " + "\n- ".join(labels)
    return {"node_id": str(aid), "text": teks, "pasal_terkait": [{"label": l} for l in labels]}


class TestBandingkan(unittest.TestCase):
    def test_empat_pengaturan_makin_panjang(self):
        items = [item(i, 9000, 30) for i in range(1, 6)]
        nama_ukuran = [(n, len(k)) for n, k in pengaturan_konteks(items)]
        self.assertEqual([n for n, _ in nama_ukuran], ["eksperimen", "disempurnakan", "luas", "penuh"])
        u = dict(nama_ukuran)
        self.assertLess(u["disempurnakan"], u["luas"])
        self.assertLess(u["luas"], u["penuh"])

    def test_baris_hasil_kutipan_dan_berkas(self):
        items = [item(i, 3000, 5) for i in range(1, 3)]
        with tempfile.TemporaryDirectory() as d:
            baris = bandingkan(LLMTiruan(), "m", items, "q?", sleep=lambda s: None, folder=Path(d))
            self.assertEqual(len(baris), 4)
            self.assertTrue(all(b["n_kutipan"] == 3 for b in baris))  # Pasal 3, Pasal 777, UU 31/1999
            self.assertTrue(all("pasal 777" in b["tidak_ditemukan"] for b in baris))
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["disempurnakan.txt", "eksperimen.txt", "luas.txt", "penuh.txt"])
            self.assertIn("777", (Path(d) / "penuh.txt").read_text(encoding="utf-8"))
        keluaran = []
        cetak_tabel(baris, cetak=keluaran.append)
        self.assertEqual(len(keluaran), 1 + 4)  # header + 4 pengaturan

    def test_galat_satu_pengaturan_tidak_menghentikan_yang_lain(self):
        bocor = [{"node_id": "1", "text": "[ULASAN / JAWABAN]: Pasal 999 saja.", "pasal_terkait": []}]
        baris = bandingkan(LLMTiruan(), "m", bocor, "q?", sleep=lambda s: None)
        self.assertEqual(len(baris), 4)
        self.assertTrue(all(b["galat"] for b in baris))


class LLMKuota:
    """429 pada `gagal_sebanyak` panggilan pertama, setelah itu berhasil."""

    def __init__(self, gagal_sebanyak):
        self.sisa, self.panggilan = gagal_sebanyak, 0

    def generate(self, model_id, messages):
        self.panggilan += 1
        if self.sisa > 0:
            self.sisa -= 1
            raise KuotaHabisError("kuota", "ClientError [429]: quota")
        return HasilLLM("Pasal 3 UU 31/1999.", True, "STOP")


class TestKuotaDanFilter(unittest.TestCase):
    ITEMS = [item(1, 600, 2)]

    def test_429_menunggu_lalu_berhasil(self):
        tidur = []
        llm = LLMKuota(1)
        baris = bandingkan(llm, "m", self.ITEMS, "q?", sleep=tidur.append, hanya={"luas"}, catat=lambda *a: None)
        self.assertEqual(len(baris), 1)
        self.assertEqual((baris[0]["nama"], baris[0]["galat"], baris[0]["ulang"]), ("luas", "", True))
        self.assertIn(65, tidur)
        keluaran = []
        cetak_tabel(baris, cetak=keluaran.append)
        self.assertTrue(any("luas*" in k for k in keluaran))
        self.assertTrue(any("diulang setelah menunggu" in k for k in keluaran))

    def test_429_dua_kali_dicatat_sebagai_galat_tanpa_ulang_lagi(self):
        llm = LLMKuota(5)
        baris = bandingkan(llm, "m", self.ITEMS, "q?", sleep=lambda s: None, hanya={"penuh"}, catat=lambda *a: None)
        self.assertIn("429", baris[0]["galat"])
        self.assertEqual(llm.panggilan, 2)  # satu percobaan + satu ulangan, tidak lebih

    def test_hanya_menyaring_pengaturan(self):
        llm = LLMKuota(0)
        baris = bandingkan(llm, "m", self.ITEMS, "q?", sleep=lambda s: None, hanya={"luas", "penuh"})
        self.assertEqual([b["nama"] for b in baris], ["luas", "penuh"])
        self.assertEqual(llm.panggilan, 2)  # tidak memanggil model untuk pengaturan yang disaring

    def test_jeda_dapat_diatur(self):
        tidur = []
        bandingkan(LLMKuota(0), "m", self.ITEMS, "q?", sleep=tidur.append, hanya={"luas"}, jeda=7)
        self.assertEqual(tidur, [7])


if __name__ == "__main__":
    unittest.main()
