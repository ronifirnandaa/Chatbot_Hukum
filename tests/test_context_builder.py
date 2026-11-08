"""Uji penyusun konteks prompt (offline)."""
import importlib.util
import re
import unittest
from pathlib import Path

from src.config import MAX_CONTEXT_CHARS, MAX_CONTEXT_CHARS_PER_DOC, TOTAL_MAKS_DISEMPURNAKAN
from src.generator.context_builder import (
    potong_konteks_disempurnakan,
    potong_konteks_eksperimen,
    siapkan_konteks,
    susun_konteks_penuh,
)


def muat_ref():
    path = Path(__file__).parent / "referensi" / "generator_notebook.py"
    spec = importlib.util.spec_from_file_location("generator_notebook_ref2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def item(aid, panjang_ulasan, n_pasal=0):
    ulasan = ("kata " * (panjang_ulasan // 5 + 1))[:panjang_ulasan]
    labels = ["Pasal %d (KUHP) [law_%06d]" % (i, i) for i in range(1, n_pasal + 1)]
    teks = "[ULASAN / JAWABAN]: " + ulasan
    if labels:
        teks += "\n\n[PASAL TERKAIT]:\n- " + "\n- ".join(labels)
    return {"node_id": str(aid), "text": teks, "pasal_terkait": [{"label": l} for l in labels]}


class TestFormatPenuh(unittest.TestCase):
    def test_panjang_sama_dengan_smoke_test_notebook(self):
        # Panjang teks & node_id dari smoke test retriever; notebook generator mencatat konteks 38.902 karakter.
        panjang = (10495, 8541, 5664, 6128, 7943)
        items = [{"node_id": nid, "text": "x" * p} for nid, p in zip(("1", "807", "340", "276", "973"), panjang)]
        self.assertEqual(len(susun_konteks_penuh(items)), 38902)

    def test_format_blok(self):
        s = susun_konteks_penuh([{"node_id": "7", "text": "AAA"}, {"node_id": "9", "text": "BBB"}])
        self.assertEqual(s, "[Konteks 1]\nNode ID: 7\nAAA\n\n[Konteks 2]\nNode ID: 9\nBBB")
        self.assertEqual(susun_konteks_penuh([]), "")


class TestModeEksperimen(unittest.TestCase):
    def test_identik_dengan_fungsi_notebook(self):
        ref = muat_ref()
        for n_item, p in ((0, 0), (1, 100), (1, 9000), (3, 1000), (5, 2000), (5, 9000), (12, 3000)):
            penuh = susun_konteks_penuh([item(i + 1, p, 4) for i in range(n_item)])
            with self.subTest(n_item=n_item, p=p):
                self.assertEqual(
                    potong_konteks_eksperimen(penuh, MAX_CONTEXT_CHARS, MAX_CONTEXT_CHARS_PER_DOC),
                    ref.potong_konteks(penuh, max_total=MAX_CONTEXT_CHARS, max_per_doc=MAX_CONTEXT_CHARS_PER_DOC),
                )

    def test_alasan_mode_disempurnakan_pasal_terbuang_pada_eksperimen(self):
        items = [item(i, 6000, 20) for i in range(1, 4)]
        self.assertNotIn("[PASAL TERKAIT]", siapkan_konteks(items, "eksperimen").teks)
        self.assertEqual(siapkan_konteks(items, "disempurnakan").teks.count("[PASAL TERKAIT]:"), 3)


class TestModeDisempurnakan(unittest.TestCase):
    def test_pasal_dibatasi_dan_sisa_dicatat(self):
        k = siapkan_konteks([item(1, 500, 34)], "disempurnakan")
        self.assertEqual(k.teks.count("\n- Pasal "), 8)
        self.assertIn("- ... dan 26 pasal terkait lainnya (tidak ditampilkan)", k.teks)
        self.assertEqual(k.n_pasal_tidak_ditampilkan, 26)
        self.assertTrue(k.dipotong)

    def test_ulasan_panjang_dipotong_pasal_tetap_ada(self):
        k = siapkan_konteks([item(1, 9000, 5)], "disempurnakan")
        self.assertIn("... [ulasan dipotong]", k.teks)
        self.assertIn("- Pasal 5 (KUHP) [law_000005]", k.teks)
        self.assertLess(len(k.teks), 2700)

    def test_dokumen_pendek_tidak_diubah(self):
        items = [item(1, 300, 3), item(2, 500, 2)]
        k = siapkan_konteks(items, "disempurnakan")
        self.assertEqual(k.teks, susun_konteks_penuh(items))  # identik dengan format penuh
        self.assertFalse(k.dipotong)
        self.assertEqual((k.n_dokumen_total, k.n_dokumen_dipakai), (2, 2))

    def test_total_dibatasi_blok_dibuang_utuh(self):
        items = [item(i, 9000, 34) for i in range(1, 11)]
        k = siapkan_konteks(items, "disempurnakan")
        self.assertLessEqual(len(k.teks), TOTAL_MAKS_DISEMPURNAKAN)
        self.assertLess(k.n_dokumen_dipakai, 10)
        self.assertTrue(k.dipotong)
        nomor = [int(n) for n in re.findall(r"\[Konteks (\d+)\]", k.teks)]
        self.assertEqual(nomor, list(range(1, k.n_dokumen_dipakai + 1)))  # berurutan, tanpa blok terpotong
        self.assertEqual(k.teks.count("[PASAL TERKAIT]:"), k.n_dokumen_dipakai)

    def test_lima_dokumen_besar_muat_semua(self):
        k = siapkan_konteks([item(i, 10000, 34) for i in range(1, 6)], "disempurnakan")
        self.assertEqual(k.n_dokumen_dipakai, 5)
        self.assertLessEqual(k.panjang_akhir, TOTAL_MAKS_DISEMPURNAKAN)

    def test_tanpa_ulasan_hanya_pasal(self):
        it = {"node_id": "3", "text": "[PASAL TERKAIT]:\n- Pasal 1 [law_000001]",
              "pasal_terkait": [{"label": "Pasal 1 [law_000001]"}]}
        k = siapkan_konteks([it], "disempurnakan")
        self.assertEqual(k.teks, "[Konteks 1]\nNode ID: 3\n[PASAL TERKAIT]:\n- Pasal 1 [law_000001]")

    def test_tanpa_pasal(self):
        k = siapkan_konteks([item(1, 400, 0)], "disempurnakan")
        self.assertNotIn("PASAL TERKAIT", k.teks)

    def test_kosong_dan_mode_tidak_dikenal(self):
        k = siapkan_konteks([], "disempurnakan")
        self.assertEqual((k.teks, k.n_dokumen_dipakai, k.dipotong), ("", 0, False))
        with self.assertRaises(ValueError):
            siapkan_konteks([item(1, 10)], "acak")

    def test_fungsi_mengembalikan_empat_nilai(self):
        teks, dipakai, potong, hilang = potong_konteks_disempurnakan([item(1, 100, 9)])
        self.assertEqual((dipakai, potong, hilang), (1, True, 1))


if __name__ == "__main__":
    unittest.main()
