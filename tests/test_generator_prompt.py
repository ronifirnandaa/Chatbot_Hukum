"""Kesetaraan prompt/pemotongan dengan notebook generator (offline).

Fungsi `generate_jawaban_rag` asli dijalankan (salinan verbatim di tests/referensi/generator_notebook.py) dengan
`_ollama_chat` diganti penangkap; pesan yang akan dikirim ke model dibandingkan dengan pesan versi website.
"""
import importlib.util
import unittest
from pathlib import Path

from src.config import MAX_CONTEXT_CHARS, MAX_CONTEXT_CHARS_PER_DOC, MAX_NEW_TOKENS, TEMPERATURE
from src.generator import prompt_template as pt
from src.generator.context_builder import potong_konteks_eksperimen, susun_konteks_penuh


def muat_ref():
    path = Path(__file__).parent / "referensi" / "generator_notebook.py"
    spec = importlib.util.spec_from_file_location("generator_notebook_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def item(aid, panjang, n_pasal=0):
    teks = "[ULASAN / JAWABAN]: " + ("kata " * panjang)[:panjang]
    labels = ["Pasal %d (KUHP) [law_%06d]" % (i, i) for i in range(1, n_pasal + 1)]
    if labels:
        teks += "\n\n[PASAL TERKAIT]:\n- " + "\n- ".join(labels)
    return {"node_id": str(aid), "text": teks, "pasal_terkait": [{"label": l} for l in labels]}


class TestKesetaraanPrompt(unittest.TestCase):
    def setUp(self):
        self.ref = muat_ref()

    def _pesan_notebook(self, pertanyaan, konteks_penuh):
        tangkap = {}

        def palsu(messages, max_new_tokens=768, temperature=0.1, do_sample=False):
            tangkap.update(messages=messages, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
            return "ok"

        self.ref._ollama_chat = palsu
        self.ref.generate_jawaban_rag(None, None, pertanyaan, konteks_penuh, retriever="hybrid",
                                      max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return tangkap

    def test_system_dan_few_shot_identik(self):
        self.assertEqual(pt.SYSTEM_PROMPT_HYBRID, self.ref.system_prompt["hybrid"])
        self.assertEqual(pt.FEW_SHOT_HYBRID, self.ref.few_shots["hybrid"])

    def test_pesan_identik_untuk_berbagai_ukuran_konteks(self):
        kasus = {
            "kosong": [],
            "satu_pendek": [item(1, 300, 2)],
            "lima_pendek": [item(i, 400, 3) for i in range(1, 6)],
            "lima_panjang_seperti_smoke_test": [item(i, p, 30) for i, p in zip((1, 807, 340, 276, 973), (10495, 8541, 5664, 6128, 7943))],
            "dua_puluh": [item(i, 3000, 10) for i in range(1, 21)],
        }
        for nama, items in kasus.items():
            with self.subTest(nama):
                penuh = susun_konteks_penuh(items)
                pertanyaan = "Apakah bisa dijerat pasal korupsi? (%s)" % nama
                tangkap = self._pesan_notebook(pertanyaan, penuh)
                konteks = potong_konteks_eksperimen(penuh, MAX_CONTEXT_CHARS, MAX_CONTEXT_CHARS_PER_DOC)
                self.assertEqual(tangkap["messages"], pt.bangun_messages(pertanyaan, konteks))

    def test_parameter_generation_sama_dengan_eksperimen(self):
        tangkap = self._pesan_notebook("q", "")
        self.assertEqual(tangkap["max_new_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(MAX_NEW_TOKENS, 1024)
        self.assertFalse(tangkap["do_sample"])  # do_sample=False -> temperature efektif 0
        self.assertEqual(TEMPERATURE, 0.0)

    def test_urutan_pesan(self):
        m = pt.bangun_messages("Pertanyaan?", "KONTEKS")
        self.assertEqual([x["role"] for x in m], ["system", "user", "assistant", "user"])
        self.assertTrue(m[-1]["content"].endswith("Pertanyaan:\nPertanyaan?"))
        self.assertIn("---\nKONTEKS\n---", m[-1]["content"])


if __name__ == "__main__":
    unittest.main()
