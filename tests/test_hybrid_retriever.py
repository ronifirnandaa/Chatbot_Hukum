"""Uji retriever (offline, tanpa Neo4j sungguhan).

Inti uji: KESETARAAN. Kelas notebook asli (tests/referensi/retriever_notebook.py, salinan verbatim) dan
kelas website dijalankan pada KG tiruan yang SAMA, lalu keluaran dan parameter query dibandingkan.
Ini membuktikan bahwa ekstraksi tidak mengubah algoritma. Ini BUKAN pengganti validasi pada Neo4j
sungguhan (lihat tests/cek_retriever.py).
"""
import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from src.config import Neo4jSettings
from src.retriever.hybrid_retriever import HybridGraphRAGRetriever, RetrievalError
from src.retriever.neo4j_connection import Neo4jConnection

SET = Neo4jSettings(uri="neo4j+s://abcd1234.databases.neo4j.io", username="userabcd",
                    password="rahasia-sangat-panjang", database="dbabcd")

# ------------------------------------------------------------------------------ KG tiruan
QA = {  # answer_id (str, seperti di database) -> data
    "1": {"ulasan": "Ulasan satu", "question": "Q1", "refs": ["law_1", "law_2"]},
    "2": {"ulasan": "Ulasan dua", "question": "Q2", "refs": ["law_2"]},
    "3": {"ulasan": "Ulasan tiga", "question": "Q3", "refs": ["law_3", "law_4"]},
    "4": {"ulasan": None, "question": "Pertanyaan empat", "refs": []},
    "5": {"ulasan": "", "question": None, "refs": []},
}
LAW = {
    "law_1": {"pasal": "Pasal 48", "judul": "KUHP"},
    "law_2": {"pasal": "Pasal 2", "judul": None},
    "law_3": {"pasal": None, "judul": "UU Tipikor"},
    "law_4": {"pasal": None, "judul": None},
}
VEC = {
    "legalqa_vector": [
        {"label": "LegalQA", "answer_id": "1", "law_id": None, "score": 0.95},
        {"label": "LegalQA", "answer_id": "4", "law_id": None, "score": 0.60},
        {"label": "LegalQA", "answer_id": "2", "law_id": None, "score": 0.80},
    ],
    "law_vector": [
        {"label": "Law", "answer_id": None, "law_id": "law_2", "score": 0.90},
        {"label": "Law", "answer_id": None, "law_id": "law_3", "score": 0.70},
        {"label": "Law", "answer_id": None, "law_id": "law_1", "score": 0.50},
    ],
}


class IndexTidakAda(Exception):
    pass


class KGTiruan:
    """Meniru semantik dua query Cypher retriever + mencatat parameter yang diterima."""

    def __init__(self, vec=None, gagal=()):
        self.vec = VEC if vec is None else vec
        self.gagal = set(gagal)
        self.log = []

    def __call__(self, query, **p):
        if "db.index.vector.queryNodes" in query:
            self.log.append(("vec", p["index_name"], p["k"], tuple(p["embedding"])))
            if p["index_name"] in self.gagal:
                return IndexTidakAda("There is no such vector schema index")
            return [dict(r) for r in self.vec.get(p["index_name"], [])[: p["k"]]]
        if "UNWIND $anchors" in query:
            self.log.append(("trav", tuple((a["answer_id"], a["law_id"], a["score"]) for a in p["anchors"]), p["limit"]))
            baris = []
            for a in p["anchors"]:
                q = a["answer_id"] if a["answer_id"] is not None and str(a["answer_id"]) in QA else None
                qps = [i for i, d in QA.items() if a["law_id"] is not None and a["law_id"] in d["refs"]] or [None]
                for qp in qps:
                    pilih = q or qp
                    if pilih is not None and (a["score"], pilih) not in [(b[0], b[1]) for b in baris]:
                        baris.append((a["score"], pilih))
            hasil = []
            for skor, aid in baris:
                d = QA[aid]
                teks = d["ulasan"] if d["ulasan"] is not None else (d["question"] or "")
                pasal = [{"law_id": l, "pasal": LAW[l]["pasal"], "judul": LAW[l]["judul"]} for l in d["refs"]]
                hasil.append({"node_id": aid, "text": teks, "score": skor, "pasal_terkait": pasal})
            hasil.sort(key=lambda r: r["score"], reverse=True)
            return hasil[: p["limit"]]
        return []


class FakeSession:
    def __init__(self, router):
        self.router = router

    def run(self, query, **p):
        out = self.router(query, **p)
        if isinstance(out, Exception):
            raise out

        class R:
            def data(self_inner):
                return out

        return R()

    def close(self):
        pass


class FakeDriver:
    def __init__(self, router):
        self.router = router

    def verify_connectivity(self):
        pass

    def session(self, database=None):
        return FakeSession(self.router)

    def close(self):
        pass


class Vektor(list):
    def tolist(self):
        return list(self)


class EncoderTiruan:
    def encode(self, teks, show_progress_bar=False):
        return Vektor([round(len(teks) / 100.0, 4), 0.5, -0.25])


def buat_koneksi(router):
    return Neo4jConnection(SET, driver_factory=lambda uri, auth: FakeDriver(router), sleep=lambda s: None)


def buat_retriever(router):
    return HybridGraphRAGRetriever(buat_koneksi(router), encoder=EncoderTiruan())


# ------------------------------------------------------------------------------ kelas referensi
def muat_kelas_referensi(holder):
    class _GD:
        @staticmethod
        def driver(uri, auth=None, **kw):
            return holder["driver"]

    class _ST:
        def __init__(self, model_name, device=None):
            pass

        def encode(self, teks, show_progress_bar=False):
            return EncoderTiruan().encode(teks)

    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = _ST
    neo = types.ModuleType("neo4j")
    neo.GraphDatabase = _GD
    path = Path(__file__).parent / "referensi" / "retriever_notebook.py"
    with mock.patch.dict(sys.modules, {"sentence_transformers": st, "neo4j": neo}):
        spec = importlib.util.spec_from_file_location("retriever_notebook_ref", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod.HybridGraphRAGRetriever


def jalankan_keduanya(kg_baru, kg_ref, pertanyaan, top_k):
    """Jalankan kelas notebook dan kelas website; kembalikan (hasil_ref, hasil_baru)."""
    holder = {"driver": FakeDriver(kg_ref)}
    Ref = muat_kelas_referensi(holder)
    ref = Ref("neo4j+s://x", "u", "p", database="db")
    baru = buat_retriever(kg_baru)
    with contextlib.redirect_stdout(io.StringIO()):  # kelas notebook mencetak "[warn]"
        h_ref = ref.cari_hybrid(pertanyaan, top_k)
    h_baru = baru.cari_hybrid(pertanyaan, top_k)
    return h_ref, h_baru


class TestKesetaraanDenganNotebook(unittest.TestCase):
    PERTANYAAN = "Apa pasal yang menjerat pengurus yayasan yang menyalahgunakan dana donasi?"

    def test_keluaran_identik_untuk_berbagai_top_k(self):
        for k in (1, 2, 3, 4, 5, 10):
            with self.subTest(top_k=k):
                kg_a, kg_b = KGTiruan(), KGTiruan()
                h_ref, h_baru = jalankan_keduanya(kg_b, kg_a, self.PERTANYAAN, k)
                self.assertEqual(h_ref, h_baru)
                self.assertEqual(kg_a.log, kg_b.log)  # parameter query (k, limit=2k, embedding, anchor) sama persis
                self.assertLessEqual(len(h_baru), k)

    def test_hasil_tidak_kosong_pada_skenario_normal(self):
        h_ref, h_baru = jalankan_keduanya(KGTiruan(), KGTiruan(), self.PERTANYAAN, 5)
        self.assertGreaterEqual(len(h_baru), 3)  # mencegah uji kesetaraan "lulus" pada dua daftar kosong

    def test_satu_indeks_gagal_sama_dengan_notebook(self):
        for gagal in ("law_vector", "legalqa_vector"):
            with self.subTest(gagal=gagal):
                kg_a, kg_b = KGTiruan(gagal=[gagal]), KGTiruan(gagal=[gagal])
                h_ref, h_baru = jalankan_keduanya(kg_b, kg_a, self.PERTANYAAN, 5)
                self.assertEqual(h_ref, h_baru)
                self.assertGreater(len(h_baru), 0)

    def test_semua_indeks_gagal_keduanya_error(self):
        holder = {"driver": FakeDriver(KGTiruan(gagal=["legalqa_vector", "law_vector"]))}
        ref = muat_kelas_referensi(holder)("neo4j+s://x", "u", "p")
        with self.assertRaises(RuntimeError):
            ref.cari_hybrid(self.PERTANYAAN, 5)
        with self.assertRaises(RetrievalError):
            buat_retriever(KGTiruan(gagal=["legalqa_vector", "law_vector"])).cari_hybrid(self.PERTANYAAN, 5)

    def test_anchor_ganda_ambil_skor_tertinggi_sama_dengan_notebook(self):
        """Anchor dengan (label, id) sama tetapi skor berbeda: yang tertinggi harus menang (urutan input diacak)."""
        vec = {
            "legalqa_vector": [
                {"label": "LegalQA", "answer_id": "1", "law_id": None, "score": 0.40},
                {"label": "LegalQA", "answer_id": "2", "law_id": None, "score": 0.85},
                {"label": "LegalQA", "answer_id": "1", "law_id": None, "score": 0.95},
            ],
            "law_vector": [
                {"label": "Law", "answer_id": None, "law_id": "law_3", "score": 0.30},
                {"label": "Law", "answer_id": None, "law_id": "law_3", "score": 0.75},
            ],
        }
        kg_a, kg_b = KGTiruan(vec=vec), KGTiruan(vec=vec)
        h_ref, h_baru = jalankan_keduanya(kg_b, kg_a, self.PERTANYAAN, 5)
        self.assertEqual(h_ref, h_baru)
        self.assertEqual(kg_a.log, kg_b.log)
        skor = {i["node_id"]: i["score"] for i in h_baru}
        self.assertEqual(skor["1"], 0.95)  # bukan 0.40
        self.assertEqual(skor["3"], 0.75)  # bukan 0.30

    def test_satu_anchor_law_banyak_ulasan_dipotong_ke_top_k(self):
        """Satu anchor Law dirujuk >1 LegalQA: hasil akhir harus tetap dipotong ke top_k (sama dengan notebook)."""
        vec = {"legalqa_vector": [],
               "law_vector": [{"label": "Law", "answer_id": None, "law_id": "law_2", "score": 0.99}]}
        kg_a, kg_b = KGTiruan(vec=vec), KGTiruan(vec=vec)
        h_ref, h_baru = jalankan_keduanya(kg_b, kg_a, self.PERTANYAAN, 1)
        self.assertEqual(h_ref, h_baru)
        self.assertEqual(len(h_baru), 1)
        # tanpa pemotongan, traversal (LIMIT 2*top_k=2) akan mengembalikan 2 ulasan (law_2 dirujuk LegalQA 1 dan 2)
        self.assertEqual(len(buat_retriever(KGTiruan(vec=vec)).cari_hybrid(self.PERTANYAAN, 2)), 2)

    def test_tanpa_anchor_keduanya_kosong(self):
        h_ref, h_baru = jalankan_keduanya(KGTiruan(vec={}), KGTiruan(vec={}), self.PERTANYAAN, 5)
        self.assertEqual((h_ref, h_baru), ([], []))


class TestPerilakuRetriever(unittest.TestCase):
    def setUp(self):
        self.r = buat_retriever(KGTiruan())

    def test_format_teks_dan_label_pasal(self):
        item = {i["node_id"]: i for i in self.r.cari_hybrid("tes", 10)}
        self.assertEqual(
            item["1"]["text"],
            "[ULASAN / JAWABAN]: Ulasan satu\n\n[PASAL TERKAIT]:\n- Pasal 48 (KUHP) [law_1]\n- Pasal 2 [law_2]",
        )
        self.assertEqual([p["label"] for p in item["3"]["pasal_terkait"]], ["[law_3]", "[law_4]"])
        self.assertEqual(item["3"]["pasal_terkait"][0]["law_id"], "law_3")

    def test_bentuk_item_dan_pemetaan_ke_legalqa(self):
        hasil = self.r.cari_hybrid("tes", 10)
        for i in hasil:
            self.assertEqual(i["labels"], "LegalQA")
            self.assertEqual(i["source_ids"], [i["node_id"]])
            self.assertEqual(i["text"], i["expanded_text"])
        ids = [i["node_id"] for i in hasil]
        self.assertEqual(len(ids), len(set(ids)))  # tanpa duplikat
        self.assertIn("2", ids)  # dicapai lewat anchor Law (law_2) maupun LegalQA langsung

    def test_urutan_menurun_menurut_skor(self):
        skor = [i["score"] for i in self.r.cari_hybrid("tes", 10)]
        self.assertEqual(skor, sorted(skor, reverse=True))

    def test_ulasan_kosong_memakai_question_atau_string_kosong(self):
        item = {i["node_id"]: i for i in self.r.cari_hybrid("tes", 10)}
        self.assertEqual(item["4"]["text"], "[ULASAN / JAWABAN]: Pertanyaan empat")


class TestCariDenganInfo(unittest.TestCase):
    def test_validasi_input(self):
        r = buat_retriever(KGTiruan())
        for bad in ("", "   ", None, 123):
            with self.assertRaises(ValueError):
                r.cari_dengan_info(bad, 5)
        for bad_k in (0, -1, 2.5, True, "5"):
            with self.assertRaises(ValueError):
                r.cari_dengan_info("tes", bad_k)

    def test_info_normal(self):
        h = buat_retriever(KGTiruan()).cari_dengan_info("  pertanyaan  ", 5)
        self.assertGreater(len(h.items), 0)
        self.assertGreaterEqual(h.waktu_detik, 0)
        self.assertEqual((h.peringatan, h.indeks_dilewati), ([], []))

    def test_indeks_dilewati_dilaporkan(self):
        h = buat_retriever(KGTiruan(gagal=["law_vector"])).cari_dengan_info("tes", 5)
        self.assertEqual(h.indeks_dilewati, ["law_vector"])
        self.assertTrue(any("law_vector" in p for p in h.peringatan))
        # peringatan tidak terbawa ke panggilan berikutnya yang sehat
        r = buat_retriever(KGTiruan())
        r._peringatan_terakhir = ["sisa"]
        self.assertEqual(r.cari_dengan_info("tes", 5).peringatan, [])

    def test_hasil_kosong_diberi_peringatan(self):
        h = buat_retriever(KGTiruan(vec={})).cari_dengan_info("tes", 5)
        self.assertEqual(h.items, [])
        self.assertTrue(any("Tidak ada konteks" in p for p in h.peringatan))

    def test_semua_indeks_gagal_pesan_aman(self):
        r = buat_retriever(KGTiruan(gagal=["legalqa_vector", "law_vector"]))
        with self.assertRaises(RetrievalError) as cm:
            r.cari_dengan_info("tes", 5)
        self.assertNotIn("abcd1234", str(cm.exception) + cm.exception.detail_teknis)

    def test_galat_encoder_dibungkus_dan_aman(self):
        class Rusak:
            def encode(self, *a, **k):
                raise OSError("model tidak ditemukan di C:\\rahasia\\path")

        r = HybridGraphRAGRetriever(buat_koneksi(KGTiruan()), encoder=Rusak())
        with self.assertRaises(RetrievalError) as cm:
            r.cari_dengan_info("tes", 5)
        self.assertNotIn("rahasia", str(cm.exception))  # path lokal tidak bocor ke pengguna

    def test_galat_sementara_dicoba_ulang(self):
        class ServiceUnavailable(Exception):
            pass

        kg = KGTiruan()
        n = {"i": 0}

        def router(q, **p):
            if "db.index.vector.queryNodes" in q and n["i"] < 1:
                n["i"] += 1
                return ServiceUnavailable("putus")
            return kg(q, **p)

        h = buat_retriever(router).cari_dengan_info("tes", 5)
        self.assertGreater(len(h.items), 0)
        self.assertEqual(h.indeks_dilewati, [])  # pulih lewat retry, indeks tidak dilewati


if __name__ == "__main__":
    unittest.main()
