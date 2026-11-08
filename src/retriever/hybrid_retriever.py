"""Hybrid retriever (LegalQA <-> Law via REFERENCES) — diekstrak dari notebook retriever.

TIDAK DIUBAH dari notebook (sel 2):
  * QUERY_VECTOR dan QUERY_TRAVERSAL (Cypher),
  * pemakaian dua indeks vektor: legalqa_vector + law_vector,
  * deduplikasi anchor berdasarkan (label, law_id/answer_id) dengan skor tertinggi, lalu dipotong top_k,
  * traversal: anchor Law dinaikkan ke LegalQA lewat REFERENCES, LIMIT = 2 * top_k,
  * pembentukan teks "[ULASAN / JAWABAN]: ..." + "[PASAL TERKAIT]:\n- ...",
  * bentuk dan isi setiap item hasil (node_id, source_ids, text, score, pasal_terkait, ...).

BERBEDA dari notebook (hanya infrastruktur, bukan algoritma):
  * memakai `Neo4jConnection` bersama (satu driver untuk seluruh aplikasi), bukan membuat driver sendiri,
  * query dijalankan lewat `Neo4jConnection.jalankan` (ada percobaan ulang untuk galat sementara),
  * `print("[warn]...")` diganti logging, dan peringatan dikembalikan ke pemanggil (untuk ditampilkan di UI),
  * encoder boleh disuntikkan (untuk uji) dan dimuat secara malas,
  * `cari_dengan_info()` menambah validasi input, pengukuran waktu, dan daftar peringatan.

`cari_hybrid()` mempertahankan nama, argumen, dan keluaran yang sama dengan notebook.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.config import INDEKS_VEKTOR_WAJIB, K_DEFAULT, MODEL_EMBEDDING
from src.retriever.neo4j_connection import KoneksiNeo4jError, Neo4jConnection

log = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Retrieval gagal. `str(e)` aman ditampilkan ke pengguna; `detail_teknis` hanya untuk log."""

    def __init__(self, pesan_pengguna: str, detail_teknis: str = ""):
        super().__init__(pesan_pengguna)
        self.detail_teknis = detail_teknis


@dataclass
class HasilRetrieval:
    items: list
    waktu_detik: float
    peringatan: list = field(default_factory=list)
    indeks_dilewati: list = field(default_factory=list)


def muat_encoder(model_name: str = MODEL_EMBEDDING, device=None):
    """Muat SentenceTransformer (impor malas). Di Streamlit, bungkus dengan st.cache_resource."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class HybridGraphRAGRetriever:
    """Hybrid retriever untuk skema NEO4J (LegalQA <-> Law via REFERENCES).

    Evidence UTAMA = node LegalQA (ulasan/jawaban hukumonline).
    Vector search: legalqa_vector + law_vector.
    Anchor Law -> naik via (l:Law)<-[:REFERENCES]-(q:LegalQA) untuk ambil ulasan.
    Output per item: node_id=q.answer_id, source_ids=[q.answer_id], text=q.ulasan (+ pasal terkait).
    """

    # --- Cypher: SALINAN PERSIS dari notebook ------------------------------------------------
    QUERY_VECTOR = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node, score
        RETURN
            labels(node)[0] AS label,
            coalesce(node.answer_id, null) AS answer_id,
            coalesce(node.law_id, null) AS law_id,
            score AS score
        """

    QUERY_TRAVERSAL = """
        UNWIND $anchors AS a
        OPTIONAL MATCH (q:LegalQA)
        WHERE a.answer_id IS NOT NULL
          AND toString(q.answer_id) = toString(a.answer_id)
        WITH a, q
        OPTIONAL MATCH (qp:LegalQA)-[:REFERENCES]->(ln:Law)
        WHERE a.law_id IS NOT NULL
          AND ln.law_id = a.law_id
        WITH DISTINCT a, coalesce(q, qp) AS ulasan
        WHERE ulasan IS NOT NULL
        OPTIONAL MATCH (ulasan:LegalQA)-[:REFERENCES]->(pasal:Law)
        RETURN
            ulasan.answer_id AS node_id,
            coalesce(ulasan.ulasan, ulasan.question, "") AS text,
            a.score AS score,
            [x IN collect(DISTINCT {law_id: pasal.law_id,
                                    pasal: pasal.pasal,
                                    judul: pasal.judul})
             WHERE x.law_id IS NOT NULL | x] AS pasal_terkait
        ORDER BY score DESC
        LIMIT $limit
        """

    def __init__(self, koneksi: Neo4jConnection, encoder=None, model_name: str = MODEL_EMBEDDING, device=None):
        self.koneksi = koneksi
        self.encoder = encoder if encoder is not None else muat_encoder(model_name, device)
        self.vector_indexes = list(INDEKS_VEKTOR_WAJIB)  # ["legalqa_vector", "law_vector"]
        self._peringatan_terakhir: list = []

    # ------------------------------------------------------------------ vector search
    def _vector_search_multi_index(self, embedding, top_k=5):
        """Vector search di legalqa_vector + law_vector; index yang tidak ada dilewati."""
        anchors = []
        gagal = {}

        for index_name in self.vector_indexes:
            try:
                rows = self.koneksi.jalankan(
                    self.QUERY_VECTOR,
                    index_name=index_name,
                    k=top_k,
                    embedding=embedding,
                )
            except KoneksiNeo4jError as e:
                gagal[index_name] = e.detail_teknis or str(e)
                continue

            for row in rows:
                anchors.append({
                    "label": row["label"],
                    "answer_id": row["answer_id"],
                    "law_id": row["law_id"],
                    "score": float(row["score"]),
                    "index": index_name,
                })

        if len(gagal) == len(self.vector_indexes):
            raise RetrievalError(
                "Pencarian gagal: indeks vektor di database tidak dapat diakses.",
                " | ".join("%s -> %s" % (k, v[:200]) for k, v in gagal.items()),
            )
        for k, v in gagal.items():
            log.warning("index dilewati: %s -> %s", k, v[:200])
        self._peringatan_terakhir = [
            "Indeks '%s' tidak dapat diakses sehingga dilewati; hasil pencarian mungkin kurang lengkap." % k
            for k in gagal
        ]

        # Deduplikasi berdasarkan (label, id), ambil score tertinggi
        best = {}
        for item in anchors:
            node_id = item["law_id"] or item["answer_id"]
            if node_id is None:
                continue
            key = (item["label"], str(node_id))
            if key not in best or item["score"] > best[key]["score"]:
                best[key] = item
        anchors = sorted(best.values(), key=lambda x: x["score"], reverse=True)
        return anchors[:top_k]

    # ------------------------------------------------------------------ retrieval utama
    def cari_hybrid(self, pertanyaan, top_k_vektor=5):
        """
        Hybrid retrieval skema LegalQA/Law:
        1. Vector search di legalqa_vector + law_vector.
        2. Anchor Law dinaikkan ke node LegalQA (ulasan) via REFERENCES.
        3. Output: node_id=q.answer_id, source_ids=[q.answer_id], text=q.ulasan (+ pasal terkait).
        """
        self._peringatan_terakhir = []
        embedding = self.encoder.encode(
            pertanyaan,
            show_progress_bar=False
        ).tolist()

        anchors = self._vector_search_multi_index(
            embedding=embedding,
            top_k=top_k_vektor
        )

        if not anchors:
            return []

        anchor_payload = [
            {
                "answer_id": item.get("answer_id"),
                "law_id": item.get("law_id"),
                "score": float(item["score"]),
            }
            for item in anchors
        ]

        rows = self.koneksi.jalankan(
            self.QUERY_TRAVERSAL,
            anchors=anchor_payload,
            limit=top_k_vektor * 2,
        )

        results = []
        sudah = set()

        def _add_text(parts, prefix, value):
            if value is None:
                return
            if isinstance(value, str):
                v = value.strip()
                if v:
                    parts.append(f"{prefix}{v}")
            elif isinstance(value, list):
                vals = [str(x) for x in value if str(x).strip()]
                if len(vals) == 1:
                    parts.append(f"{prefix}{vals[0]}")
                elif vals:
                    parts.append(f"{prefix}\n- " + "\n- ".join(vals))

        for row in rows:
            aid = row.get("node_id")
            if aid is None:
                continue
            kunci = str(aid)
            if kunci in sudah:
                continue
            sudah.add(kunci)

            teks = row.get("text")
            pasal_terkait = []
            for p in (row.get("pasal_terkait") or []):
                law_id = str(p.get("law_id") or "").strip()
                pasal = str(p.get("pasal") or "").strip()
                judul = str(p.get("judul") or "").strip()
                if not (law_id or pasal or judul):
                    continue
                if pasal and judul:
                    label = "%s (%s) [%s]" % (pasal, judul, law_id)
                elif pasal:
                    label = "%s [%s]" % (pasal, law_id)
                else:
                    label = "[%s]" % (law_id or judul)
                pasal_terkait.append({
                    "law_id": law_id,
                    "pasal": p.get("pasal"),
                    "judul": p.get("judul"),
                    "label": label,
                })

            expanded_parts = []
            _add_text(expanded_parts, "[ULASAN / JAWABAN]: ", teks)
            if pasal_terkait:
                daftar_pasal = [p["label"] for p in pasal_terkait]
                expanded_parts.append("[PASAL TERKAIT]:\n- " + "\n- ".join(daftar_pasal))

            item = {
                "node_id": aid,
                "source_ids": [aid],
                "all_source_ids": [aid],
                "text": teks,
                "expanded_text": "\n\n".join(expanded_parts) or (teks or ""),
                "score": row.get("score"),
                "labels": "LegalQA",
                "sumber_hukum": [],
                "induk_konten": [],
                "rincian_ayat": [],
                "pasal_rujukan": [],
                "pasal_perujuk": [],
                "tindak_pidana": [],
                "konsep_terkait": [],
                "pasal_terkait": pasal_terkait,
            }
            item["text"] = item["expanded_text"]
            results.append(item)

        return results[:top_k_vektor]

    # ------------------------------------------------------------------ untuk website
    def cari_dengan_info(self, pertanyaan: str, top_k: int = K_DEFAULT) -> HasilRetrieval:
        """Seperti `cari_hybrid`, tetapi dengan validasi input, waktu, dan peringatan untuk UI."""
        if not isinstance(pertanyaan, str) or not pertanyaan.strip():
            raise ValueError("Pertanyaan tidak boleh kosong.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k harus bilangan bulat >= 1.")

        mulai = time.perf_counter()
        try:
            items = self.cari_hybrid(pertanyaan.strip(), top_k)
        except (RetrievalError, KoneksiNeo4jError):
            raise
        except Exception as e:  # noqa: BLE001  (mis. galat encoder)
            raise RetrievalError(
                "Terjadi kesalahan saat memproses pertanyaan.", "%s: %s" % (type(e).__name__, str(e)[:300])
            ) from None
        waktu = time.perf_counter() - mulai

        peringatan = list(self._peringatan_terakhir)
        if not items:
            peringatan.append("Tidak ada konteks hukum yang relevan ditemukan untuk pertanyaan ini.")
        return HasilRetrieval(
            items=items,
            waktu_detik=waktu,
            peringatan=peringatan,
            indeks_dilewati=[p.split("'")[1] for p in self._peringatan_terakhir],
        )
