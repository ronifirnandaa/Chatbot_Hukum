"""SALINAN PERSIS (verbatim) kelas HybridGraphRAGRetriever dari sel 2 notebook retriever
(notebook6eb1275c39.ipynb). Hanya dipakai sebagai REFERENSI pada uji kesetaraan
(tests/test_hybrid_retriever.py). Jangan diedit dan jangan diimpor oleh kode aplikasi.
"""
from contextlib import contextmanager

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase


class HybridGraphRAGRetriever:
    """Hybrid retriever untuk skema NEO4J BARU (LegalQA <-> Law via REFERENCES).

    Evidence UTAMA = node LegalQA (ulasan/jawaban hukumonline).
    Vector search: legalqa_vector + law_vector.
    Anchor Law -> naik via (l:Law)<-[:REFERENCES]-(q:LegalQA) untuk ambil ulasan.
    Output per item: node_id=q.answer_id, source_ids=[q.answer_id], text=q.ulasan.
    """

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

    def __init__(self, uri, user, password, database="neo4j",
                 model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device=None):
        self.uri = uri
        self.database = database or "neo4j"
        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                notifications_disabled_classifications=["DEPRECATION"],
            )
        except Exception:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.encoder = SentenceTransformer(model_name, device=device)
        self.vector_indexes = [
            "legalqa_vector",
            "law_vector",
        ]

    @contextmanager
    def _sesi(self):
        s = self.driver.session(database=self.database)
        try:
            yield s
        finally:
            s.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def _vector_search_multi_index(self, embedding, top_k=5):
        """Vector search di legalqa_vector + law_vector; index yang tidak ada dilewati."""
        anchors = []
        gagal = {}

        for index_name in self.vector_indexes:
            try:
                with self._sesi() as s:
                    rows = s.run(
                        self.QUERY_VECTOR,
                        index_name=index_name,
                        k=top_k,
                        embedding=embedding,
                    ).data()
            except Exception as e:
                gagal[index_name] = str(e)
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
            raise RuntimeError(
                "Semua index vektor gagal diakses, tidak ada evidence yang bisa diambil. "
                + " | ".join("%s -> %s" % (k, v[:200]) for k, v in gagal.items())
                + "\ncek: index dibuat di database yang sama? (lihat konstruksi_KG_ontology.ipynb)"
            )
        for k, v in gagal.items():
            print("[warn] index dilewati: %s -> %s" % (k, v[:200]))

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

    def cari_hybrid(self, pertanyaan, top_k_vektor=5):
        """
        Hybrid retrieval skema baru (LegalQA/Law):
        1. Vector search di legalqa_vector + law_vector.
        2. Anchor Law dinaikkan ke node LegalQA (ulasan) via REFERENCES.
        3. Output: node_id=q.answer_id, source_ids=[q.answer_id], text=q.ulasan.
        """
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

        with self._sesi() as session:
            rows = session.run(
                self.QUERY_TRAVERSAL,
                anchors=anchor_payload,
                limit=top_k_vektor * 2,
            ).data()

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
