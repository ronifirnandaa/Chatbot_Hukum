"""Validasi retriever pada Neo4j SUNGGUHAN (dijalankan oleh Anda, bukan bagian dari uji otomatis).

Membandingkan hasil website dengan hasil smoke test asli di notebook retriever
(sel "3. SMOKE TEST", pertanyaan tentang pengurus yayasan filantropi, top_k=5).

Jalankan dari folder project:
    pip install sentence-transformers      # sekali saja; unduhan awal cukup besar (torch + model +-470 MB)
    python tests/cek_retriever.py

Kode keluar: 0 = sama dengan notebook, 1 = gagal konfigurasi/koneksi/retrieval, 2 = ADA PERBEDAAN dengan notebook.
Skrip tidak mencetak URI, username, maupun password.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, load_neo4j_settings  # noqa: E402
from src.retriever.hybrid_retriever import (  # noqa: E402
    HybridGraphRAGRetriever,
    RetrievalError,
    muat_encoder,
)
from src.retriever.neo4j_connection import KoneksiNeo4jError, Neo4jConnection  # noqa: E402

PERTANYAAN = (
    "Apa pasal yang bisa menjerat pengurus yayasan filantropi yang menyalahgunakan dana donasi? "
    "Apakah penyelewengan atau penyalahgunaan dana donasi yayasan swasta itu bisa dikategorikan sebagai korupsi? "
    "Apakah masyarakat dapat melaporkan kasus tersebut kepada KPK?"
)
TOP_K = 5

# Nilai ACUAN dari output tersimpan notebook retriever (smoke test, top_k=5)
ACUAN_NODE_ID = ["1", "807", "340", "276", "973"]
ACUAN_SKOR = [0.9548, 0.8433, 0.8078, 0.8028, 0.7835]       # dibulatkan 4 desimal seperti di notebook
ACUAN_PANJANG_TEKS = [10495, 8541, 5664, 6128, 7943]
TOLERANSI_SKOR = 0.0006                                      # selisih pembulatan 4 desimal


def main() -> int:
    try:
        settings = load_neo4j_settings()
    except ConfigError as e:
        print("[GAGAL] Konfigurasi:", e)
        return 1

    print("Memuat model embedding (pertama kali bisa lama karena mengunduh)...")
    try:
        encoder = muat_encoder()
    except Exception as e:  # noqa: BLE001
        print("[GAGAL] Model embedding tidak dapat dimuat:", type(e).__name__)
        print("        Pastikan 'pip install sentence-transformers' berhasil dan ada koneksi internet.")
        return 1

    try:
        with Neo4jConnection(settings) as koneksi:
            koneksi.pastikan_kg_valid()
            retriever = HybridGraphRAGRetriever(koneksi, encoder=encoder)
            hasil = retriever.cari_dengan_info(PERTANYAAN, TOP_K)
            urutan_ulang = [
                [str(i["node_id"]) for i in retriever.cari_dengan_info(PERTANYAAN, TOP_K).items] for _ in range(2)
            ]
    except (KoneksiNeo4jError, RetrievalError, ValueError) as e:
        print("[GAGAL]", e)
        return 1

    items = hasil.items
    print("\nWaktu retrieval: %.2f detik | hasil: %d | peringatan: %s" % (hasil.waktu_detik, len(items), hasil.peringatan or "-"))
    print("\n%-3s %-8s %-8s %-9s | %-8s %-8s %-9s | %s" % ("#", "node_id", "skor", "panjang", "(nb)id", "(nb)skor", "(nb)pjg", "status"))

    beda = []
    for i in range(max(len(items), len(ACUAN_NODE_ID))):
        if i < len(items):
            it = items[i]
            nid, skor, pjg = str(it["node_id"]), float(it["score"]), len(it["text"] or "")
        else:
            nid, skor, pjg = "-", float("nan"), -1
        a_id = ACUAN_NODE_ID[i] if i < len(ACUAN_NODE_ID) else "-"
        a_skor = ACUAN_SKOR[i] if i < len(ACUAN_SKOR) else float("nan")
        a_pjg = ACUAN_PANJANG_TEKS[i] if i < len(ACUAN_PANJANG_TEKS) else -1
        ok_id = nid == a_id
        ok_skor = abs(skor - a_skor) <= TOLERANSI_SKOR if (skor == skor and a_skor == a_skor) else False
        ok_pjg = pjg == a_pjg
        status = "OK" if (ok_id and ok_skor and ok_pjg) else "BEDA"
        if status == "BEDA":
            beda.append(i + 1)
        print("%-3d %-8s %-8.4f %-9d | %-8s %-8.4f %-9d | %s" % (i + 1, nid, skor, pjg, a_id, a_skor, a_pjg, status))

    print("\nJumlah pasal terkait per hasil:", [len(it["pasal_terkait"]) for it in items])
    awal_ok = all((it["text"] or "").startswith("[ULASAN / JAWABAN]: ") for it in items)
    print("Semua teks diawali '[ULASAN / JAWABAN]: ':", "ya" if awal_ok else "TIDAK")

    ids_pertama = [str(i["node_id"]) for i in items]
    stabil = all(u == ids_pertama for u in urutan_ulang)
    print("Urutan hasil sama pada 3 kali pemanggilan berturut-turut:", "ya" if stabil else "TIDAK (urutan tidak deterministik)")

    if beda or not stabil:
        print("\nHASIL: ADA PERBEDAAN dengan notebook (baris: %s). Kirimkan keluaran ini apa adanya." % (beda or "-"))
        return 2
    print("\nHASIL: SAMA dengan hasil notebook (id, skor, dan panjang teks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
