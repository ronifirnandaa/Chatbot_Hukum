"""Bandingkan gaya jawaban 'ringkas' dan 'naratif' pada pertanyaan yang sama (dijalankan oleh Anda; 2 panggilan kuota).

Retrieval dijalankan sekali, konteks sama untuk kedua gaya. Untuk tiap gaya dicatat: jumlah kata, waktu, kutipan hukum
yang tidak ada di konteks (pemeriksaan kasar), dan - khusus 'ringkas' - apakah model mengikuti format yang diminta.
Jawaban lengkap disimpan di hasil_perbandingan_gaya/. Jawaban 'ringkas' juga dicetak langsung.

    python tests/bandingkan_gaya.py [id_model]
    python tests/bandingkan_gaya.py --pertanyaan "Apa ancaman pidana penggelapan?"
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, load_neo4j_settings  # noqa: E402
from src.generator.gaya_jawaban import jumlah_kata, periksa_format_ringkas  # noqa: E402
from src.generator.llm_client import GenerasiError, KuotaHabisError  # noqa: E402
from src.generator.llm_factory import buat_llm_dari_konfigurasi  # noqa: E402
from src.generator.model_registry import model_terkonfigurasi  # noqa: E402
from src.generator.pemeriksa_kutipan import periksa_kutipan  # noqa: E402
from src.generator.rag_service import hasilkan_jawaban  # noqa: E402
from src.retriever.hybrid_retriever import HybridGraphRAGRetriever, RetrievalError, muat_encoder  # noqa: E402
from src.retriever.neo4j_connection import KoneksiNeo4jError, Neo4jConnection  # noqa: E402

PERTANYAAN = (
    "Apa pasal yang bisa menjerat pengurus yayasan filantropi yang menyalahgunakan dana donasi? "
    "Apakah penyelewengan atau penyalahgunaan dana donasi yayasan swasta itu bisa dikategorikan sebagai korupsi? "
    "Apakah masyarakat dapat melaporkan kasus tersebut kepada KPK?"
)
FOLDER_HASIL = Path(__file__).resolve().parent.parent / "hasil_perbandingan_gaya"
GAYA = ("ringkas", "naratif")


def bandingkan_gaya(llm, model_id, items, pertanyaan, sleep=time.sleep, folder=None, tunggu_kuota=65,
                    catat=print, mode_potong=None):
    """Jalankan kedua gaya pada konteks yang sama; kembalikan list baris hasil."""
    baris = []
    for gaya in GAYA:
        t0 = time.perf_counter()
        try:
            try:
                j = hasilkan_jawaban(pertanyaan, items, model_id, llm, mode_potong, gaya=gaya)
            except KuotaHabisError:
                catat("  [%s] kuota habis (429); menunggu %d detik lalu mencoba sekali lagi..." % (gaya, tunggu_kuota))
                sleep(tunggu_kuota)
                j = hasilkan_jawaban(pertanyaan, items, model_id, llm, mode_potong, gaya=gaya)
            p = periksa_kutipan(j.jawaban, j.konteks.teks)
            baris.append({
                "gaya": gaya, "jawaban": j.jawaban, "kata": jumlah_kata(j.jawaban),
                "detik": time.perf_counter() - t0, "normal": j.selesai_normal,
                "kutipan_hilang": p.ringkas() if p.n_tidak_ditemukan else "-",
                "format": ("sesuai" if not periksa_format_ringkas(j.jawaban) else "; ".join(periksa_format_ringkas(j.jawaban)))
                if gaya == "ringkas" else "(n/a)",
                "galat": "",
            })
            if folder is not None:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / ("%s.txt" % gaya)).write_text(j.jawaban, encoding="utf-8")
        except GenerasiError as e:
            baris.append({"gaya": gaya, "jawaban": "", "kata": 0, "detik": time.perf_counter() - t0, "normal": False,
                          "kutipan_hilang": "-", "format": "-", "galat": e.detail_teknis or str(e)})
        sleep(3)
    return baris


def cetak_tabel(baris, cetak=print):
    cetak("\n%-9s %6s %7s %-7s %-14s %s" % ("gaya", "kata", "detik", "normal", "format ringkas", "kutipan yang tidak ada di konteks"))
    for b in baris:
        cetak("%-9s %6d %7.1f %-7s %-14s %s" % (b["gaya"], b["kata"], b["detik"], "ya" if b["normal"] else "TIDAK",
                                                 b["format"][:14], b["galat"] and ("GALAT: " + b["galat"][:80]) or b["kutipan_hilang"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Bandingkan gaya jawaban ringkas vs naratif.")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--pertanyaan", default=PERTANYAAN)
    args = ap.parse_args()
    try:
        llm = buat_llm_dari_konfigurasi()
        ada = llm.daftar_model()
    except (ConfigError, GenerasiError) as e:
        print("[GAGAL]", e)
        return 1
    except ImportError:
        print("[GAGAL] Paket penyedia belum terpasang (mis. 'pip install google-genai').")
        return 1
    tersedia = [m.kunci for m in model_terkonfigurasi() if m.kunci in ada]
    model_id = args.model or (tersedia[0] if tersedia else None)
    if not model_id or model_id not in ada:
        print("[GAGAL] Model tidak tersedia.")
        return 1

    print("Memuat model embedding dan terhubung ke Neo4j...")
    try:
        encoder = muat_encoder()
        with Neo4jConnection(load_neo4j_settings()) as koneksi:
            koneksi.pastikan_kg_valid()
            items = HybridGraphRAGRetriever(koneksi, encoder=encoder).cari_dengan_info(args.pertanyaan, 5).items
    except (ConfigError, KoneksiNeo4jError, RetrievalError) as e:
        print("[GAGAL]", e)
        return 1

    print("Model: %s | %d dokumen. Menjalankan 2 gaya..." % (model_id, len(items)))
    baris = bandingkan_gaya(llm, model_id, items, args.pertanyaan, folder=FOLDER_HASIL)
    cetak_tabel(baris)
    ringkas = next((b for b in baris if b["gaya"] == "ringkas" and b["jawaban"]), None)
    if ringkas:
        print("\n" + "=" * 70 + "\nJAWABAN GAYA RINGKAS\n" + "=" * 70 + "\n" + ringkas["jawaban"] + "\n" + "=" * 70)
    print("\nJawaban kedua gaya tersimpan di: %s" % FOLDER_HASIL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
