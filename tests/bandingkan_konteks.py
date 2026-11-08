"""Perbandingan panjang konteks pada pertanyaan yang sama (dijalankan oleh Anda; memakai 4 panggilan kuota gratis).

Pertanyaan: smoke test notebook. Retrieval dijalankan SEKALI; konteks yang sama dipotong dengan empat pengaturan:
  eksperimen        : persis notebook (2.500 karakter/blok, total 12.000)
  disempurnakan     : pengaturan website sekarang (ulasan 2.000, 8 pasal, total 14.000)
  luas              : ulasan 6.000, 15 pasal, total 36.000
  penuh             : tanpa pemotongan
Untuk tiap pengaturan dicatat: ukuran konteks, waktu, panjang jawaban, dan kutipan hukum pada jawaban yang TIDAK
ditemukan di konteks yang dikirim (pemeriksaan kasar; lihat src/generator/pemeriksa_kutipan.py).
Jawaban lengkap disimpan di folder hasil_perbandingan_konteks/ untuk Anda baca dan nilai sendiri.

    python tests/bandingkan_konteks.py [id_model]                  # default: model pertama yang tersedia
    python tests/bandingkan_konteks.py --hanya luas,penuh          # hanya pengaturan tertentu (hemat kuota)

Bila Google menolak dengan 429 (kuota), skrip menunggu 65 detik lalu mencoba SEKALI lagi (batas per menit biasanya
sudah pulih). Bila tetap ditolak, kemungkinan itu batas harian.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_PER_DOC,
    ConfigError,
    load_neo4j_settings,
)
from src.generator.context_builder import (  # noqa: E402
    potong_konteks_disempurnakan,
    potong_konteks_eksperimen,
    susun_konteks_penuh,
)
from src.generator.llm_client import GenerasiError, KuotaHabisError  # noqa: E402
from src.generator.llm_factory import buat_llm_dari_konfigurasi  # noqa: E402
from src.generator.model_registry import model_terkonfigurasi  # noqa: E402
from src.generator.pemeriksa_kutipan import periksa_kutipan  # noqa: E402
from src.generator.prompt_template import bangun_messages  # noqa: E402
from src.retriever.hybrid_retriever import HybridGraphRAGRetriever, RetrievalError, muat_encoder  # noqa: E402
from src.retriever.neo4j_connection import KoneksiNeo4jError, Neo4jConnection  # noqa: E402

PERTANYAAN = (
    "Apa pasal yang bisa menjerat pengurus yayasan filantropi yang menyalahgunakan dana donasi? "
    "Apakah penyelewengan atau penyalahgunaan dana donasi yayasan swasta itu bisa dikategorikan sebagai korupsi? "
    "Apakah masyarakat dapat melaporkan kasus tersebut kepada KPK?"
)
FOLDER_HASIL = Path(__file__).resolve().parent.parent / "hasil_perbandingan_konteks"


def pengaturan_konteks(items):
    """Daftar (nama, teks_konteks) untuk empat pengaturan."""
    penuh = susun_konteks_penuh(items)
    return [
        ("eksperimen", potong_konteks_eksperimen(penuh, MAX_CONTEXT_CHARS, MAX_CONTEXT_CHARS_PER_DOC)),
        ("disempurnakan", potong_konteks_disempurnakan(items)[0]),
        ("luas", potong_konteks_disempurnakan(items, 6000, 15, 36000)[0]),
        ("penuh", penuh),
    ]


def bandingkan(llm, model_id, items, pertanyaan, sleep=time.sleep, folder=None, jeda=3, tunggu_kuota=65,
               hanya=None, catat=print):
    """Jalankan pengaturan konteks; kembalikan list baris hasil (dict). Jawaban disimpan ke `folder` bila diberikan.

    `hanya`: himpunan nama pengaturan yang dijalankan (None = semua). 429 -> tunggu `tunggu_kuota` detik, ulangi sekali."""
    baris = []
    for nama, konteks in pengaturan_konteks(items):
        if hanya and nama not in hanya:
            continue
        pesan = bangun_messages(pertanyaan, konteks)
        t0 = time.perf_counter()
        ulang = False
        try:
            try:
                h = llm.generate(model_id, pesan)
            except KuotaHabisError:
                catat("  [%s] kuota habis (429); menunggu %d detik lalu mencoba sekali lagi..." % (nama, tunggu_kuota))
                ulang = True
                sleep(tunggu_kuota)
                h = llm.generate(model_id, pesan)
            dt = time.perf_counter() - t0
            p = periksa_kutipan(h.teks, konteks)
            baris.append({"nama": nama, "n_konteks": len(konteks), "detik": dt, "n_jawaban": len(h.teks),
                          "n_kutipan": p.n_dikutip, "tidak_ditemukan": p.ringkas() if p.n_tidak_ditemukan else "-",
                          "normal": h.selesai_normal, "galat": "", "ulang": ulang})
            if folder is not None:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / ("%s.txt" % nama)).write_text(h.teks, encoding="utf-8")
        except GenerasiError as e:
            baris.append({"nama": nama, "n_konteks": len(konteks), "detik": time.perf_counter() - t0, "n_jawaban": 0,
                          "n_kutipan": 0, "tidak_ditemukan": "-", "normal": False,
                          "galat": e.detail_teknis or str(e), "ulang": ulang})
        sleep(jeda)
    return baris


def cetak_tabel(baris, cetak=print):
    cetak("\n%-14s %9s %7s %9s %8s %-8s %s" % ("pengaturan", "konteks", "detik", "jawaban", "kutipan", "normal", "kutipan yang tidak ada di konteks"))
    for b in baris:
        nama = b["nama"] + ("*" if b.get("ulang") else "")
        cetak("%-14s %9d %7.1f %9d %8d %-8s %s" % (
            nama, b["n_konteks"], b["detik"], b["n_jawaban"], b["n_kutipan"],
            "ya" if b["normal"] else "TIDAK", b["galat"] and ("GALAT: " + b["galat"][:90]) or b["tidak_ditemukan"]))
    if any(b.get("ulang") for b in baris):
        cetak("* = sempat ditolak 429, lalu diulang setelah menunggu")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bandingkan panjang konteks pada pertanyaan yang sama.")
    ap.add_argument("model", nargs="?", help="kunci model penyedia:id (default: model pertama yang tersedia)")
    ap.add_argument("--hanya", help="nama pengaturan dipisah koma: eksperimen,disempurnakan,luas,penuh")
    ap.add_argument("--jeda", type=float, default=3, help="jeda antarpengaturan (detik)")
    args = ap.parse_args()
    hanya = {x.strip() for x in args.hanya.split(",") if x.strip()} if args.hanya else None
    if hanya and not hanya <= {"eksperimen", "disempurnakan", "luas", "penuh"}:
        print("[GAGAL] Nama pengaturan tidak dikenal:", ", ".join(sorted(hanya - {"eksperimen", "disempurnakan", "luas", "penuh"})))
        return 1
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
            items = HybridGraphRAGRetriever(koneksi, encoder=encoder).cari_dengan_info(PERTANYAAN, 5).items
    except (ConfigError, KoneksiNeo4jError, RetrievalError) as e:
        print("[GAGAL]", e)
        return 1

    print("Model: %s | retrieval: %d dokumen | konteks penuh: %d karakter" % (model_id, len(items), len(susun_konteks_penuh(items))))
    print("Menjalankan %s (bisa beberapa menit; 429 akan ditunggu otomatis)..." % (", ".join(sorted(hanya)) if hanya else "4 pengaturan"))
    baris = bandingkan(llm, model_id, items, PERTANYAAN, folder=FOLDER_HASIL, jeda=args.jeda, hanya=hanya)
    cetak_tabel(baris)
    print("\nJawaban lengkap tersimpan di: %s" % FOLDER_HASIL)
    print("Kirimkan tabel di atas ke saya. Baca juga paragraf terakhir tiap jawaban (soal pelaporan ke KPK).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
