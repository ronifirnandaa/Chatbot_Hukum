"""Validasi generator pada layanan SUNGGUHAN (dijalankan oleh Anda, bukan bagian dari uji otomatis).

Alur uji: Neo4j -> retriever -> penyusun konteks -> Google Gemini API (Gemma 4) -> jawaban.
Pertanyaan yang dipakai sama dengan smoke test di notebook.

Jalankan dari folder project:
    pip install google-genai
    python tests/cek_generator.py                      # memakai model pertama yang tersedia
    python tests/cek_generator.py groq:openai/gpt-oss-20b   # memilih model tertentu (kunci "penyedia:id")

Kebutuhan: NEO4J_* dan GROQ_API_KEY dan/atau GOOGLE_API_KEY terisi (Secrets/.env/environment), sentence-transformers terpasang.
Satu panggilan generasi memakai sedikit kuota gratis Anda.
Kode keluar: 0 = berhasil, 1 = gagal (konfigurasi/koneksi/API).
Skrip tidak mencetak kunci API, URI, username, maupun password.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, get_mode_potong, load_neo4j_settings  # noqa: E402
from src.generator.llm_client import GenerasiError  # noqa: E402
from src.generator.llm_factory import buat_llm_dari_konfigurasi  # noqa: E402
from src.generator.model_registry import model_terkonfigurasi  # noqa: E402
from src.generator.rag_service import hasilkan_jawaban  # noqa: E402
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


def main() -> int:
    # 1) Kunci API + daftar model
    try:
        mode = get_mode_potong()
        llm = buat_llm_dari_konfigurasi()
        ada = llm.daftar_model()
    except ConfigError as e:
        print("[GAGAL] Konfigurasi:", e)
        return 1
    except GenerasiError as e:
        print("[GAGAL] Tidak bisa mengambil daftar model:", e, "|", e.detail_teknis)
        return 1
    except ImportError:
        print("[GAGAL] Paket penyedia belum terpasang (mis. 'pip install google-genai').")
        return 1

    terkonfigurasi = model_terkonfigurasi()
    print("Model terkonfigurasi vs terdaftar di API Anda:")
    for m in terkonfigurasi:
        print("  %-40s %s" % (m.kunci, "TERSEDIA" if m.kunci in ada else "tidak terdaftar"))
    for nama, pesan in llm.peringatan_penyedia().items():
        print("  [PERINGATAN] penyedia %s: %s" % (nama, pesan))
    tersedia = [m for m in terkonfigurasi if m.kunci in ada]
    if not tersedia:
        print("\n[GAGAL] Tidak ada model terkonfigurasi yang terdaftar. Model yang terlihat di akun Anda:")
        print("  ", sorted(ada) or "(tidak ada)")
        print("  Isi GENERATOR_MODELS (dipisah koma, format penyedia:id) dengan salah satunya, lalu jalankan ulang.")
        return 1

    model_id = sys.argv[1] if len(sys.argv) > 1 else tersedia[0].kunci
    if model_id not in ada:
        print("\n[GAGAL] Model '%s' tidak terdaftar di API." % model_id)
        return 1

    # 2) Retrieval
    print("\nMemuat model embedding dan terhubung ke Neo4j...")
    try:
        encoder = muat_encoder()
        with Neo4jConnection(load_neo4j_settings()) as koneksi:
            koneksi.pastikan_kg_valid()
            retriever = HybridGraphRAGRetriever(koneksi, encoder=encoder)
            hasil = retriever.cari_dengan_info(PERTANYAAN, TOP_K)
    except (ConfigError, KoneksiNeo4jError, RetrievalError) as e:
        print("[GAGAL]", e)
        return 1
    print("Retrieval: %d hasil dalam %.2f detik" % (len(hasil.items), hasil.waktu_detik))

    # 3) Generasi
    print("Memanggil model '%s' (mode pemotongan: %s)..." % (model_id, mode))
    try:
        jawab = hasilkan_jawaban(PERTANYAAN, hasil.items, model_id, llm, mode)
    except GenerasiError as e:
        print("[GAGAL]", e)
        print("        detail:", e.detail_teknis)
        return 1

    k = jawab.konteks
    print("\nKonteks ke model : %d dari %d dokumen dipakai | %d -> %d karakter | dipotong: %s | pasal tidak ditampilkan: %d"
          % (k.n_dokumen_dipakai, k.n_dokumen_total, k.panjang_penuh, k.panjang_akhir,
             "ya" if k.dipotong else "tidak", k.n_pasal_tidak_ditampilkan))
    print("Waktu generasi   : %.1f detik" % jawab.waktu_generasi)
    print("Selesai normal   : %s" % ("ya" if jawab.selesai_normal else "TIDAK (terpotong batas panjang)"))
    if jawab.token_masuk is not None:
        print("Token            : %s masuk / %s keluar" % (jawab.token_masuk, jawab.token_keluar))
    google = llm.penyedia.get("google")
    if model_id.startswith("google:") and google is not None:
        print("system_instruction: %s" % ("digabung ke pesan pertama (model menolak)" if model_id[7:] in google._gabung_system else "didukung"))
    print("\n" + "=" * 70 + "\nJAWABAN (%s)\n" % model_id + "=" * 70)
    print(jawab.jawaban)
    print("=" * 70)
    print("\nPeriksa: apakah jawaban hanya memakai informasi dari konteks, dan nomor pasal yang disebut benar-benar ada di konteks?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
