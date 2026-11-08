"""Diagnosis cepat galat dari Gemma 4 lewat Google API (dijalankan oleh Anda).

Menjalankan 4 skenario kecil pada tiap model terkonfigurasi (satu percobaan per skenario, tanpa retry otomatis)
untuk membedakan: (1) layanan Google sedang bermasalah, (2) pengaturan thinking_level memicu galat,
(3) jawaban kosong karena token berpikir menghabiskan batas.

Pemakaian (memakai sekitar 8 permintaan dari kuota gratis; jeda 3 detik antarpermintaan):
    python tests/diagnosa_gemma.py
Skrip tidak mencetak kunci API.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, get_google_api_key  # noqa: E402
from src.generator.llm_client import GenerasiError, GeminiLLM, buat_klien_google  # noqa: E402
from src.generator.model_registry import model_terkonfigurasi  # noqa: E402
from src.generator.prompt_template import bangun_messages  # noqa: E402

PERTANYAAN = "Apa ancaman pidana penggelapan?"
KONTEKS = (
    "[Konteks 1]\nNode ID: 1\n[ULASAN / JAWABAN]: Pasal 372 KUHP mengatur penggelapan dengan ancaman pidana "
    "penjara paling lama empat tahun atau denda.\n\n[PASAL TERKAIT]:\n- Pasal 372 (KUHP) [law_000372]"
)
KECIL = [{"role": "user", "content": "Jawab dalam satu kalimat: apa ibu kota Indonesia?"}]
NYATA = bangun_messages(PERTANYAAN, KONTEKS)

# (kode, deskripsi, pesan, batas token, thinking_level)
SKENARIO = [
    ("A", "prompt kecil, tanpa thinking_level", KECIL, 256, None),
    ("B", "prompt kecil, thinking_level=minimal", KECIL, 256, "minimal"),
    ("C", "prompt nyata (system+few-shot), minimal, 1024", NYATA, 1024, "minimal"),
    ("D", "prompt nyata, tanpa thinking_level, batas 4096", NYATA, 4096, None),
]


def jalankan(llm, model_ids, sleep=time.sleep, cetak=print):
    hasil = []
    for mid in model_ids:
        cetak("\nModel: %s" % mid)
        for kode, desk, pesan, mx, th in SKENARIO:
            t0 = time.perf_counter()
            try:
                h = llm.generate(mid, pesan, max_new_tokens=mx, thinking_level=th, percobaan=1)
                status, info = "OK", "%d karakter, finish=%s" % (len(h.teks), h.finish_reason)
            except GenerasiError as e:
                status, info = "GAGAL", (e.detail_teknis or str(e))
            dt = time.perf_counter() - t0
            hasil.append({"model": mid, "kode": kode, "status": status, "info": info})
            cetak("  [%s] %-8s %-48s %5.1fs  %s" % (kode, status, desk, dt, info[:110]))
            sleep(3)
    return hasil


def ringkas(hasil):
    """Kesimpulan per model berdasarkan pola keberhasilan skenario A-D."""
    baris = []
    for mid in dict.fromkeys(h["model"] for h in hasil):
        st = {h["kode"]: h["status"] == "OK" for h in hasil if h["model"] == mid}
        if not any(st.values()):
            s = "semua skenario gagal -> layanan Google/model sedang bermasalah (coba lagi nanti atau pakai model/penyedia lain)"
        elif all(st.values()):
            s = "semua skenario berhasil -> pengaturan sekarang aman; galat sebelumnya kemungkinan sementara"
        elif st.get("A") and st.get("D") and not (st.get("B") or st.get("C")):
            s = "thinking_level=minimal memicu galat -> perlu pengaturan lain (kirim keluaran ini ke saya)"
        elif st.get("D") and not st.get("C"):
            s = "jawaban hanya berhasil tanpa thinking_level dengan batas token besar -> mode berpikir tidak bisa dimatikan; batas token perlu dinaikkan"
        elif st.get("C"):
            s = "prompt nyata dengan thinking_level=minimal berhasil -> pengaturan sekarang bekerja"
        else:
            s = "pola campuran -> kirim keluaran ini ke saya"
        baris.append("%s: %s" % (mid, s))
    return baris


def main() -> int:
    try:
        kunci = get_google_api_key()
        llm = GeminiLLM(buat_klien_google(kunci), rahasia=(kunci,))
        ada = llm.daftar_model()
    except ConfigError as e:
        print("[GAGAL] Konfigurasi:", e)
        return 1
    except ImportError:
        print("[GAGAL] Paket 'google-genai' belum terpasang.")
        return 1
    except GenerasiError as e:
        print("[GAGAL]", e, "|", e.detail_teknis)
        return 1
    ids = [m.id for m in model_terkonfigurasi() if m.provider == "google" and m.id in ada]
    if not ids:
        print("[GAGAL] Tidak ada model Google terkonfigurasi yang terdaftar di API.")
        return 1
    hasil = jalankan(llm, ids)
    print("\nKESIMPULAN SEMENTARA:")
    for b in ringkas(hasil):
        print(" -", b)
    print("\nKirimkan seluruh keluaran ini ke saya.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
