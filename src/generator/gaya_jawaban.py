"""Gaya jawaban chatbot.

* "naratif": prompt PENELITIAN, disalin persis dari notebook (lihat prompt_template.py). Tidak diubah.
* "ringkas": varian BARU untuk website, BUKAN dari notebook dan BELUM dievaluasi dengan RAGAS. Dibuat karena jawaban
  naratif dinilai kurang langsung ke inti. Hanya format yang berubah; aturan kejujuran (hanya dari konteks, tidak
  mengarang, pesan baku bila konteks kosong) dipertahankan dan dipertegas. Tanpa contoh few-shot, agar tidak ada
  contoh jawaban yang memuat informasi di luar konteks.
"""
from __future__ import annotations

import re

from src.config import GAYA_VALID, PESAN_KONTEKS_KOSONG
from src.generator import prompt_template as pt

GAYA_NARATIF, GAYA_RINGKAS = "naratif", "ringkas"
LABEL_GAYA = {
    GAYA_RINGKAS: "Ringkas (langsung ke inti)",
    GAYA_NARATIF: "Naratif (prompt penelitian)",
}

SYSTEM_PROMPT_RINGKAS = '''Anda adalah asisten hukum Indonesia yang ahli dalam hukum pidana dan peraturan perundang-undangan.
Anda menjawab pertanyaan hukum secara singkat, jelas, dan langsung ke inti HANYA berdasarkan konteks hukum yang diberikan.

PANDUAN MENJAWAB:
1. Gunakan bahasa Indonesia yang formal, lugas, dan mudah dipahami orang awam.
2. Mulailah langsung dengan jawaban, tanpa judul, salam, atau kalimat pengantar.
3. Susun jawaban persis dengan format berikut. Hapus bagian yang tidak relevan, kecuali dua bagian pertama.

**Jawaban singkat:** satu atau dua kalimat yang langsung menjawab pertanyaan (misalnya "Ya, ...", "Tidak, ...", atau "Bergantung pada ..., yaitu ..."). Jika pertanyaan terdiri dari beberapa pertanyaan, jawab masing-masing dalam satu butir bernomor.

**Dasar hukum:**
- Pasal dan nama peraturan, diikuti isi ketentuan atau sanksinya dalam satu baris.

**Hal penting:**
- Unsur, syarat, pengecualian, atau langkah praktis yang disebutkan dalam konteks, masing-masing dalam satu butir singkat.

**Catatan:** isi hanya jika konteks tidak cukup untuk menjawab sebagian pertanyaan; sebutkan bagian mana yang informasinya tidak tersedia.

4. Perhatikan label semantik pada konteks (seperti [ULASAN / JAWABAN] dan [PASAL TERKAIT]) untuk menyusun jawaban.
5. Sebutkan nomor pasal dan nama peraturan HANYA jika tertulis dalam konteks. Jangan menambahkan pasal, peraturan, putusan, angka, atau penjelasan yang tidak ada dalam konteks.
6. Jika terdapat sanksi, tuliskan jenis, durasi, dan nominalnya persis seperti yang tertulis dalam konteks.
7. Batasi jawaban sekitar 150 sampai 250 kata, kecuali pertanyaan memang menuntut lebih banyak.
8. JIKA KONTEKS KOSONG ATAU TIDAK RELEVAN, jawab dengan tegas: "''' + PESAN_KONTEKS_KOSONG + '''" Jangan pernah mengarang jawaban.'''

_BAGIAN_WAJIB = ("**Jawaban singkat:**", "**Dasar hukum:**")


def bangun_messages_gaya(pertanyaan: str, konteks: str, gaya: str = GAYA_NARATIF) -> list:
    """Pesan untuk model sesuai gaya. `konteks` harus sudah dipotong. gaya='naratif' = persis notebook."""
    if gaya == GAYA_NARATIF:
        return pt.bangun_messages(pertanyaan, konteks)
    if gaya == GAYA_RINGKAS:
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RINGKAS},
            {"role": "user", "content": pt.user_prompt(pertanyaan, konteks)},
        ]
    raise ValueError("Gaya jawaban tidak dikenal: %r (pilihan: %s)" % (gaya, ", ".join(GAYA_VALID)))


def periksa_format_ringkas(teks: str) -> list:
    """Daftar masalah format pada jawaban gaya ringkas (kosong = sesuai). Dipakai skrip perbandingan, bukan UI."""
    t = (teks or "").strip()
    masalah = []
    if not t.startswith(_BAGIAN_WAJIB[0]):
        masalah.append("tidak diawali '%s'" % _BAGIAN_WAJIB[0])
    if PESAN_KONTEKS_KOSONG in t:  # jawaban baku konteks kosong sah tanpa bagian lain
        return []
    for b in _BAGIAN_WAJIB:
        if b not in t:
            masalah.append("bagian '%s' tidak ada" % b)
    return masalah


def jumlah_kata(teks: str) -> int:
    return len(re.findall(r"\S+", teks or ""))
