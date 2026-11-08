# Chatbot Tanya Jawab Hukum Pidana Indonesia

Hybrid Retriever (Neo4j) + LLM, dengan halaman evaluasi hasil penelitian. Dibangun bertahap dari tiga notebook penelitian.

> Status: **tahap 1-5 selesai (+ gaya jawaban ringkas)** (kerangka project, konfigurasi, koneksi Neo4j, Hybrid Retriever, Generator, halaman Chatbot). Halaman evaluasi menyusul.

## Menjalankan secara lokal

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install streamlit neo4j sentence-transformers google-genai   # sentence-transformers memasang torch (besar)
```

Isi kredensial (pilih salah satu cara; jangan pernah di-commit):

* `.streamlit/secrets.toml` (salin dari `secrets.toml.example`), atau
* `.env` (salin dari `.env.example`), atau
* environment variable.

```bash
python -m unittest discover -s tests -v      # uji offline (tanpa database)
python tests/cek_koneksi_neo4j.py            # cek koneksi SUNGGUHAN ke Neo4j
python tests/cek_retriever.py                # validasi retriever vs hasil notebook (butuh Neo4j + model embedding)
python tests/cek_generator.py                # uji end-to-end: retrieval -> Gemma 4 -> jawaban (memakai sedikit kuota gratis)
streamlit run app.py                         # beranda (status sistem); halaman Chatbot ada di sidebar
```

## Keamanan

* Tidak ada kredensial di source code. `.gitignore` mengecualikan `.env` dan `.streamlit/secrets.toml`.
* Pesan error ke pengguna tidak memuat URI, username, atau password.

## Catatan generator

* Prompt (system prompt, few-shot, format pertanyaan) dan parameter (temperature 0, maks 1024 token) disalin persis dari
  notebook generator; kesetaraannya diuji otomatis terhadap salinan kode notebook (`tests/referensi/`).
* Model berjalan lewat Google Gemini API (tier gratis, Gemma 4). Model yang diteliti di notebook (`qwen3.5:4b`, Ollama)
  tidak dilayani website ini; model di chatbot ditandai "belum dievaluasi" karena belum punya hasil RAGAS.
* `MODE_POTONG=eksperimen` meniru pemotongan konteks notebook persis; `disempurnakan` (default) menjaga daftar
  [PASAL TERKAIT] tetap masuk prompt. Sumber hukum di UI selalu berasal dari hasil retriever lengkap.
* Pertanyaan yang diajukan dan konteks hukum yang ditemukan dikirim ke API Google; pada tier gratis, Google dapat
  memakainya untuk meningkatkan produknya. Tampilkan catatan ini kepada pengguna.

## Halaman Chatbot

* Pilih model di sidebar, ketik pertanyaan, dan jawaban tampil bersama **sumber hukum** (dokumen, skor retrieval, pasal terkait)
  serta **konteks yang dikirim ke model** (bagian yang dapat dibuka/ditutup).
* Setiap pertanyaan diproses **mandiri**; riwayat hanya ditampilkan di layar selama sesi, bukan memori sistem.
* Pembatas laju (per sesi dan global) melindungi kuota API gratis yang dipakai bersama semua pengunjung.
* Pemeriksa kutipan memberi peringatan bila jawaban menyebut pasal/peraturan/putusan yang tidak ada di konteks. Ini
  pemeriksaan teks kasar untuk membantu pengguna waspada, bukan penilaian kebenaran hukum.
* Uji otomatis halaman memakai Streamlit tiruan (`tests/test_halaman_chatbot.py`); tampilan sebenarnya harus dicek
  dengan menjalankan aplikasi.

## Gaya jawaban

Sidebar menyediakan dua gaya. Hanya format yang berbeda; aturan kejujuran (hanya dari konteks, tidak mengarang pasal,
pesan baku bila konteks kosong) sama pada keduanya.

* **Ringkas (default)**: varian untuk website (`src/generator/gaya_jawaban.py`): *Jawaban singkat*, *Dasar hukum*,
  *Hal penting*, *Catatan*. **Bukan dari notebook dan belum dievaluasi dengan RAGAS.** Tanpa contoh few-shot.
* **Naratif**: prompt penelitian, disalin persis dari notebook generator (`src/generator/prompt_template.py`) dan
  dijaga identik oleh uji otomatis. Gunakan gaya ini bila ingin mendekati kondisi eksperimen.

Catatan penelitian: contoh few-shot pada prompt naratif asli memuat penjelasan di luar konteks contohnya sendiri
(daya paksa fisik/psikis, *ultimum remedium*), padahal aturan prompt menyuruh menjawab hanya dari konteks.

Bandingkan kedua gaya pada model sungguhan: `python tests/bandingkan_gaya.py` (atau `--pertanyaan "..."`).
Ganti gaya awal lewat secret `GAYA_JAWABAN`.

## Penyedia model (Groq dan/atau Google)

Model dikenali dengan kunci `penyedia:id` (mis. `groq:qwen/qwen3.8-27b`). Isi `GROQ_API_KEY` dan/atau `GOOGLE_API_KEY`;
dropdown menampilkan model dari penyedia yang kuncinya terisi dan benar-benar terdaftar di akunnya. Daftar model dapat
diganti lewat `GENERATOR_MODELS` (dipisah koma; entri tanpa awalan dianggap Google).

* **Groq** (`src/generator/groq_client.py`, API kompatibel OpenAI): Qwen mematikan penalaran (`reasoning_effort=none`);
  GPT-OSS tidak bisa mematikan penalaran sehingga diatur `low`, disembunyikan, dan batas keluaran ditambah 512 token.
* **Google** (`src/generator/llm_client.py`): mode berpikir Gemma 4 dimatikan (`thinking_level=minimal`).
* Batas tier gratis tidak sama dan dapat berubah. Menurut sumber sekunder (Okt 2026), Groq gratis dibatasi sekitar 8.000
  token/menit dan 200.000 token/hari per model; satu pertanyaan di sistem ini memakai sekitar 4.000-5.500 token, sehingga
  kapasitasnya hanya puluhan pertanyaan per hari. Halaman Chatbot menampilkan jumlah token tiap jawaban untuk memantau ini.
* Model di chatbot belum memiliki hasil evaluasi RAGAS (evaluasi pada penelitian memakai qwen3.5:4b lewat Ollama).
