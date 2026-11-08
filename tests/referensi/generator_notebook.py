"""SALINAN PERSIS (verbatim) kode generator dari notebook (code_generator.ipynb):
sel 4 (system_prompt), sel 5 (few_shots), sel 6 (user_prompt), sel 8 (potong_konteks), sel 7 (generate_jawaban_rag).
Hanya REFERENSI uji kesetaraan (tests/test_generator_prompt.py). Jangan diedit; jangan diimpor oleh kode aplikasi.
"""
import re

# Variabel global yang dibaca generate_jawaban_rag pada notebook (nilai diset oleh uji):
BACKEND = "ollama"
MAX_CONTEXT_CHARS = 12000
MAX_CONTEXT_CHARS_PER_DOC = 2500


def _ollama_chat(messages, max_new_tokens=768, temperature=0.1, do_sample=False):  # diganti oleh uji
    raise NotImplementedError


system_prompt = {
    "vector": """Anda adalah asisten hukum Indonesia yang bertugas menjawab pertanyaan MURNI berdasarkan teks dokumen yang diberikan.

PANDUAN MENJAWAB:
1. Gunakan bahasa Indonesia formal dan susun jawaban secara naratif.
2. Jawaban HANYA boleh berasal dari informasi yang tertera pada bagian KONTEKS HUKUM. 
3. DILARANG KERAS menggunakan pengetahuan bawaan Anda untuk menebak nama Undang-Undang, nomor pasal, atau sanksi pidana jika tidak tertulis secara eksplisit di dalam konteks.
4. Jika konteks hanya memuat definisi kejahatan tanpa menyebutkan hukuman/sanksinya, maka Anda harus menjawab definisi tersebut dan menyatakan bahwa informasi hukumannya tidak tersedia.
5. Jika konteks sama sekali tidak relevan dengan pertanyaan, jawab dengan: "Maaf, informasi hukum tersebut tidak tersedia dalam dokumen yang ditemukan.""",
    "t2c": """Anda adalah asisten hukum Indonesia yang bertugas menjawab pertanyaan MURNI berdasarkan teks hasil pencarian database relasional (Konteks Hukum).

PANDUAN MENJAWAB:
1. Gunakan bahasa Indonesia formal, objektif, dan susun jawaban secara naratif.
2. Jawaban HANYA boleh bersumber dari teks yang tertera pada "Konteks Hukum".
3. DILARANG menggunakan pengetahuan bawaan Anda untuk melengkapi nama Undang-Undang, nomor pasal, atau ancaman sanksi jika tidak tertulis di dalam konteks.
4. JIKA KONTEKS KOSONG: Ini berarti sistem tidak menemukan kecocokan kata kunci eksak di dalam database. Anda WAJIB menjawab: "Maaf, sistem tidak menemukan aturan hukum yang memuat kata kunci spesifik dari pertanyaan Anda di dalam basis data." Jangan mengarang jawaban apapun.
5. Jika konteks hanya memuat sebagian informasi (misalnya hanya definisi tanpa sanksi), jawab sesuai definisi tersebut dan nyatakan secara eksplisit bahwa informasi lainnya tidak tersedia di dokumen.""",
    "hybrid": """Anda adalah asisten hukum Indonesia yang ahli dalam hukum pidana dan peraturan perundang-undangan.
Anda menjawab pertanyaan hukum secara naratif, formal, dan komprehensif HANYA berdasarkan konteks hukum yang diberikan.

PANDUAN MENJAWAB:
1. Gunakan bahasa Indonesia formal dan lugas layaknya ulasan hukum profesional.
2. Strukturkan jawaban secara naratif berparagraf (hindari penggunaan bullet point jika tidak mendesak).
3. Perhatikan label semantik pada konteks (seperti [ULASAN / JAWABAN] dan [PASAL TERKAIT]) untuk merangkai jawaban.
4. Sebutkan secara eksplisit nomor pasal dan nama undang-undang yang relevan.
5. Jika terdapat sanksi, jelaskan secara lengkap (jenis, sifat, nominal, dan durasi).
6. JIKA KONTEKS KOSONG ATAU TIDAK RELEVAN, jawab dengan tegas: "Maaf, informasi hukum tersebut tidak tersedia dalam basis data saya." Jangan pernah mengarang jawaban."""
}

few_shots = {
    "vector": [
    {
        "role": "user", 
        "content": """Berikut adalah konteks hukum yang tersedia:
---
[Konteks 1]
Node ID: KONTEN-Pasal362-KUHP
Labels: ['KontenHukum']
Isi:
Barang siapa mengambil barang sesuatu, yang seluruhnya atau sebagian kepunyaan orang lain, dengan maksud untuk dimiliki secara melawan hukum, diancam karena pencurian, dengan pidana penjara paling lama lima tahun atau pidana denda paling banyak sembilan ratus rupiah.
---
Berdasarkan konteks di atas, jawablah pertanyaan hukum berikut secara naratif dan komprehensif:
Apa itu pencurian dan berapa lama ancaman penjaranya?"""
    },
    {
        "role": "assistant",
        "content": """Berdasarkan dokumen hukum yang tersedia, pencurian didefinisikan sebagai tindakan seseorang yang mengambil suatu barang, baik seluruhnya maupun sebagian yang merupakan milik orang lain, dengan maksud untuk memiliki barang tersebut secara melawan hukum. Terhadap tindak pidana pencurian tersebut, pelaku dapat diancam dengan sanksi berupa pidana penjara paling lama lima tahun atau pidana denda paling banyak sembilan ratus rupiah."""
    },
    {
        "role": "user", 
        "content": """Berikut adalah konteks hukum yang tersedia:
---
[Konteks 1]
Node ID: KONTEN-DefinisiMakar
Labels: ['KontenHukum']
Isi:
Makar adalah niat untuk melakukan suatu perbuatan yang telah diwujudkan dengan adanya permulaan pelaksanaan perbuatan tersebut.
---
Berdasarkan konteks di atas, jawablah pertanyaan hukum berikut secara naratif dan komprehensif:
Apa yang dimaksud dengan makar dan berapa sanksi pidananya?"""
    },
    {
        "role": "assistant",
        "content": """Berdasarkan dokumen yang tersedia, makar diartikan sebagai niat untuk melakukan suatu perbuatan yang telah diwujudkan melalui adanya permulaan pelaksanaan atas perbuatan tersebut. 

Namun, informasi mengenai ancaman sanksi pidana untuk tindak pidana makar tidak tersedia dalam konteks hukum yang ditemukan."""
    }
],
    "t2c": [
    {
        "role": "user", 
        "content": """Berikut adalah konteks hukum yang tersedia:
---
[Konteks 1]
Node ID: KONTEN-Pasal362-DOK-1-1946
Source IDs: ['99']
Isi:
Barang siapa mengambil barang sesuatu, yang seluruhnya atau sebagian kepunyaan orang lain, dengan maksud untuk dimiliki secara melawan hukum, diancam karena pencurian.
---
Berdasarkan konteks di atas, jawablah pertanyaan hukum berikut secara naratif dan komprehensif:
Apa itu pencurian dan berapa tahun penjaranya?"""
    },
    {
        "role": "assistant",
        "content": """Berdasarkan aturan hukum yang ditemukan, pencurian diartikan sebagai tindakan seseorang yang mengambil suatu barang, baik seluruhnya maupun sebagian yang merupakan kepunyaan orang lain, dengan maksud untuk memilikinya secara melawan hukum. 

Namun, dokumen hukum yang tersedia tidak memuat informasi mengenai ancaman hukuman atau durasi maksimal penjara bagi tindak pidana tersebut."""
    },
    {
        "role": "user", 
        "content": """Berikut adalah konteks hukum yang tersedia:
---

---
Berdasarkan konteks di atas, jawablah pertanyaan hukum berikut secara naratif dan komprehensif:
Apa ancaman pidana bagi penggunaan gps ilegal menurut undang-undang?"""
    },
    {
        "role": "assistant",
        "content": """Maaf, sistem tidak menemukan aturan hukum yang memuat kata kunci spesifik mengenai "penggunaan gps ilegal" di dalam basis data. Oleh karena itu, saya tidak dapat memberikan informasi mengenai ancaman pidananya."""
    },
    {
        "role": "user", 
        "content": """Berikut adalah konteks hukum yang tersedia:
---
[Konteks 1]
Node ID: KONTEN-Pasal1-Angka1-DOK-16-2001
Source IDs: ['120']
Isi:
Yayasan adalah badan hukum yang terdiri atas kekayaan yang dipisahkan dan diperuntukkan untuk mencapai tujuan tertentu di bidang sosial, keagamaan, dan kemanusiaan, yang tidak mempunyai anggota.
---
Berdasarkan konteks di atas, jawablah pertanyaan hukum berikut secara naratif dan komprehensif:
Tolong jelaskan pengertian yayasan!"""
    },
    {
        "role": "assistant",
        "content": """Berdasarkan dokumen yang tersedia, Yayasan didefinisikan sebagai sebuah badan hukum yang terdiri atas kekayaan yang dipisahkan. Kekayaan tersebut diperuntukkan khusus guna mencapai tujuan-tujuan tertentu di bidang sosial, keagamaan, dan kemanusiaan. Selain itu, yayasan memiliki karakteristik khusus yaitu tidak memiliki keanggotaan."""
    }
],
    "hybrid": [
    {
        "role": "user", 
        "content": """Berikut adalah konteks hukum yang tersedia:
---
[Konteks 1]
Node ID: 1
[ULASAN / JAWABAN]: Pasal 48 KUHP menyatakan bahwa barang siapa melakukan perbuatan karena pengaruh daya paksa, tidak dipidana. Alasan penghapus pidana ini dikenal dengan istilah overmacht atau daya paksa. Ketentuan serupa dipertahankan dalam Pasal 48 Undang-Undang Nomor 1 Tahun 2023 tentang KUHP baru.

[PASAL TERKAIT]:
- Pasal 48 (KUHP) [law_000001]
- Pasal 48 (UU 1/2023) [law_000002]
---
Berdasarkan konteks di atas, jawablah pertanyaan hukum berikut secara naratif dan komprehensif:
Bagaimana ketentuan overmacht dalam hukum pidana? Lalu, apa yang dimaksud hukum pidana sebagai ultime remedium?"""
    },
    {
        "role": "assistant",
        "content": """**Ketentuan Overmacht dalam Hukum Pidana Indonesia**

Overmacht atau daya paksa merupakan salah satu alasan penghapus pidana yang diatur dalam hukum pidana Indonesia. Secara normatif, ketentuan mengenai overmacht diatur dalam Pasal 48 Kitab Undang-Undang Hukum Pidana ("KUHP") lama yang menyatakan bahwa barang siapa melakukan perbuatan karena pengaruh daya paksa, tidak dipidana. Ketentuan serupa juga dipertahankan dalam Pasal 48 Undang-Undang Nomor 1 Tahun 2023 tentang KUHP baru sebagai pembaruan hukum pidana nasional.

Daya paksa dalam pengertian ini mencakup bentuk paksaan fisik maupun psikis yang membuat seseorang tidak lagi memiliki pilihan bebas. Dalam kondisi demikian, pelaku dianggap tidak memiliki kesalahan sehingga pertanggungjawaban pidananya hapus.

Adapun mengenai konsep *ultimum remedium*, prinsip ini menempatkan hukum pidana sebagai upaya terakhir dalam penyelesaian suatu permasalahan hukum. Artinya, penerapan sanksi pidana hanya dibenarkan apabila instrumen hukum lain dipandang tidak memadai atau tidak efektif untuk memberikan perlindungan."""
    }
]
}

def user_prompt(pertanyaan, konteks):
    return f"""Berikut adalah konteks hukum yang tersedia:
---
{konteks}
---
Pertanyaan:
{pertanyaan}"""

def potong_konteks(konteks, max_total=0, max_per_doc=0):
    """Potong konteks agar tidak OOM.

    - max_per_doc: batasi tiap blok '[Konteks N]' supaya semua dokumen tetap
      terwakili (bukan hanya dokumen awal).
    - max_total: batas total panjang akhir.
    """
    konteks = str(konteks)
    if max_per_doc and max_per_doc > 0:
        blocks = re.split(r"(?=\[Konteks\s+\d+\])", konteks)
        blocks = [b for b in blocks if b.strip()]
        if len(blocks) > 1:
            hasil = []
            for b in blocks:
                if len(b) > max_per_doc:
                    b = b[:max_per_doc] + "\n... [dokumen dipotong]"
                hasil.append(b)
            konteks = "\n".join(hasil)
    if max_total and len(konteks) > max_total:
        konteks = konteks[:max_total] + "\n... [konteks dipotong: melebihi batas total]"
    return konteks

def generate_jawaban_rag(
    tokenizer,
    model,
    pertanyaan_user,
    konteks_hukum,
    retriever='hybrid',
    max_new_tokens=768,
    temperature=0.1,
    do_sample=False,
):
    """
    Generator RAG memakai Qwen2.5-7B-Instruct.

    Catatan:
    - Output model.generate berbentuk [batch, sequence_length], jadi jawaban
      dipotong dengan outputs[0, panjang_input:].
    - Device diambil dari parameter model (aman untuk device_map="auto").
    """
    konteks_hukum = potong_konteks(
        konteks_hukum,
        max_total=globals().get("MAX_CONTEXT_CHARS", 0),
        max_per_doc=globals().get("MAX_CONTEXT_CHARS_PER_DOC", 0),
    )

    messages = [{"role": "system", "content": system_prompt[retriever]}]
    messages.extend(few_shots[retriever])
    messages.append({"role": "user", "content": user_prompt(pertanyaan_user, konteks_hukum)})

    # ---- Backend Ollama (HTTP) ----
    if globals().get("BACKEND", "transformers") == "ollama":
        return _ollama_chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    # ---- Backend transformers ----
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        device = next(model.parameters()).device
    except Exception:
        device = getattr(model, "device", "cpu")
    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": 0.9})

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    panjang_input = inputs.input_ids.shape[-1]
    jawaban_final = tokenizer.decode(
        outputs[0, panjang_input:], skip_special_tokens=True
    ).strip()

    # Bersihkan cache GPU tiap komputasi (hindari fragmentasi/OOM menumpuk).
    del inputs, outputs
    if torch.cuda.is_available() and int(globals().get("CLEAR_CACHE_EVERY", 0)):
        torch.cuda.empty_cache()
    return jawaban_final

