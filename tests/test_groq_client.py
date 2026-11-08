"""Uji adaptor Groq dengan HTTP tiruan (offline)."""
import unittest

from src.config import GROQ_HEADROOM_REASONING
from src.generator import prompt_template as pt
from src.generator.groq_client import GroqLLM, pengaturan_reasoning
from src.generator.llm_client import GenerasiError, KuotaHabisError, ModelTidakTersediaError

KUNCI = "gsk_KUNCI-RAHASIA-1234567890"
MSG = pt.bangun_messages("Apa sanksinya?", "[Konteks 1]\nNode ID: 1\nIsi")
QWEN, GPTOSS = "qwen/qwen3.8-27b", "openai/gpt-oss-20b"


class Resp:
    def __init__(self, kode, isi, headers=None, text=""):
        self.status_code, self._isi, self.headers, self.text = kode, isi, headers or {}, text

    def json(self):
        if isinstance(self._isi, Exception):
            raise self._isi
        return self._isi


def ok(teks="Jawaban.", finish="stop", masuk=100, keluar=50, reasoning_tokens=None):
    pakai = {"prompt_tokens": masuk, "completion_tokens": keluar}
    if reasoning_tokens is not None:
        pakai["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return Resp(200, {"choices": [{"message": {"content": teks}, "finish_reason": finish}], "usage": pakai})


def galat(kode, pesan, headers=None):
    return Resp(kode, {"error": {"message": pesan}}, headers)


class FakeHTTP:
    def __init__(self, post=None, get=None):
        self.q_post, self.q_get, self.posts, self.gets = list(post or [ok()]), get, [], []

    def post(self, url, headers, json, timeout):
        self.posts.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        x = self.q_post.pop(0) if len(self.q_post) > 1 else self.q_post[0]
        if isinstance(x, Exception):
            raise x
        return x

    def get(self, url, headers, timeout):
        self.gets.append({"url": url, "headers": headers})
        if isinstance(self.q_get, Exception):
            raise self.q_get
        return self.q_get


def buat(post=None, get=None):
    http = FakeHTTP(post, get)
    tidur = []
    return GroqLLM(KUNCI, http=http, sleep=tidur.append), http, tidur


class TestPayload(unittest.TestCase):
    def test_permintaan_qwen_pesan_tidak_diubah_penalaran_dimatikan(self):
        llm, http, _ = buat()
        hasil = llm.generate(QWEN, MSG)
        p = http.posts[0]
        self.assertEqual(p["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(p["headers"]["Authorization"], "Bearer " + KUNCI)
        j = p["json"]
        self.assertEqual(j["messages"], MSG)  # system + few-shot + pertanyaan persis seperti notebook
        self.assertEqual((j["model"], j["temperature"], j["max_completion_tokens"], j["stream"]), (QWEN, 0.0, 1024, False))
        self.assertEqual(j["reasoning_effort"], "none")
        self.assertNotIn("include_reasoning", j)
        self.assertEqual((hasil.teks, hasil.selesai_normal, hasil.token_masuk, hasil.token_keluar), ("Jawaban.", True, 100, 50))

    def test_gpt_oss_penalaran_rendah_tersembunyi_dan_batas_ditambah(self):
        llm, http, _ = buat()
        llm.generate(GPTOSS, MSG)
        j = http.posts[0]["json"]
        self.assertEqual((j["reasoning_effort"], j["include_reasoning"]), ("low", False))
        self.assertEqual(j["max_completion_tokens"], 1024 + GROQ_HEADROOM_REASONING)

    def test_model_lain_tanpa_parameter_penalaran(self):
        llm, http, _ = buat()
        llm.generate("meta/llama-x", MSG)
        self.assertNotIn("reasoning_effort", http.posts[0]["json"])
        self.assertEqual(pengaturan_reasoning("meta/llama-x"), {})

    def test_parameter_generasi_dapat_diubah(self):
        llm, http, _ = buat()
        llm.generate(QWEN, MSG, max_new_tokens=300, temperature=0.5)
        self.assertEqual((http.posts[0]["json"]["max_completion_tokens"], http.posts[0]["json"]["temperature"]), (300, 0.5))


class TestRespons(unittest.TestCase):
    def test_tag_think_dibuang(self):
        llm, _, _ = buat([ok("<think>menalar panjang</think>\n\n**Jawaban singkat:** Ya.")])
        self.assertEqual(llm.generate(QWEN, MSG).teks, "**Jawaban singkat:** Ya.")

    def test_berhenti_karena_batas_panjang_ditandai(self):
        llm, _, _ = buat([ok("terpotong", finish="length")])
        h = llm.generate(QWEN, MSG)
        self.assertEqual((h.selesai_normal, h.finish_reason), (False, "length"))

    def test_konten_kosong_didiagnosis(self):
        llm, _, _ = buat([ok("", finish="length", keluar=1536, reasoning_tokens=1536)])
        with self.assertRaises(GenerasiError) as cm:
            llm.generate(GPTOSS, MSG)
        d = cm.exception.detail_teknis
        self.assertIn("finish_reason=length", d)
        self.assertIn("reasoning_tokens=1536", d)

    def test_hanya_think_dianggap_kosong(self):
        llm, _, _ = buat([ok("<think>hanya berpikir</think>")])
        with self.assertRaises(GenerasiError):
            llm.generate(QWEN, MSG)

    def test_respons_bukan_json_chat(self):
        for r in (Resp(200, ValueError("bukan json")), Resp(200, {"tidak": "ada choices"})):
            llm, _, _ = buat([r])
            with self.assertRaises(GenerasiError) as cm:
                llm.generate(QWEN, MSG)
            self.assertIn("tidak dapat dibaca", str(cm.exception))


class TestGalatHTTP(unittest.TestCase):
    def test_429_kuota_tanpa_retry_dengan_retry_after(self):
        llm, http, tidur = buat([galat(429, "Rate limit reached. Try again in 6.2s. key " + KUNCI, {"retry-after": "6.2"})])
        with self.assertRaises(KuotaHabisError) as cm:
            llm.generate(QWEN, MSG)
        self.assertEqual((len(http.posts), tidur), (1, []))  # tidak mengulang -> tidak membakar kuota
        self.assertIn("sekitar 7 detik", str(cm.exception))
        self.assertNotIn(KUNCI, str(cm.exception) + cm.exception.detail_teknis)

    def test_429_tanpa_retry_after(self):
        llm, _, _ = buat([galat(429, "limit")])
        with self.assertRaises(KuotaHabisError) as cm:
            llm.generate(QWEN, MSG)
        self.assertIn("beberapa saat", str(cm.exception))

    def test_413_terlalu_besar(self):
        llm, http, _ = buat([galat(413, "Request too large ... tokens per minute (TPM): Limit 8000, Requested 9500")])
        with self.assertRaises(GenerasiError) as cm:
            llm.generate(QWEN, MSG)
        self.assertIn("terlalu besar", str(cm.exception))
        self.assertIn("TPM", cm.exception.detail_teknis)
        self.assertEqual(len(http.posts), 1)

    def test_401_403_pesan_aman(self):
        for kode in (401, 403):
            llm, _, _ = buat([galat(kode, "Invalid API Key " + KUNCI)])
            with self.assertRaises(GenerasiError) as cm:
                llm.generate(QWEN, MSG)
            self.assertNotIn("kunci", str(cm.exception).lower())
            self.assertNotIn(KUNCI, cm.exception.detail_teknis)

    def test_404_model_tidak_tersedia(self):
        llm, _, _ = buat([galat(404, "The model does not exist")])
        with self.assertRaises(ModelTidakTersediaError):
            llm.generate(QWEN, MSG)

    def test_400_lain_tidak_diulang(self):
        llm, http, _ = buat([galat(400, "messages must be non-empty")])
        with self.assertRaises(GenerasiError):
            llm.generate(QWEN, MSG)
        self.assertEqual(len(http.posts), 1)

    def test_5xx_diulang_lalu_berhasil_dengan_jeda_bertambah(self):
        llm, http, tidur = buat([galat(503, "over capacity"), galat(500, "internal"), ok("pulih")])
        self.assertEqual(llm.generate(QWEN, MSG).teks, "pulih")
        self.assertEqual((len(http.posts), tidur), (3, [2.0, 4.0]))

    def test_5xx_terus_menerus_menyerah_setelah_3_kali(self):
        llm, http, _ = buat([galat(503, "over capacity")])
        with self.assertRaises(GenerasiError) as cm:
            llm.generate(QWEN, MSG)
        self.assertEqual(len(http.posts), 3)
        self.assertIn("sibuk", str(cm.exception))

    def test_galat_jaringan_sementara_diulang(self):
        class ReadTimeout(Exception):
            pass

        llm, http, _ = buat([ReadTimeout("timed out"), ok("ok")])
        self.assertEqual(llm.generate(QWEN, MSG).teks, "ok")
        self.assertEqual(len(http.posts), 2)

    def test_galat_tak_dikenal_dibungkus_tanpa_membocorkan_kunci(self):
        llm, _, _ = buat([ValueError("aneh " + KUNCI)])
        with self.assertRaises(GenerasiError) as cm:
            llm.generate(QWEN, MSG)
        self.assertNotIn(KUNCI, str(cm.exception) + cm.exception.detail_teknis)


class TestPenolakanPenalaran(unittest.TestCase):
    def test_parameter_penalaran_dilepas_dan_diingat(self):
        tolak = galat(400, "reasoning_effort value 'none' is not supported for this model")
        llm, http, _ = buat([tolak, ok("ok1"), ok("ok2")])
        self.assertEqual(llm.generate(QWEN, MSG).teks, "ok1")
        self.assertIn("reasoning_effort", http.posts[0]["json"])
        self.assertNotIn("reasoning_effort", http.posts[1]["json"])
        llm.generate(QWEN, MSG)
        self.assertEqual(len(http.posts), 3)  # panggilan berikut langsung tanpa parameter itu
        self.assertNotIn("reasoning_effort", http.posts[2]["json"])

    def test_tidak_memengaruhi_model_lain(self):
        llm, http, _ = buat([galat(400, "reasoning_effort not supported"), ok("a"), ok("b")])
        llm.generate(QWEN, MSG)
        llm.generate(GPTOSS, MSG)
        self.assertIn("reasoning_effort", http.posts[2]["json"])

    def test_model_tanpa_pengaturan_penalaran_tidak_diulang(self):
        llm, http, _ = buat([galat(400, "reasoning is weird")])
        with self.assertRaises(GenerasiError):
            llm.generate("meta/llama-x", MSG)
        self.assertEqual(len(http.posts), 1)

    def test_penolakan_ganda_tidak_berputar(self):
        llm, http, _ = buat([galat(400, "reasoning_effort invalid")])
        with self.assertRaises(GenerasiError):
            llm.generate(QWEN, MSG)
        self.assertEqual(len(http.posts), 2)  # satu percobaan + satu ulangan tanpa parameter, lalu berhenti


class TestDaftarModel(unittest.TestCase):
    def test_hanya_model_aktif(self):
        data = {"data": [{"id": QWEN, "active": True}, {"id": GPTOSS}, {"id": "lama/x", "active": False},
                         {"id": "whisper-large-v3", "active": True}, {"active": True}]}
        llm, http, _ = buat(get=Resp(200, data))
        self.assertEqual(llm.daftar_model(), {QWEN, GPTOSS, "whisper-large-v3"})
        self.assertEqual(http.gets[0]["url"], "https://api.groq.com/openai/v1/models")
        self.assertEqual(http.gets[0]["headers"]["Authorization"], "Bearer " + KUNCI)

    def test_galat_dipetakan_dan_aman(self):
        llm, _, _ = buat(get=galat(401, "Invalid " + KUNCI))
        with self.assertRaises(GenerasiError) as cm:
            llm.daftar_model()
        self.assertNotIn(KUNCI, str(cm.exception) + cm.exception.detail_teknis)
        llm2, _, _ = buat(get=ConnectionError("tidak ada jaringan " + KUNCI))
        with self.assertRaises(GenerasiError) as cm2:
            llm2.daftar_model()
        self.assertNotIn(KUNCI, str(cm2.exception) + cm2.exception.detail_teknis)

    def test_respons_bukan_json(self):
        llm, _, _ = buat(get=Resp(200, ValueError("x")))
        with self.assertRaises(GenerasiError):
            llm.daftar_model()


if __name__ == "__main__":
    unittest.main()
