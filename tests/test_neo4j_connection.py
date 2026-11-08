"""Uji koneksi Neo4j dengan driver tiruan (offline)."""
import unittest

from src.config import Neo4jSettings
from src.retriever import neo4j_connection as nc
from src.retriever.neo4j_connection import (
    KGTidakSesuaiError,
    KoneksiNeo4jError,
    Neo4jConnection,
    panggil_dengan_retry,
)

SET = Neo4jSettings(
    uri="neo4j+s://abcd1234.databases.neo4j.io",
    username="userabcd",
    password="rahasia-sangat-panjang",
    database="dbabcd",
)


class FakeResult:
    def __init__(self, data):
        self._d = data

    def data(self):
        return self._d


class FakeSession:
    def __init__(self, router, log):
        self.router, self.log, self.closed = router, log, False

    def run(self, query, **params):
        self.log.append(query)
        out = self.router(query, **params)
        if isinstance(out, Exception):
            raise out
        return FakeResult(out)

    def close(self):
        self.closed = True


class FakeDriver:
    def __init__(self, router=None, verify_errors=()):
        self.router = router or (lambda q, **p: [])
        self.verify_errors = list(verify_errors)
        self.verify_calls, self.closed, self.log, self.sessions = 0, False, [], []
        self.db = None

    def verify_connectivity(self):
        self.verify_calls += 1
        if self.verify_errors:
            raise self.verify_errors.pop(0)

    def session(self, database=None):
        self.db = database
        s = FakeSession(self.router, self.log)
        self.sessions.append(s)
        return s

    def close(self):
        self.closed = True


def router_kg(indeks=("law_vector", "legalqa_vector", "q_vector"), q=1010, ul=1010, law=9066, ref=32063):
    def r(query, **p):
        if "SHOW VECTOR INDEXES" in query:
            return [{"name": n} for n in indeks]
        if "count(q.ulasan)" in query:
            return [{"c_q": q, "c_ulasan": ul, "c_law": law}]
        if "REFERENCES" in query:
            return [{"c": ref}]
        return []

    return r


def buat(driver, **kw):
    return Neo4jConnection(SET, driver_factory=lambda uri, auth: driver, sleep=lambda s: None, **kw)


class AuthError(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


class TestRetry(unittest.TestCase):
    def test_berhasil_setelah_galat_sementara(self):
        n = {"i": 0}
        tidur = []

        def fn():
            n["i"] += 1
            if n["i"] < 3:
                raise ServiceUnavailable("putus")
            return "ok"

        self.assertEqual(panggil_dengan_retry(fn, sleep=tidur.append), "ok")
        self.assertEqual(tidur, [2.0, 4.0])  # jeda bertambah: 2s, 4s

    def test_galat_non_sementara_tidak_diulang(self):
        n = {"i": 0}

        def fn():
            n["i"] += 1
            raise ValueError("salah query")

        with self.assertRaises(ValueError):
            panggil_dengan_retry(fn, sleep=lambda s: None)
        self.assertEqual(n["i"], 1)

    def test_habis_percobaan(self):
        n = {"i": 0}

        def fn():
            n["i"] += 1
            raise ServiceUnavailable("putus")

        with self.assertRaises(ServiceUnavailable):
            panggil_dengan_retry(fn, attempts=3, sleep=lambda s: None)
        self.assertEqual(n["i"], 3)


class TestKoneksi(unittest.TestCase):
    def test_verifikasi_gagal_pesan_aman(self):
        err = ServiceUnavailable("gagal ke abcd1234.databases.neo4j.io pakai rahasia-sangat-panjang")
        drv = FakeDriver(verify_errors=[err, err, err])
        with self.assertRaises(KoneksiNeo4jError) as cm:
            buat(drv)
        e = cm.exception
        for rahasia in ("abcd1234", "rahasia-sangat-panjang", "userabcd", "neo4j.io"):
            self.assertNotIn(rahasia, str(e))
            self.assertNotIn(rahasia, e.detail_teknis)
        self.assertEqual(drv.verify_calls, 3)
        self.assertTrue(drv.closed)

    def test_auth_error_pesan_khusus(self):
        drv = FakeDriver(verify_errors=[AuthError("unauthorized")])
        with self.assertRaises(KoneksiNeo4jError) as cm:
            buat(drv)
        self.assertIn("kredensial", str(cm.exception).lower())
        self.assertEqual(drv.verify_calls, 1)  # tidak diulang

    def test_pemulihan_dari_putus_sementara(self):
        drv = FakeDriver(verify_errors=[ServiceUnavailable("putus")])
        k = buat(drv)
        self.assertEqual(drv.verify_calls, 2)
        k.close()

    def test_factory_gagal(self):
        def f(uri, auth):
            raise AuthError("salah")

        with self.assertRaises(KoneksiNeo4jError):
            Neo4jConnection(SET, driver_factory=f, sleep=lambda s: None)

    def test_sesi_memakai_database_dan_selalu_ditutup(self):
        drv = FakeDriver()
        k = buat(drv)
        with self.assertRaises(RuntimeError):
            with k.session():
                raise RuntimeError("x")
        self.assertEqual(drv.db, "dbabcd")
        self.assertTrue(drv.sessions[0].closed)

    def test_jalankan_retry_dan_galat_aman(self):
        berhasil = {"n": 0}

        def r(q, **p):
            berhasil["n"] += 1
            return ServiceUnavailable("putus abcd1234") if berhasil["n"] < 2 else [{"a": 1}]

        k = buat(FakeDriver(router=r))
        self.assertEqual(k.jalankan("RETURN 1"), [{"a": 1}])

        k2 = buat(FakeDriver(router=lambda q, **p: ValueError("Syntax error di rahasia-sangat-panjang")))
        with self.assertRaises(KoneksiNeo4jError) as cm:
            k2.jalankan("BAD")
        self.assertNotIn("rahasia-sangat-panjang", str(cm.exception) + cm.exception.detail_teknis)

    def test_context_manager_menutup_driver(self):
        drv = FakeDriver()
        with buat(drv):
            pass
        self.assertTrue(drv.closed)


class TestCekKG(unittest.TestCase):
    def test_kg_final_valid_tanpa_peringatan(self):
        h = buat(FakeDriver(router=router_kg())).pastikan_kg_valid()
        self.assertEqual(h["galat"], [])
        self.assertEqual(h["peringatan"], [])
        self.assertEqual((h["n_legalqa"], h["n_law"], h["n_references"]), (1010, 9066, 32063))

    def test_indeks_hilang_adalah_galat(self):
        k = buat(FakeDriver(router=router_kg(indeks=("legalqa_vector",))))
        with self.assertRaises(KGTidakSesuaiError) as cm:
            k.pastikan_kg_valid()
        self.assertIn("law_vector", str(cm.exception))

    def test_tanpa_legalqa_adalah_galat(self):
        h = buat(FakeDriver(router=router_kg(q=0, ul=0))).cek_kg()
        self.assertTrue(any("LegalQA" in g for g in h["galat"]))

    def test_ulasan_tidak_lengkap_adalah_galat(self):
        h = buat(FakeDriver(router=router_kg(ul=1000))).cek_kg()
        self.assertTrue(any("ulasan" in g for g in h["galat"]))

    def test_kg_berbeda_hanya_peringatan(self):
        h = buat(FakeDriver(router=router_kg(law=7523, ref=29789))).pastikan_kg_valid()
        self.assertEqual(h["galat"], [])
        self.assertTrue(any("berbeda dari KG" in p for p in h["peringatan"]))

    def test_tanpa_references_peringatan(self):
        h = buat(FakeDriver(router=router_kg(ref=0))).cek_kg()
        self.assertTrue(any("REFERENCES" in p for p in h["peringatan"]))


if __name__ == "__main__":
    unittest.main()
