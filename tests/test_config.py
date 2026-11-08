"""Uji konfigurasi (offline). Jalankan: python -m unittest discover -s tests -v"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config
from src.config import ConfigError, Neo4jSettings, load_neo4j_settings

LENGKAP = {
    "NEO4J_URI": "neo4j+s://abcd1234.databases.neo4j.io",
    "NEO4J_USERNAME": "userabcd",
    "NEO4J_PASSWORD": "rahasia-sangat-panjang",
    "NEO4J_DATABASE": "dbabcd",
}


def _bersih_env(extra=None):
    """Environment tanpa variabel NEO4J_* dan tanpa memuat .env asli."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEO4J_")}
    env.update(extra or {})
    return env


class TestConfig(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(config, "_DOTENV_SUDAH_DIMUAT", True)  # cegah pembacaan .env sungguhan
        p.start()
        self.addCleanup(p.stop)
        p2 = mock.patch.object(config, "_dari_streamlit_secrets", lambda nama: None)
        p2.start()
        self.addCleanup(p2.stop)

    def test_lengkap(self):
        with mock.patch.dict(os.environ, _bersih_env(LENGKAP), clear=True):
            s = load_neo4j_settings()
        self.assertEqual(s.database, "dbabcd")
        self.assertEqual(s.uri, LENGKAP["NEO4J_URI"])

    def test_database_default_neo4j(self):
        env = dict(LENGKAP)
        del env["NEO4J_DATABASE"]
        with mock.patch.dict(os.environ, _bersih_env(env), clear=True):
            self.assertEqual(load_neo4j_settings().database, "neo4j")

    def test_kurang_menyebut_nama_bukan_nilai(self):
        env = dict(LENGKAP)
        del env["NEO4J_PASSWORD"]
        with mock.patch.dict(os.environ, _bersih_env(env), clear=True):
            with self.assertRaises(ConfigError) as cm:
                load_neo4j_settings()
        pesan = str(cm.exception)
        self.assertIn("NEO4J_PASSWORD", pesan)
        self.assertNotIn(LENGKAP["NEO4J_USERNAME"], pesan)
        self.assertNotIn(LENGKAP["NEO4J_URI"], pesan)

    def test_nilai_kosong_dianggap_kurang(self):
        env = dict(LENGKAP, NEO4J_PASSWORD="   ")
        with mock.patch.dict(os.environ, _bersih_env(env), clear=True):
            with self.assertRaises(ConfigError):
                load_neo4j_settings()

    def test_skema_uri_tidak_valid(self):
        env = dict(LENGKAP, NEO4J_URI="http://contoh.com")
        with mock.patch.dict(os.environ, _bersih_env(env), clear=True):
            with self.assertRaises(ConfigError):
                load_neo4j_settings()

    def test_repr_tidak_membocorkan_rahasia(self):
        s = Neo4jSettings(**{k.split("_", 1)[1].lower(): v for k, v in LENGKAP.items()})
        teks = repr(s) + str(s)
        for rahasia in (LENGKAP["NEO4J_PASSWORD"], LENGKAP["NEO4J_USERNAME"], "abcd1234"):
            self.assertNotIn(rahasia, teks)

    def test_streamlit_secrets_dipakai_bila_env_kosong(self):
        with mock.patch.dict(os.environ, _bersih_env(), clear=True):
            with mock.patch.object(config, "_dari_streamlit_secrets", lambda nama: LENGKAP.get(nama)):
                self.assertEqual(load_neo4j_settings().username, "userabcd")

    def test_env_lebih_prioritas_dari_streamlit(self):
        env = dict(LENGKAP, NEO4J_USERNAME="dari-env")
        with mock.patch.dict(os.environ, _bersih_env(env), clear=True):
            with mock.patch.object(config, "_dari_streamlit_secrets", lambda nama: "dari-st"):
                self.assertEqual(load_neo4j_settings().username, "dari-env")

    def test_dotenv_tidak_menimpa_env(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / ".env"
            f.write_text('# komentar\nNEO4J_USERNAME="dari-file"\nNEO4J_URI=neo4j+s://x.io\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {"NEO4J_USERNAME": "dari-env"}, clear=True):
                self.assertTrue(config.muat_dotenv(f))
                self.assertEqual(os.environ["NEO4J_USERNAME"], "dari-env")
                self.assertEqual(os.environ["NEO4J_URI"], "neo4j+s://x.io")

    def test_dotenv_tidak_ada(self):
        self.assertFalse(config.muat_dotenv(Path("/tidak/ada/.env")))


if __name__ == "__main__":
    unittest.main()
