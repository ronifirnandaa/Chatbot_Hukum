"""Cek koneksi ke Neo4j SUNGGUHAN (dijalankan oleh Anda, bukan bagian dari uji otomatis).

Jalankan dari folder project:
    python tests/cek_koneksi_neo4j.py

Kredensial dibaca dari environment / .env / .streamlit/secrets.toml (lewat Streamlit bila terpasang).
Skrip ini tidak pernah mencetak URI, username, maupun password.
Kode keluar: 0 = OK, 1 = konfigurasi/koneksi gagal, 2 = database bukan KG yang diharapkan.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, load_neo4j_settings  # noqa: E402
from src.retriever.neo4j_connection import (  # noqa: E402
    KGTidakSesuaiError,
    KoneksiNeo4jError,
    Neo4jConnection,
)


def main() -> int:
    try:
        settings = load_neo4j_settings()
    except ConfigError as e:
        print("[GAGAL] Konfigurasi:", e)
        return 1
    print("Konfigurasi dimuat. Database:", settings.database)

    try:
        with Neo4jConnection(settings) as koneksi:
            print("[OK] Terhubung ke Neo4j.")
            h = koneksi.cek_kg()
    except KoneksiNeo4jError as e:
        print("[GAGAL] Koneksi:", e)
        return 1

    print("\nIsi database:")
    print("  indeks vektor :", h["indeks"])
    print("  LegalQA       : %d (ulasan terisi: %d)" % (h["n_legalqa"], h["n_ulasan"]))
    print("  Law           :", h["n_law"])
    print("  REFERENCES    :", h["n_references"])
    for p in h["peringatan"]:
        print("[PERINGATAN]", p)
    for g in h["galat"]:
        print("[GALAT]", g)
    if h["galat"]:
        print("\nHASIL: database BUKAN KG yang diharapkan.")
        return 2
    print("\nHASIL: KG valid. Siap lanjut ke integrasi retriever.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
