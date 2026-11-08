"""Beranda aplikasi + panel status sistem. Halaman Chatbot ada di pages/; halaman evaluasi menyusul."""
import streamlit as st

from src.app_resources import get_koneksi, get_model_tersedia
from src.config import ConfigError
from src.generator.llm_client import GenerasiError
from src.retriever.neo4j_connection import KoneksiNeo4jError

st.set_page_config(
    page_title="Chatbot Hukum Pidana", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("LegalBot ID")
    st.markdown("Dasbor pemantauan sistem *Graph Retrieval-Augmented Generation* (GraphRAG).")
    st.info("Gunakan menu navigasi di samping untuk membuka halaman Chatbot.")
    st.divider()
    st.caption("© 2026 - Roni Firnanda | Universitas Trunojoyo Madura")

st.title("Beranda Sistem")
st.markdown("Pantau status koneksi *Knowledge Graph* (Neo4j) dan ketersediaan *Large Language Model*.")
st.divider()

st.subheader("Status Knowledge Graph (Neo4j)")
with st.container(border=True):
    try:
        with st.spinner("Memeriksa koneksi database..."):
            hasil = get_koneksi().cek_kg()
    except ConfigError as e:
        st.error(f"Kesalahan Konfigurasi: {e}")
    except KoneksiNeo4jError as e:
        st.error(f"Gagal Terhubung: {e}")
    else:
        if hasil["galat"]:
            st.error("Database tersambung, tetapi struktur Knowledge Graph tidak sesuai:")
            for galat in hasil["galat"]:
                st.markdown(f"- {galat}")
        else:
            st.success("Database terhubung dan Knowledge Graph valid.")
        
        st.markdown("##### Metrik Data")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Node: LegalQA", hasil["n_legalqa"])
        c2.metric("Node: Law", hasil["n_law"])
        c3.metric("Relasi: REFERENCES", hasil["n_references"])
        c4.metric("Indeks Vektor", len(hasil["indeks"]))
        
        if hasil["peringatan"]:
            st.divider()
            for p in hasil["peringatan"]:
                st.warning(f"Peringatan: {p}")

st.subheader("Status Model Generator (LLM)")
with st.container(border=True):
    try:
        with st.spinner("Memuat daftar model..."):
            daftar_model = get_model_tersedia()
    except ConfigError as e:
        st.warning(f"Peringatan Konfigurasi: {e}")
    except ImportError:
        st.warning("Paket 'google-genai' belum terpasang (jalankan `pip install google-genai`).")
    except GenerasiError as e:
        st.error(f"Kesalahan Generator: {e}")
    else:
        if daftar_model:
            st.success("Model API siap digunakan.")
            st.markdown("##### Model yang tersedia")
            for m in daftar_model:
                st.markdown(f"- `{m.label_lengkap}`")
        else:
            st.warning("Tidak ada model terkonfigurasi. Periksa parameter `GENERATOR_MODELS`.")