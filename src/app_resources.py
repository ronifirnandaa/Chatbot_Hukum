"""Sumber daya bersama aplikasi Streamlit: dibuat SEKALI lalu dipakai semua halaman dan semua sesi pengunjung.

Modul ini satu-satunya tempat objek berat (koneksi Neo4j, model embedding, klien LLM) dibuat, supaya halaman mana
pun tidak membuat salinannya sendiri. Hanya diimpor oleh skrip Streamlit (bergantung pada `streamlit`).
"""
import streamlit as st

from src.chat.chat_service import RetrieverTerkunci
from src.chat.pembatas_laju import PembatasLaju
from src.config import get_batas_laju, load_neo4j_settings
from src.generator.llm_factory import buat_llm_dari_konfigurasi
from src.generator.llm_router import LLMRouter
from src.generator.model_registry import model_tersedia
from src.retriever.hybrid_retriever import HybridGraphRAGRetriever, muat_encoder
from src.retriever.neo4j_connection import Neo4jConnection


@st.cache_resource(show_spinner="Menghubungkan ke database...")
def get_koneksi() -> Neo4jConnection:
    return Neo4jConnection(load_neo4j_settings())


@st.cache_resource(show_spinner="Memuat model embedding (pertama kali bisa beberapa menit)...")
def get_encoder():
    return muat_encoder()


@st.cache_resource(show_spinner=False)
def get_retriever() -> RetrieverTerkunci:
    return RetrieverTerkunci(HybridGraphRAGRetriever(get_koneksi(), encoder=get_encoder()))


@st.cache_resource(show_spinner=False)
def get_llm() -> LLMRouter:
    return buat_llm_dari_konfigurasi()


@st.cache_data(ttl=600, show_spinner="Memeriksa model generator...")
def get_model_tersedia():
    return model_tersedia(get_llm())


@st.cache_resource(show_spinner=False)
def get_pembatas_global() -> PembatasLaju:
    maks, jendela = get_batas_laju()["global"]
    return PembatasLaju(maks, jendela)
