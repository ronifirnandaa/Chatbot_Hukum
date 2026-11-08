"""Halaman 1: Chatbot tanya jawab hukum pidana Indonesia.

Alur: pertanyaan -> Hybrid Retriever (Neo4j) -> konteks hukum -> model generator terpilih -> jawaban + sumber hukum.
Setiap pertanyaan diproses MANDIRI: sistem tidak mengingat pertanyaan sebelumnya (riwayat hanya ditampilkan di layar
selama sesi berlangsung). Evaluasi RAGAS tidak dijalankan di sini.
"""
import logging
import streamlit as st

from src.app_resources import get_llm, get_model_tersedia, get_pembatas_global, get_retriever
from src.chat.chat_service import HasilChat, proses_pertanyaan
from src.chat.pembatas_laju import PembatasLaju
from src.config import (
    GAYA_VALID,
    ConfigError,
    get_batas_laju,
    get_gaya_default,
    get_mode_potong,
    get_peringatan_kutipan,
)
from src.generator.gaya_jawaban import LABEL_GAYA
from src.generator.llm_client import GenerasiError
from src.retriever.neo4j_connection import KoneksiNeo4jError

log = logging.getLogger(__name__)

st.set_page_config(page_title="Chatbot Hukum Pidana", layout="wide")

CONTOH = "Apa pasal yang dapat menjerat pengurus yayasan yang menyalahgunakan dana donasi?"

# ------------------------------------------------------------------------------ tampilan hasil
def _angka(x: int) -> str:
    return f"{x:,}".replace(",", ".")

def _skor(nilai) -> str:
    try:
        return f"{float(nilai):.4f}"
    except (TypeError, ValueError):
        return "-"

def _cuplikan_ulasan(teks: str, maks: int = 350) -> str:
    isi = (teks or "").split("\n\n[PASAL TERKAIT]:")[0].replace("[ULASAN / JAWABAN]: ", "", 1).strip()
    return isi[:maks] + ("…" if len(isi) > maks else "")

def _tampilkan_sumber(hasil: HasilChat) -> None:
    items = hasil.retrieval.items
    with st.expander(f"Sumber Hukum yang Ditemukan ({len(items)})"):
        if not items:
            st.info("Tidak ada dokumen yang relevan ditemukan.")
            return
            
        for rank, it in enumerate(items, 1):
            st.markdown(f"**{rank}. Dokumen #{it.get('node_id')}** — *Skor Retrieval: {_skor(it.get('score'))}*")
            cuplikan = _cuplikan_ulasan(it.get("text"))
            if cuplikan:
                st.write(f"*{cuplikan}*")
                
            pasal = it.get("pasal_terkait") or []
            if pasal:
                st.markdown("**Pasal terkait:**\n" + "\n".join(f"- {p['label']}" for p in pasal))
            else:
                st.caption("Tidak ada pasal spesifik pada dokumen ini.")
            if rank < len(items):
                st.divider()

def _tampilkan_konteks(hasil: HasilChat, n: int) -> None:
    k = hasil.jawaban.konteks
    if not k.teks:
        return
    with st.expander("Detail Konteks Model"):
        ringkas = (f"{k.n_dokumen_dipakai} dari {k.n_dokumen_total} dokumen | "
                   f"{_angka(k.panjang_akhir)} / {_angka(k.panjang_penuh)} karakter")
        if k.n_pasal_tidak_ditampilkan:
            ringkas += f" | {k.n_pasal_tidak_ditampilkan} pasal terkait dikecualikan"
        if k.dipotong:
            ringkas += " | Sebagian teks dipotong"
            
        st.caption(ringkas)
        st.text_area("Teks Konteks", value=k.teks, height=250, disabled=True, key=f"ctx_{n}",
                     label_visibility="collapsed")

def tampilkan_hasil(hasil: HasilChat, n: int) -> None:
    if hasil.jawaban:
        st.markdown(hasil.jawaban.jawaban)
        
        j = hasil.jawaban
        token = (f" | Token: {_angka(j.token_masuk)} masuk / {_angka(j.token_keluar)} keluar"
                 if j.token_masuk is not None and j.token_keluar is not None else "")
        
        st.caption(
            f"**Model:** {hasil.model_label or hasil.model_id} | **Gaya:** {LABEL_GAYA.get(hasil.gaya, hasil.gaya)} | "
            f"**Waktu:** Retrieval {hasil.retrieval.waktu_detik:.1f}s, Generasi {j.waktu_generasi:.1f}s{token}"
        )
        
        if not hasil.jawaban.selesai_normal:
            st.warning("Respons terpotong karena batas maksimal panjang keluaran tercapai.")
            
        if hasil.kutipan and not hasil.kutipan.semua_ditemukan and get_peringatan_kutipan():
            st.warning(
                f"Pemeriksaan Otomatis: {hasil.kutipan.ringkas()}. Kutipan ini tidak ditemukan dalam konteks "
                "yang diberikan kepada model. Mohon verifikasi langsung pada peraturan aslinya."
            )
            
    if hasil.galat:
        st.error(hasil.galat)
    if hasil.retrieval:
        for p in hasil.retrieval.peringatan:
            st.warning(p)
            
        _tampilkan_sumber(hasil)
        if hasil.jawaban:
            _tampilkan_konteks(hasil, n)

# ------------------------------------------------------------------------------ sidebar
def sidebar():
    """Render sidebar; kembalikan (model terpilih (ModelInfo) atau None, gaya jawaban)."""
    model = None
    with st.sidebar:
        st.title("Pengaturan Sesi")
        
        try:
            daftar = get_model_tersedia()
        except ConfigError as e:
            st.error(str(e))
            daftar = []
        except ImportError:
            st.error("Paket 'google-genai' belum terpasang.")
            daftar = []
        except GenerasiError as e:
            st.error(str(e))
            daftar = []
        else:
            if not daftar:
                st.warning("Tidak ada model generator yang tersedia.")
            try:
                for nama, pesan in getattr(get_llm(), "peringatan_penyedia", lambda: {})().items():
                    st.warning(f"{nama}: {pesan}")
            except (ConfigError, ImportError):
                pass
                
        if daftar:
            model = st.selectbox("Model Generator", daftar, format_func=lambda m: m.label_lengkap)
            
        try:
            gaya_awal = get_gaya_default()
        except ConfigError as e:
            st.error(str(e))
            gaya_awal = GAYA_VALID[0]
            
        gaya = st.selectbox("Gaya Jawaban", GAYA_VALID, index=GAYA_VALID.index(gaya_awal),
                            format_func=lambda g: LABEL_GAYA[g])
                            
        st.divider()
        if st.button("Bersihkan Percakapan", use_container_width=True):
            st.session_state.riwayat = []
            st.rerun()
            
        with st.expander("Informasi Sistem"):
            st.markdown(
                """
                - **Konteks:** Dicari via Hybrid Retriever (Vektor + Graf) di Neo4j.
                - **Privasi Sesi:** Tiap pertanyaan diproses mandiri; tidak ada memori lintas pertanyaan.
                - **Gaya:** *Ringkas* untuk pembacaan cepat, *Naratif* berbasis prompt riset.
                - **Disclaimer:** Jawaban dihasilkan oleh AI, bukan nasihat hukum. Periksa kembali sumber aslinya.
                """
            )
            
    return model, gaya

# ------------------------------------------------------------------------------ utama
def _izinkan():
    ok, tunggu = st.session_state.pembatas_sesi.coba()
    if not ok:
        return False, f"Batas pertanyaan per sesi tercapai. Silakan tunggu {int(tunggu) + 1} detik."
    ok, tunggu = get_pembatas_global().coba()
    if not ok:
        return False, f"Server sedang sibuk. Silakan coba dalam {int(tunggu) + 1} detik."
    return True, ""

def main() -> None:
    try:
        if "pembatas_sesi" not in st.session_state:
            maks, jendela = get_batas_laju()["sesi"]
            st.session_state.pembatas_sesi = PembatasLaju(maks, jendela)
    except ConfigError as e:
        st.error(str(e))
        st.stop()
        
    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []

    model, gaya = sidebar()

    st.title("Tanya Jawab Hukum Pidana")
    st.markdown("Cari tahu landasan hukum pidana Indonesia berdasarkan arsip dokumen yang tersedia.")
    st.divider()

    if not st.session_state.riwayat:
        with st.chat_message("assistant"):
            st.write("Halo! Silakan ajukan pertanyaan terkait hukum pidana Indonesia.")
            st.caption(f"Contoh: *\"{CONTOH}\"*")

    for n, pesan in enumerate(st.session_state.riwayat):
        with st.chat_message(pesan["peran"]):
            if pesan["peran"] == "user":
                st.markdown(pesan["teks"])
            else:
                tampilkan_hasil(pesan["hasil"], n)

    pertanyaan = st.chat_input("Ketik pertanyaan Anda di sini...", disabled=model is None)
    
    if not pertanyaan or model is None:
        return

    try:
        diizinkan, pesan_batas = _izinkan()
    except ConfigError as e:
        st.error(str(e))
        return
        
    if not diizinkan:
        st.warning(pesan_batas)
        return

    try:
        retriever, llm = get_retriever(), get_llm()
    except (ConfigError, KoneksiNeo4jError) as e:
        st.error(str(e))
        return
    except Exception:
        log.exception("Sumber daya belum siap")
        st.error("Sistem pencarian belum siap. Pastikan konfigurasi dan *service* berjalan dengan baik.")
        return

    n_asisten = len(st.session_state.riwayat) + 1
    st.session_state.riwayat.append({"peran": "user", "teks": pertanyaan})
    
    with st.chat_message("user"):
        st.markdown(pertanyaan)
        
    with st.chat_message("assistant"):
        with st.spinner("Menganalisis dokumen dan menyusun jawaban..."):
            try:
                mode = get_mode_potong()
            except ConfigError:
                mode = None 
                
            hasil = proses_pertanyaan(pertanyaan, model.kunci, retriever, llm, mode_potong=mode,
                                      periksa=True, model_label=model.label_lengkap, gaya=gaya)
        tampilkan_hasil(hasil, n_asisten)
        
    st.session_state.riwayat.append({"peran": "assistant", "hasil": hasil})

main()