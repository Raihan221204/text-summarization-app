import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import fitz  
from langdetect import detect
import matplotlib.pyplot as plt
from collections import Counter
import re
import gdown
import zipfile
import os


# Konfigurasi halaman
st.set_page_config(page_title="Text Summarization", layout="centered")

# ID file Google Drive yang berisi folder model (file ZIP)
google_drive_file_id = "1yGYKv-wqR6QdSlQqabx1qiuFQnO1X_Xu" 

# URL untuk mendownload file ZIP
download_url = f"https://drive.google.com/uc?export=download&id={google_drive_file_id}"

# Unduh file ZIP
zip_file = "model.zip"
gdown.download(download_url, zip_file, quiet=False)

# Ekstrak ZIP
if zipfile.is_zipfile(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("summarization_model")  # Folder untuk mengekstrak file
else:
    print("File bukan ZIP yang valid")

# Sekarang model akan berada di folder 'summarization_model' dan bisa dimuat
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "summarization_model"

# Load model dan tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

tokenizer, model = load_model()

# Judul Aplikasi
st.title("📚 Text Summarization App")
st.markdown("Masukkan teks atau upload dokumen untuk diringkas dengan gaya dan panjang yang kamu tentukan.")

# Upload file atau input teks manual
uploaded_file = st.file_uploader("📄 Upload Dokumen (PDF/TXT)", type=["pdf", "txt"])
text_input = ""

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text_input = " ".join([page.get_text() for page in doc])
    elif uploaded_file.type == "text/plain":
        text_input = uploaded_file.read().decode("utf-8")
else:
    text_input = st.text_area("✍️ Masukkan Teks", height=300)

# Deteksi bahasa input
lang_code = "en"  # default
if text_input.strip():
    try:
        lang_code = detect(text_input)
        lang_detected = "Indonesia" if lang_code == "id" else "English"
        st.info(f"📌 Bahasa terdeteksi: **{lang_detected}**")
    except:
        st.warning("⚠️ Tidak dapat mendeteksi bahasa. Gunakan default bahasa Inggris.")

# Pilihan gaya dan panjang ringkasan
style = st.selectbox("🎨 Gaya Ringkasan", ["Formal", "Santai", "Berita"])
length = st.radio("⏱️ Panjang Ringkasan", ["Singkat", "Sedang", "Panjang"])

# Tombol untuk meringkas
if st.button("🔍 Ringkas Teks"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        with st.spinner("Sedang meringkas..."):

            # Panjang ringkasan
            max_len, min_len = (100, 85) if length == "Singkat" else (200, 185) if length == "Sedang" else (300, 285)

            # Prompt berdasarkan gaya & bahasa
            prompt_map = {
                "id": {
                    "Formal": "Ringkas teks berikut ini dalam gaya bahasa formal dan akademik:",
                    "Santai": "Ringkas teks berikut ini dengan bahasa yang santai dan mudah dipahami:",
                    "Berita": "Ringkas teks berikut ini seperti gaya penulisan berita:",
                },
                "en": {
                    "Formal": "Summarize this text in a formal and academic tone:",
                    "Santai": "Summarize this text in a casual and simple tone:",
                    "Berita": "Summarize this text like a news article:",
                }
            }

            prompt = prompt_map.get(lang_code, prompt_map["en"])[style] + " " + text_input

            # Tokenisasi dan inference
            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = inputs.to(device)

            summary_ids = model.generate(
                inputs,
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Statistik
        original_len = len(text_input.split())
        summary_len = len(summary.split())
        reduction = 100 - (summary_len / original_len * 100)

        # Tampilkan ringkasan
        st.markdown("### 📝 Ringkasan:")
        st.success(summary)

        # Kata kunci penting
        words = re.findall(r'\w+', summary.lower())
        keyword_counts = Counter(words).most_common(5)
        keywords = [kw for kw, _ in keyword_counts if len(kw) > 3]
        st.markdown("**🔍 Kata Kunci Penting:** " + ", ".join(keywords))

        # Statistik visual
        st.markdown("### 📊 Statistik Ringkasan")
        st.write(f"🔢 Panjang Teks Asli: {original_len} kata")
        st.write(f"📉 Panjang Ringkasan: {summary_len} kata")
        st.write(f"📊 Efisiensi Ringkasan: {reduction:.2f}%")

        fig, ax = plt.subplots()
        ax.bar(["Teks Asli", "Ringkasan"], [original_len, summary_len], color=["skyblue", "salmon"])
        ax.set_ylabel("Jumlah Kata")
        st.pyplot(fig)

        # Tombol download ringkasan
        st.download_button("💾 Download Ringkasan", summary, file_name="ringkasan.txt")

# Footer
st.markdown("---")
st.markdown("<center>Made by Raihan</center>", unsafe_allow_html=True)
