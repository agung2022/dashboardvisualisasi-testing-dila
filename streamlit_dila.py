import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import string
from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
# Hapus SMOTE import untuk menghindari error

st.set_page_config(layout="wide", page_title="Klasifikasi Stunting NBC")

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        if 'createdAt' in df.columns:
            df = df.drop(columns=['createdAt', 'year'], errors='ignore')
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file}' tidak ditemukan. Pastikan path file benar.")
        return None

def manual_oversample(X, y, random_state=42):
    """
    Manual oversampling untuk menggantikan SMOTE
    """
    np.random.seed(random_state)
    
    # Hitung jumlah sampel untuk setiap kelas
    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    
    X_resampled = []
    y_resampled = []
    
    for label in unique:
        # Ambil sampel untuk kelas ini
        mask = y == label
        X_class = X[mask]
        y_class = y[mask]
        
        # Jika kelas ini minority, lakukan oversampling
        current_count = len(X_class)
        if current_count < max_count:
            # Hitung berapa banyak sampel tambahan yang dibutuhkan
            additional_needed = max_count - current_count
            
            # Random sampling dengan replacement
            indices = np.random.choice(len(X_class), additional_needed, replace=True)
            X_additional = X_class[indices]
            y_additional = y_class[indices]
            
            # Gabungkan dengan data asli
            X_class_resampled = np.vstack([X_class, X_additional])
            y_class_resampled = np.hstack([y_class, y_additional])
        else:
            X_class_resampled = X_class
            y_class_resampled = y_class
        
        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)
    
    # Gabungkan semua kelas
    X_final = np.vstack(X_resampled)
    y_final = np.hstack(y_resampled)
    
    # Shuffle data
    indices = np.random.permutation(len(X_final))
    return X_final[indices], y_final[indices]

@st.cache_resource
def train_naive_bayes(_X_train, _y_train):
    model = GaussianNB()
    model.fit(_X_train, _y_train)
    return model

def main():
    st.sidebar.header("Konfigurasi Dataset")
    input_file = st.sidebar.text_input("Path File Data CSV", "./program_MSIB_labeled_and_embedded_v2.csv")

    st.markdown("""
        <h2 style="text-align:center; color:#2d3e50;">
            Dashboard Visualisasi Data Analisis Sentimen Program Magang dan Studi Independen Bersertifikat (MSIB) 
            <br>Menggunakan Algoritma <span style="color:#007bff;">Naive Bayes Classifier</span>
        </h2>
        <hr>
    """, unsafe_allow_html=True)

    df = load_data(input_file)
    if df is None:
        st.stop()

    emb_cols = [col for col in df.columns if col.startswith('embedding_')]
    if not emb_cols:
        st.error("Error: Kolom embeddings (dengan prefix 'embedding_') tidak ditemukan dalam file.")
        st.stop()

    X = df[emb_cols].values
    y = df['label'].values
    label_names = {0: "Negatif", 1: "Positif"}

    X_train_ori, X_test, y_train_ori, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Terapkan manual oversampling menggantikan SMOTE
    X_train, y_train = manual_oversample(X_train_ori, y_train_ori, random_state=42)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribusi Data (Sebelum & Sesudah Balancing)", 
        "Kata Populer & Wordcloud",
        "Modeling Naive Bayes", 
        "Evaluasi Model (Confusion Matrix & Classification Report)"
    ])

    # PERBAIKAN: Semua konten untuk tab1 dimasukkan ke dalam blok 'with' ini
    with tab1:
        st.header("Distribusi Data: Label & Pembagian Train/Test")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("Jumlah Data per Label (Dataset Asli)")
            vc = pd.Series(y).value_counts().rename(index=label_names)
            df_vc = vc.to_frame('Jumlah Data')
            total = pd.DataFrame({'Jumlah Data': [df_vc['Jumlah Data'].sum()]}, index=['Total'])
            st.dataframe(pd.concat([df_vc, total]))

        with col2:
            st.markdown("Visualisasi Distribusi Label (Dataset Asli)")
            fig_vc, ax_vc = plt.subplots(figsize=(8, 4))
            sns.barplot(x=vc.index, y=vc.values, ax=ax_vc, palette="viridis")
            ax_vc.set_title("Distribusi Label pada Keseluruhan Dataset")
            for i, bar in enumerate(ax_vc.patches):
                bar_val = int(bar.get_height())
                ax_vc.text(bar.get_x() + bar.get_width()/2., bar.get_height(), bar_val, ha='center', va='bottom')
            st.pyplot(fig_vc)

        st.markdown("---")
        st.subheader("Distribusi Data Setelah Balancing (Data Latih)")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Sebelum Balancing**")
            fig1, ax1 = plt.subplots()
            sns.countplot(x=y_train_ori, palette="rocket", ax=ax1)
            ax1.set_xticklabels(label_names.values())
            ax1.set_title("Data Training Asli")
            st.pyplot(fig1)

        with col4:
            st.markdown("**Setelah Balancing**")
            fig2, ax2 = plt.subplots()
            sns.countplot(x=y_train, palette="crest", ax=ax2)
            ax2.set_xticklabels(label_names.values())
            ax2.set_title("Data Training Setelah Balancing")
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Seluruh Hasil Embedding dari Dataset")

        st.markdown("### Sampel Data Hasil Pembagian")
        idx_full = np.arange(len(df))
        idx_train, idx_test = train_test_split(idx_full, test_size=0.2, stratify=y, random_state=42)
        train_df = df.iloc[idx_train]
        test_df = df.iloc[idx_test]

        tab_train, tab_test = st.tabs(["Data Latih (Train)", "Data Uji (Test)"])
        with tab_train:
            st.dataframe(train_df.reset_index(drop=True), use_container_width=True, height=300)
            csv_train = train_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data Train (.csv)",
                data=csv_train,
                file_name="embedding_train.csv",
                mime="text/csv"
            )
        with tab_test:
            st.dataframe(test_df.reset_index(drop=True), use_container_width=True, height=300)
            csv_test = test_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data Test (.csv)",
                data=csv_test,
                file_name="embedding_test.csv",
                mime="text/csv"
            )

    with tab2:
        st.header("Kata Paling Sering Muncul dan Wordcloud per Sentimen")
        text_col = "teks"
        if text_col not in df.columns:
            st.warning(f"Kolom '{text_col}' tidak ditemukan. Visualisasi kata tidak dapat dibuat.")
        else:
            col_pos, col_neg = st.columns(2)

            def generate_word_visuals(texts, palette, wc_background, wc_colormap):
                cv = CountVectorizer(stop_words='english')
                corpus = " ".join(texts.astype(str).dropna().tolist()).lower().translate(str.maketrans('', '', string.punctuation))
                if not corpus.strip():
                    st.write("Tidak ada teks untuk dianalisis.")
                    return
                X_corpus = cv.fit_transform([corpus])
                word_freq = dict(zip(cv.get_feature_names_out(), X_corpus.toarray()[0]))
                common_words = Counter(word_freq).most_common(10)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=[x[1] for x in common_words], y=[x[0] for x in common_words], ax=ax, palette=palette)
                ax.set_title("10 Kata Paling Sering Muncul")
                st.pyplot(fig)
                wc = WordCloud(width=400, height=250, background_color=wc_background, colormap=wc_colormap).generate(corpus)
                st.image(wc.to_array(), use_container_width=True)

            with col_pos:
                st.subheader(":green[Sentimen Positif]")
                pos_texts = df[df['label'] == 1][text_col]
                generate_word_visuals(pos_texts, "crest", "white", "Greens")

            with col_neg:
                st.subheader(":red[Sentimen Negatif]")
                neg_texts = df[df['label'] == 0][text_col]
                generate_word_visuals(neg_texts, "rocket", "black", "Reds")

    # Latih model sekali saja
    model = train_naive_bayes(X_train, y_train)

    with tab3:
        st.header("Modeling Naive Bayes")
        st.success("Model dilatih dengan data hasil balancing!")

        # Lakukan prediksi pada data latih dan uji
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        def get_metrics(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            return acc, prec, rec, f1

        train_metrics = get_metrics(y_train, y_pred_train)
        test_metrics = get_metrics(y_test, y_pred_test)

        report_df = pd.DataFrame({
            'Akurasi':  [train_metrics[0], test_metrics[0]],
            'Presisi':  [train_metrics[1], test_metrics[1]],
            'Recall':   [train_metrics[2], test_metrics[2]],
            'F1-Score': [train_metrics[3], test_metrics[3]]
        }, index=['Data Latih (Train)', 'Data Uji (Test)'])

        st.dataframe(report_df.round(4))

        st.markdown("---")
        st.subheader("Visualisasi Perbandingan Performa")
        fig_perf, ax_perf = plt.subplots()
        report_df.T.plot(kind='bar', figsize=(10, 6), colormap='Paired', rot=0, ax=ax_perf)
        ax_perf.set_title("Performa Model Naive Bayes")
        ax_perf.set_ylabel("Skor")
        ax_perf.set_ylim(0, 1.1)
        for container in ax_perf.containers:
            ax_perf.bar_label(container, fmt='%.3f')
        st.pyplot(fig_perf)

    with tab4:
        st.header("Evaluasi Model (Confusion Matrix & Classification Report)")
        
        # Gunakan prediksi yang sudah dihitung sebelumnya
        y_pred_test = model.predict(X_test)

        col1_eval, col2_eval = st.columns(2)
        with col1_eval:
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, display_labels=label_names.values(), ax=ax_cm, cmap='Blues')
            st.pyplot(fig_cm)

        with col2_eval:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred_test, target_names=label_names.values(), output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(4))

        st.markdown("---")
        st.subheader("Analisis Hasil Prediksi Salah")
        idx_full = np.arange(len(df))
        _, idx_test = train_test_split(idx_full, test_size=0.2, stratify=y, random_state=42)
        test_df_reset = df.iloc[idx_test].copy().reset_index(drop=True)
        eval_table = test_df_reset[['teks', 'label']].copy()
        eval_table.rename(columns={'label': 'Label Asli'}, inplace=True)
        eval_table['Prediksi'] = y_pred_test
        eval_table['Status'] = np.where(eval_table['Label Asli'] == eval_table['Prediksi'], 'Benar ✅', 'Salah ❌')
        eval_table['Label Asli'] = eval_table['Label Asli'].map(label_names)
        eval_table['Prediksi'] = eval_table['Prediksi'].map(label_names)

        misclassified_df = eval_table[eval_table['Status'] == 'Salah ❌']
        st.write(f"Jumlah data yang salah diklasifikasikan: {len(misclassified_df)} dari {len(y_test)} data uji.")
        st.dataframe(misclassified_df, use_container_width=True)
        csv_export = misclassified_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data Salah Prediksi (.csv)",
            data=csv_export,
            file_name="hasil_prediksi_salah_nb.csv",
            mime="text/csv",
        )

# PERBAIKAN: Menggunakan __name__ (dua garis bawah)
if __name__ == "__main__":
    main()
