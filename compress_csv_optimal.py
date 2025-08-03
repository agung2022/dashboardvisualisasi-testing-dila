#!/usr/bin/env python3
"""
Script untuk mengkompresi file CSV ke ukuran 20-25MB untuk upload GitHub
Hasil tetap dalam format CSV dengan kompresi yang tidak terlalu agresif
"""

import pandas as pd
import numpy as np
import os

def check_file_size(filename):
    """Mengecek ukuran file dalam MB"""
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        return size_mb
    return 0

def compress_csv_moderate(input_file, output_file, target_size_mb=22):
    """
    Kompresi file CSV dengan target ukuran tertentu (default 22MB)
    Menggunakan strategi kompresi moderat
    """
    
    print(f"🔍 Memuat file: {input_file}")
    df = pd.read_csv(input_file)
    
    original_size = check_file_size(input_file)
    print(f"📏 Ukuran file asli: {original_size:.2f} MB")
    print(f"📊 Shape data asli: {df.shape}")
    print(f"🎯 Target ukuran: {target_size_mb} MB")
    
    # Hitung rasio kompresi yang dibutuhkan
    compression_ratio = target_size_mb / original_size
    print(f"📉 Rasio kompresi yang dibutuhkan: {compression_ratio:.2f}")
    
    # Strategi 1: Jika rasio > 0.8, hanya kurangi presisi
    if compression_ratio > 0.8:
        print("🔧 Menggunakan strategi: Pengurangan presisi saja")
        df_compressed = df.copy()
        
        # Kurangi presisi embedding ke 4 digit desimal
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        print(f"🔢 Mengurangi presisi {len(embedding_cols)} kolom embedding...")
        
        for col in embedding_cols:
            df_compressed[col] = df_compressed[col].round(4)
    
    # Strategi 2: Jika rasio 0.5-0.8, kurangi presisi + sampling ringan
    elif compression_ratio > 0.5:
        print("🔧 Menggunakan strategi: Pengurangan presisi + sampling ringan")
        
        # Sampling dengan mempertahankan distribusi label
        target_samples = int(len(df) * 0.7)  # Ambil 70% data
        
        df_pos = df[df['label'] == 1]
        df_neg = df[df['label'] == 0]
        
        pos_samples = int(target_samples * len(df_pos) / len(df))
        neg_samples = target_samples - pos_samples
        
        df_pos_sampled = df_pos.sample(n=min(pos_samples, len(df_pos)), random_state=42)
        df_neg_sampled = df_neg.sample(n=min(neg_samples, len(df_neg)), random_state=42)
        
        df_compressed = pd.concat([df_pos_sampled, df_neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Kurangi presisi
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        for col in embedding_cols:
            df_compressed[col] = df_compressed[col].round(4)
        
        print(f"📊 Data setelah sampling: {df_compressed.shape}")
    
    # Strategi 3: Jika rasio < 0.5, sampling lebih agresif + presisi lebih rendah
    else:
        print("🔧 Menggunakan strategi: Sampling moderat + pengurangan presisi")
        
        # Sampling dengan target 60% data
        target_samples = int(len(df) * 0.6)
        
        df_pos = df[df['label'] == 1]
        df_neg = df[df['label'] == 0]
        
        pos_samples = int(target_samples * len(df_pos) / len(df))
        neg_samples = target_samples - pos_samples
        
        df_pos_sampled = df_pos.sample(n=min(pos_samples, len(df_pos)), random_state=42)
        df_neg_sampled = df_neg.sample(n=min(neg_samples, len(df_neg)), random_state=42)
        
        df_compressed = pd.concat([df_pos_sampled, df_neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Kurangi presisi ke 3 digit desimal
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        for col in embedding_cols:
            df_compressed[col] = df_compressed[col].round(3)
        
        print(f"📊 Data setelah sampling: {df_compressed.shape}")
    
    # Simpan file terkompresi
    print(f"💾 Menyimpan file terkompresi...")
    df_compressed.to_csv(output_file, index=False)
    
    # Cek hasil
    compressed_size = check_file_size(output_file)
    compression_achieved = (1 - compressed_size/original_size) * 100
    
    print(f"\n✅ HASIL KOMPRESI:")
    print(f"📏 Ukuran file hasil: {compressed_size:.2f} MB")
    print(f"📉 Kompresi tercapai: {compression_achieved:.1f}%")
    print(f"📊 Shape data hasil: {df_compressed.shape}")
    print(f"📈 Distribusi label hasil:")
    print(df_compressed['label'].value_counts().sort_index())
    
    # Validasi distribusi label
    original_dist = df['label'].value_counts(normalize=True).sort_index()
    compressed_dist = df_compressed['label'].value_counts(normalize=True).sort_index()
    
    print(f"\n📊 PERBANDINGAN DISTRIBUSI LABEL:")
    print(f"Original  - Negatif: {original_dist[0]:.1%}, Positif: {original_dist[1]:.1%}")
    print(f"Compressed - Negatif: {compressed_dist[0]:.1%}, Positif: {compressed_dist[1]:.1%}")
    
    if compressed_size <= 25:
        if compressed_size >= 20:
            print(f"🎉 PERFECT! File berukuran {compressed_size:.2f} MB (dalam target 20-25MB)")
        else:
            print(f"✅ SUCCESS! File berukuran {compressed_size:.2f} MB (di bawah 25MB)")
        return True
    else:
        print(f"⚠️ File masih {compressed_size:.2f} MB (di atas 25MB)")
        return False

def main():
    input_file = "program_MSIB_labeled_and_embedded_v2.csv"
    
    # Cek apakah file input ada
    if not os.path.exists(input_file):
        print(f"❌ File {input_file} tidak ditemukan!")
        print("Pastikan file ada di direktori yang sama dengan script ini.")
        return
    
    print("🗜️ KOMPRESI FILE CSV UNTUK GITHUB")
    print("=" * 50)
    
    # Coba beberapa target ukuran jika yang pertama tidak berhasil
    target_sizes = [22, 20, 18, 15]
    
    for i, target_size in enumerate(target_sizes):
        output_file = f"program_MSIB_compressed_{target_size}mb.csv"
        print(f"\n--- PERCOBAAN {i+1}: Target {target_size}MB ---")
        
        success = compress_csv_moderate(input_file, output_file, target_size)
        
        if success:
            print(f"\n🎯 REKOMENDASI:")
            print(f"   Gunakan file: {output_file}")
            print(f"   Update path di streamlit_dila.py ke: './{output_file}'")
            break
        
        # Jika tidak berhasil dan bukan percobaan terakhir, coba target yang lebih kecil
        if not success and i < len(target_sizes) - 1:
            print(f"Mencoba target yang lebih kecil...")
            continue
    
    print(f"\n📋 FILE YANG DIHASILKAN:")
    for target_size in target_sizes:
        filename = f"program_MSIB_compressed_{target_size}mb.csv"
        if os.path.exists(filename):
            size = check_file_size(filename)
            status = "✅" if size <= 25 else "❌"
            print(f"   {filename}: {size:.2f} MB {status}")

if __name__ == "__main__":
    main()
