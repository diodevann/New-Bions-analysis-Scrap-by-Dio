# New-Bions-analysis-by-Dio
# 📈 BIONS Infrastructure & Market Stress Analysis
**Portofolio Proyek - BNI Sekuritas Internship Program 2026-2027**

Proyek ini bertujuan untuk mengidentifikasi akar penyebab keluhan pengguna pada aplikasi BIONS (BNI Sekuritas) melalui pendekatan berbasis data (*data-driven*). Menggunakan *Natural Language Processing* (NLP) pada puluhan ribu ulasan publik, analisis ini membuktikan bahwa stabilitas infrastruktur pada jam perdagangan bursa merupakan metrik krusial yang berdampak langsung pada potensi kehilangan transaksi (hilangnya *fee* broker) dan *churn* nasabah.

**Oleh:** I Gede Dio Devan Satria Prawira
**Institusi:** Mekatronika dan Kecerdasan Buatan, Universitas Pendidikan Indonesia (UPI)  
**Divisi Pilihan:** IT - Application Development & IT - Project Management Office

---

## 🎯 Objektif Proyek
1. **Analisis Komparatif (ABSA):** Mengekstrak dan membandingkan sentimen negatif pengguna BIONS terhadap 4 kompetitor utama (Ajaib, IPOT, MOST, Stockbit) menggunakan *Aspect-Based Sentiment Analysis*.
2. **Korelasi Data Pasar (Event Study):** Mengukur apakah lonjakan keluhan infrastruktur (lambat, *error*, gagal *login*) berkorelasi dengan volatilitas dan volume transaksi di Bursa Efek Indonesia (IHSG).
3. **Kalkulasi Dampak Bisnis:** Memodelkan potensi kerugian finansial (*fee* berisiko) akibat kegagalan eksekusi *order* di hari-hari saat pasar sedang stres.

---

## 🔬 Metodologi
*   **Pengumpulan Data:** *Scraping* >22.000 ulasan Google Play Store dari aplikasi BIONS dan kompetitor (periode penarikan hingga Nov 2026).
*   **Pemrosesan Bahasa Alami (NLP):** Pembuatan kamus aspek (*dictionary-based*) dan klasifikasi sentimen per klausa untuk memisahkan keluhan UI/UX, biaya, layanan pelanggan, dan **infrastruktur/kecepatan**. Akurasi pelabelan infrastruktur telah divalidasi secara manual dengan tingkat akurasi **99%**.
*   **Analisis Runtun Waktu (*Time-Series*):** Penggabungan data ulasan harian dengan data historis IHSG (Yahoo Finance) untuk melakukan *event study* pada hari-hari dengan volatilitas dan volume ekstrem.

---

## 📊 Temuan Utama

### 1. Gap Infrastruktur Sangat Signifikan
Keluhan mengenai **"Kecepatan & Latency"** adalah aspek yang paling menjauhkan BIONS dari standar industri. Sebanyak **29,2%** keluhan BIONS berpusat pada masalah kecepatan/infrastruktur, dibandingkan dengan median kompetitor yang hanya **5,4%** (Gap: +23,9 *percentage points*, signifikan secara statistik pada CI 95%).

### 2. Keluhan Menumpuk Saat Jam Perdagangan (Lift 1.23x)
Masalah infrastruktur BIONS bukan terjadi secara acak. Terdapat probabilitas yang lebih tinggi (55,6% vs 45,1%) bahwa pengguna mengalami dan melaporkan kegagalan teknis tepat saat jam perdagangan bursa sedang berlangsung (*market hours*).

### 3. Skenario Dampak Finansial (*Business Value*)
Mengacu pada rata-rata nilai transaksi harian BEI (Rp20,5 Triliun per Oktober 2026) dan tarif dasar BIONS (0,17% beli / 0,27% jual), masalah infrastruktur ini berpotensi membakar pendapatan perusahaan. 
Berdasarkan model kalibrasi dari sampel data:
*   Terdapat **~55 hari stres** di pasar per tahun (sekitar 14,4% dari total hari bursa).
*   Diestimasikan terjadi kehilangan **1% hingga 5%** transaksi nasabah (bergeser ke kompetitor atau gagal tereksekusi) khusus pada hari-hari stres tersebut.
*   *Setiap lonjakan trafik yang tertangani oleh server adalah transaksi yang tidak hilang dan nasabah yang tidak pindah.*

---

## 📁 Struktur Repositori

```text
├── data/
│   ├── reviews_absa.csv                  # Data ulasan yang telah dilabeli aspek dan sentimennya
│   ├── t0_overview.csv                   # Gambaran umum rating dan volume ulasan kelima aplikasi
│   ├── t1_aspect_gap.csv                 # Perhitungan statistik gap antar aspek
│   └── hasil_cek_akurasi_infra_neg.xlsx  # Log validasi manual akurasi NLP (99%)
│
├── notebooks/
│   └── bions_pipeline.py                 # Pipeline kode utama (Ekstraksi, ABSA, Korelasi, Event Study)
│
├── reports/
│   ├── summary.md                        # Ringkasan eksekutif dan interpretasi data
│   └── *.png                             # Visualisasi grafik (Gap Aspek, Distribusi Jam, dll)
│
└── README.md                             # Halaman ini
