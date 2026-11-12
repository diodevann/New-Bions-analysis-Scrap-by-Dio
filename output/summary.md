# RINGKASAN HASIL - isi placeholder slide dari angka di bawah

## [A0] Gambaran data
| app          |    n |   rating_mean |   share_1_2_star |   share_infra_neg | first                     | last                      |
|:-------------|-----:|--------------:|-----------------:|------------------:|:--------------------------|:--------------------------|
| Ajaib        | 8500 |       4.30529 |         0.156471 |          0.128588 | 2026-05-10 11:08:28+07:00 | 2026-11-10 07:42:01+07:00 |
| BIONS (BNIS) |  951 |       2.43954 |         0.610936 |          0.480547 | 2025-04-17 17:09:14+07:00 | 2026-11-10 02:57:11+07:00 |
| IPOT         | 3919 |       3.78617 |         0.280429 |          0.240112 | 2025-04-10 09:42:40+07:00 | 2026-11-10 11:25:35+07:00 |
| MOST         |  519 |       3.67245 |         0.304432 |          0.279383 | 2025-04-14 08:48:16+07:00 | 2026-11-10 12:18:05+07:00 |
| Stockbit     | 8428 |       4.014   |         0.2201   |          0.185691 | 2025-04-10 05:41:00+07:00 | 2026-11-10 13:59:44+07:00 |

## [A1] Aspek mana yang paling menjauhkan BIONS dari kompetitor?
Gap = (% ulasan BIONS berkeluhan aspek X) - (median kompetitor). CI = bootstrap 95%.

- Kecepatan & latency: BIONS 29.2% vs median 5.4% | gap +23.9 pp [+21.0, +27.1] -> SIGNIFIKAN (CI tidak memuat 0)
- Tampilan & fitur: BIONS 16.9% vs median 4.2% | gap +12.7 pp [+10.2, +15.3] -> SIGNIFIKAN (CI tidak memuat 0)
- Login & akses: BIONS 16.2% vs median 7.8% | gap +8.3 pp [+5.9, +11.2] -> SIGNIFIKAN (CI tidak memuat 0)
- Akurasi data & saldo: BIONS 7.9% vs median 3.6% | gap +4.3 pp [+2.4, +6.0] -> SIGNIFIKAN (CI tidak memuat 0)
- Stabilitas & error: BIONS 9.7% vs median 5.7% | gap +3.9 pp [+1.7, +6.1] -> SIGNIFIKAN (CI tidak memuat 0)
- Biaya: BIONS 0.3% vs median 0.4% | gap -0.1 pp [-0.4, +0.3] -> tidak signifikan
- Keamanan & kepercayaan: BIONS 0.4% vs median 0.5% | gap -0.1 pp [-0.5, +0.4] -> tidak signifikan
- Layanan pelanggan: BIONS 2.0% vs median 2.2% | gap -0.2 pp [-1.1, +0.8] -> tidak signifikan

Aspek dengan gap terbesar: **Kecepatan & latency** (infrastruktur).
[A1b] Porsi 'suara membantu' pada ulasan berkeluhan infrastruktur: 71.7%

## [A2a] Apakah keluhan infrastruktur menumpuk pada jam perdagangan?
Porsi ulasan keluhan infra yang diposting saat jam bursa: 55.6% vs ulasan lain 45.1% | lift 1.23x [1.09, 1.40] (n=457)
Catatan: waktu ulasan = waktu menulis, bukan waktu kejadian gangguan.

## [A2b] Korelasi harian (5 hasil terkuat)
aktivitas  lag_hari ukuran       rho        p   n
    vol_z         1 jumlah  0.216234 0.084638 362
  abs_ret         0 jumlah -0.133923 0.126958 381
    range         2  rasio -0.213851 0.133289 124
    range         1  rasio -0.211519 0.147284 124
  abs_ret         0  rasio -0.142109 0.176941 124

## [A2c] Event study - trading halt: n=3, rasio keluhan/hari 0.91x [0.00, 1.50] (hari acara 1.11 vs biasa 1.22)
## [A2c] Event study - top 5% |return|: n=19, rasio keluhan/hari 0.49x [0.30, 0.71] (hari acara 0.65 vs biasa 1.32)

[A2d] Ulasan BIONS setelah peluncuran resmi (2025-05-05): 939

## [A3] Sinyal dini (Google Trends)
- Tren pencarian vs volatilitas pasar: puncak di k=0 minggu, rho=0.38, p=0.000 (k>0 = tren mendahului)
- Tren pencarian vs keluhan BIONS: puncak di k=0 minggu, rho=0.14, p=0.195 (k>0 = tren mendahului)
- Granger tren -> pasar: tidak tersedia (pasang statsmodels)
- Granger pasar -> tren: tidak tersedia (pasang statsmodels)

## [A4] Apakah sinyal hari ini memprediksi hari stres besok? (lift > 1 = lebih baik dari menebak)
                 sinyal  n_sinyal  presisi    ci_lo    ci_hi  base_rate     lift
|return| kemarin >= P90        39 0.410256 0.270791 0.565817   0.144357 2.841958
volume kemarin >= +1 SD        82 0.024390 0.006714 0.084635   0.144357 0.168958

## Catatan & batasan wajib dicantumkan di slide/Q&A
- Uji korelasi dijalankan 18x; gunakan koreksi Bonferroni (p < 0.0028).
- Analisis Google Trends dilewati: grangercausalitytests() got an unexpected keyword argument 'verbose'
- Korelasi bukan kausalitas. Ulasan hanya mewakili pengguna yang menulis ulasan (bias seleksi).
- Volume Yahoo untuk ^JKSE adalah proksi; gunakan nilai/frekuensi transaksi BEI bila tersedia.
- Google Trends bernilai relatif (0-100) & bisa sedikit berubah antar penarikan; cache disimpan.
- Kampanye BIONS Cuanpionship #4 (8 Apr-27 Jun 2025) bisa mengacaukan lonjakan awal; lihat sensitivitas [A2d].
- Aspek diekstrak dengan kamus + sentimen per klausa (bukan model ABSA terlatih); validasi manual 100 ulasan (hitung akurasi) sebelum mengklaim hasil.