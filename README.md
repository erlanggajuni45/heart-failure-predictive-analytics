# Laporan Proyek Machine Learning - Erlangga Juni Saputra

## Domain Proyek

Penyakit Kardiovaskular (*Cardiovascular Diseases* / CVDs) merupakan penyebab utama kematian di seluruh dunia. Menurut data dari Organisasi Kesehatan Dunia (*World Health Organization* / WHO), diperkirakan 19.8 juta orang meninggal di tahun 2022, 32% diantaranya disebabkan oleh penyakit CVDs. Dari kematian tersebut, 85% disebabkan oleh serangan jantung dan stroke. Sebagian besar penyakit kardiovaskular sebenarnya dapat dicegah melalui penanganan dini terhadap faktor risiko seperti tekanan darah tinggi, peningkatan kadar kolesterol, glukosa darah tinggi, serta pemantauan aktivitas fisik yang tepat. 

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan:**
* Diagnosis penyakit jantung yang terlambat sering kali berakibat fatal. Pemeriksaan medis konvensional sering kali memerlukan waktu, peralatan khusus, dan analisis mendalam dari tenaga medis spesialis yang ketersediaannya terbatas di beberapa fasilitas kesehatan.
* Dengan memanfaatkan riwayat rekam medis dan data klinis pasien, model *predictive analytics* berbasis *machine learning* dapat dikembangkan sebagai sistem pendukung keputusan. Sistem ini memungkinkan deteksi dini risiko penyakit gagal jantung secara cepat dan non-invasif sehingga tenaga medis dapat segera mengambil tindakan preventif atau intervensi medis sebelum kondisi pasien memburuk.

**Referensi:**
1. World Health Organization (WHO). (2025). *Cardiovascular disease (CVDs)*. WHO Fact Sheets.
2. Fedesoriano. (2021). *Heart Failure Prediction Dataset*. Kaggle.
3. Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J. J., Sandhu, S., Guppy, K. H., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. The American Journal of Cardiology, 64(5), 304-310.


## Business Understanding

Pengembangan model *predictive analytics* ini ditujukan untuk menfasilitasi deteksi dini risiko penyakit jantung secara efisien menggunakan data klinis non-invasif pasien.

### Problem Statements
* Dari serangkaian fitur rekam medis pasien, faktor klinis apa saja yang paling berkorelasi kuat terhadap risiko penyakit jantung?
* Bagaimana membangun model *machine learning* yang mampu mengidentifikasi risiko penyakit jantung pasien secara akurat berdasarkan fitur klinis tersebut?

### Goals
* Mengetahui karakteristik dan fitur klinis yang memiliki korelasi dominan terhadap diagnosis penyakit jantung melalui analisis data eksploratif.
* Membangun model klasifikasi *machine learning* dengan metrik evaluasi yang optimal untuk memprediksi potensi penyakit jantung secara andal.

### Solution Statements
Untuk mencapai target yang telah ditetapkan, diajukan beberapa pendekatan solusi berikut:
* Membangun dan membandingkan tiga algoritma dengan karakteristik berbeda:
  1. *K-Nearest Neighbors* (KNN) sebagai model berbasis jarak.
  2. *Random Forest Classifier* sebagai representasi metode *ensemble bagging*.
  3. *XGBoost Classifier* sebagai representasi metode *ensemble boosting*.
* Melakukan optimasi hyperparameter menggunakan *Grid Search Cross-Validation* pada model *Random Forest* untuk mencari kombinasi parameter terbaik guna meningkatkan generalisasi model.
* Kinerja seluruh model dievaluasi dan dibandingkan menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *ROC-AUC* pada *test set* independen.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
