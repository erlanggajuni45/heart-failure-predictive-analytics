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
Dataset yang digunakan dalam proyek ini adalah **Heart Failure Prediction Dataset** yang diperoleh dari platform publik [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). Dataset ini merupakan hasil kurasi dan penggabungan 5 dataset penyakit jantung independen dari repositori UCI dengan total **918 baris data** dan **12 fitur** tanpa adanya *missing value* eksplisit.

### Variabel-variabel pada Heart Failure Prediction Dataset:
* **Age**: Usia pasien (tahun) [Numerik: 28 - 77]
* **Sex**: Jenis kelamin pasien [Kategorik: `M` = Male, `F` = Female]
* **ChestPainType**: Tipe nyeri data [Kategorik: `TA` = Typical Angina, `ATA` = Atypical Angina, `NAP` = Non-Anginal Pain, `ASY` = Asymptomatic]
* **RestingBP**: Tekanan darah istirahat (mm Hg) [Numerik: 0 - 200]
* **Cholesterol**: Kadar kolesterol serum (mg/dl) [Numerik: 0 - 603]
* **FastingBS**: Kadar gula darah puasa [Kategorik Biner: `1` jika FastingBS > 120 mg/dl, `0` jika sebaliknya]
* **RestingECG**: Hasil elektrokardiogram saat istirahat [Kategorik: `Normal` = Normal, `ST` = memiliki kelainan gelombang ST-T, `LVH` = hipertrofi ventrikel kiri]
* **MaxHR**: Detak jantung maksimum yang dicapai [Numerik: 60 - 202]
* **ExerciseAngina**: Angina yang dipicu oleh aktivitas fisik/olahraga [Kategorik: `Y` = Yes, `N` = No]
* **Oldpeak**: Depresi segmen ST yang diinduksi oleh latihan relatif terhadap istirahat [Numerik: -2.6 - 6.2]
* **ST_Slope**: Kemiringan puncak segmen ST saat latihan puncak [Kategorik: `UP` = Upsloping, `Flat` = Flat, `Down` = Downsloping]
* **HeartDisease**: Label target diagnosis penyakit jantung [Kategorik Biner: `1` = Memiliki penyakit jantung, `0` = Sehat]

### Exploratory Data Analysis (EDA)

Berdasarkan tahapan eksplorasi data yang telah dilakukan, diperoleh beberapa wawasan penting:
1. **Kondisi dan Kualitas Data**
   * Fitur Target `HeartDisease` memiliki distribusi yang relatif seimbang, yaitu 508 pasien positif dan 410 pasien negatif, sehingga data tidak memerlukan teknik *resampling* ekstrem.
   * Ditemukan nilai $0$ tidak wajar pada `RestingBP` dan `Cholesterol`, mengingat secara fisiologis tekanan darah dan kolesterol manusia tidak mungkin bernilai 0.
2. **Analisi Univariate**
   * Mayoritas pasien berjenis kelamin laki-laki, yaitu 725 pasien dibandingkan perempuan yang hanya sebanyak 193 pasien.
   * Sebagian besar pasien berada pada rentang usia 47 hingga 60 tahun.
3. **Analisis Multivariate & Korelasi**
   * `Oldpeak`, `Age`, dan `FastingBS` memiliki korelasi positif signifikan dengan `HeartDisease`.
   * `MaxHR` memiliki korelasi negatif kuat, mengindikasikan bahwa pasien yang tidak mampu mencapai detak jantung maksimal tinggi saat beraktivitas memiliki risiko gagal jantung lebih besar.
   * Pasien dengan tipe nyeri data `ASY` mengalami `ExerciseAngina` = `Y`, serta bentuk `ST_Slope` bertipe `FLAT` atau `Down` memiliki proporsi kasus positif penyakit jantung yang lebih dominan dibanding kategori lainnya.
   

## Data Preparation
Tahap persiapan data dilakukan secara berurutan untuk memastikan dataa siap digunakan oleh model *machine learning* dan mencegah terjadinya *data leakage*.

### Tahapan Data Preparation yang Dilakukan
1. **Penanganan Nilai 0 Anomali**
   * **Proses**: Nilai 0 pada fitur `RestingBP` diganti menggunakan nilai median dari baris non-nol. Pada kolom `Cholesterol`, nilai 0 diimputasi menggunakan median dari data valid non-nol.
   * **Alasan**: Secara medis, tekanan darah dan kadar kolesterol serum manusia tidak mungkin bernilai 0. Nilai ini merupakan *implicit missing value*. Imputasi median dipilih karena lebih tahan terhadap pengaruh nilai ekstrim dibandingkan mean.
2. **Transformasi Fitur Kategorikal**
   * **Proses**: Menerapkan teknik *One-Hot Encoding* menggunakan `pd.get_dummies()` dengan argumen `drop_first=True` pada seluruh variabel kategorikal teks.
   * **Alasan**: Algoritma *machine learning* berbasis matematika membutuhkan input numerik. Penerapan `drop_first=True` bertujuan untuk mencegah terjadinya *multicollinearity* atau *dummy variable trap*. Tahap ini menghasilkan 15 fitur prediktor.
3. **Pembagian Dataset**
   * **Proses**: Membagi dataset menjadi data latih sebesar 80% dan data uji sebesar 20% menggunakan `train_test_split` dengan parameter `stratify=y` dan `random_state=42`.
   * **Alasan**: Pembagian dataset diperlukan untuk mengevaluasi kemampuan generalisasi model terhadap data baru yang belum pernah dilihat sebelumnya. Parameter `stratify` memastikan proporsi kelas target `HeartDisease` pada data latih dan data uji tetap identik dengan distribusi populasi aslinya.
4. **Standarisasi Fitur Numerik**
   * **Proses**: Menstandarisasi 5 fitur numerik kontinu (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) menggunakan `StandardScaler`. Fungsi `fit_transform()` hanya diterapkan pada data latih, sedangkan data uji hanya diterapkan `transform()`.
   * **Alasan**: Algoritma seperti KNN dan *Gradient Boosting* sensitif terhadap perbedaan skala antar variabel. Standarisasi mengubah data menjadi berdistribusi dengan rata-rata 0 dan standar deviasi 1. Penerapan `fit` hanya pada data latih bertujuan mutlak untuk mencegah *data leakage* dari data uji.

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
