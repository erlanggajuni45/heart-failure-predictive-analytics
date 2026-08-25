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
1. **Pengecekan Data Duplikat dan Outlier**
   * Berdasarkan fungsi `df.duplicated().sum()`, tidak ditemukan adanya data duplikat.
   * Visualisasi boxplot menunjukkan adanya beberapa titik outlier pada fitur `RestingBP`, `Cholesterol`, dan `Oldpeak`. Namun nilai-nilai tersebut tetap dipertahankan karena secara medis nilai ekstrem tersebut merepresentasikan kondisi klinis nyata pasien darurat kardiovaskular.
2. **Kondisi dan Kualitas Data**
   * Fitur Target `HeartDisease` memiliki distribusi yang relatif seimbang, yaitu 508 pasien positif dan 410 pasien negatif, sehingga data tidak memerlukan teknik *resampling* ekstrem.
   * Ditemukan nilai $0$ tidak wajar pada `RestingBP` dan `Cholesterol`, mengingat secara fisiologis tekanan darah dan kolesterol manusia tidak mungkin bernilai 0.
3. **Analisi Univariate**
   * Mayoritas pasien berjenis kelamin laki-laki, yaitu 725 pasien dibandingkan perempuan yang hanya sebanyak 193 pasien.
   * Sebagian besar pasien berada pada rentang usia 47 hingga 60 tahun.
4. **Analisis Multivariate & Korelasi**
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
Pada tahap ini, dibangun tiga algoritma *machine learning* yang berbeda untuk menyelesaikan masalah klasifikasi biner deteksi penyakit jantung. Selain itu, dilakukan *hyperparameter tuning* pada model *ensemble* untuk memaksimalkan performa.

### 1. K-Nearest Neigbors (KNN)
* **Cara Kerja**: Mengklasifikasikan data uji berdasarkan mayoritas label dari sejumlah $k$ tetangga terdekatnya menggunakan perhitungan jarak Euclideann. Pada proyek ini digunakan parameter default $k=5$.
* **Kelebihan**: Sederhana, tidak memiliki asumsi terhadap distribusi data dan tidak membutuhkan fase training yang kompleks.
* **Kekurangan**: Sangat lambat saat memprediksi data dalam jumlah besar, sensitif terhadap *outlier*, dan rentan terhadap fitur yang tidak relevaan jika jarak tidak terstandarisasi.

### 2. Random Forest Classifier
* **Cara Kerja**: Merupakan metode *ensemble bagging* yang membangun puluhan/ratusan pohon keputusan secara paralel dengan subset data dan subset fitur yang berbeda, kemudian menggabungkan hasil prediksinya melalui *majority voting*.
* **Kelebihan**: Sangat tahan terhadap **overfitting**, mampu menangani hubungan non-linear yang kompleks, serta stabil terhadap data dengan banyak fitur.
* **Kekurangan**: Model bersifat *black-box* dan membutuhkan memori komputasi lebih besar.

### 3. XGBoost (Extreme Gradient Boosting)
* **Cara Kerja**: Merupakan metode *ensemble boosting* berbasis *gradient boosting framework* yang membangun pohon keputusan secara sekuensial, di mana setiap pohon baru bertugas mengoreksi residual error dari pohon-pohon sebelumnya dengan regularisasi L1/L2.
* **Kelebihan**: kecepatan eksekusi tinggi berkat optimasi komputasi paralel, memiliki mekanisme regularisasi internal untuk mencegah *overfitting*, serta performa yang umumnya sangat unggul pada data tabular.
* **Kekurangan**: Memiliki banyak parameter kompleks yang memerlukan tuning intensif dan relatif sensitif terhadap *noisy labels*.

---

### Hyperparameter Tuning
Optimasi hyperparameter dilakukan pada model **Random Forest** menggunakan `GridSearchCV` dengan 5-Fold Cross-Validation (`cv=5`) dan metrik penentu skor `F1`.

* **Ruang Parameter yang Diuji**
  * `n_estimators`: `[50, 100, 200]`
  * `max_depth`: `[4, 6, 8, 10]`
  * `min_samples_split`: `[2, 5, 10]`
* **Hasil Parameter Terbaik**
  * `max_depth`: `6`
  * `min_samples_split`: `5`
  * `n_estimators`: `50`

---

### Pemilihan Model Terbaik
Berdasarkan hasil pengujian pada *test set*, *Random Forest** tanpa tuning dipilih sebagai model terbaik untuk solusi akhir karena mencatatkan performa tertinggi secara konsisten dengan **Akurasi 87.50%**, **F1-Score 0.8667**, dan **ROC-AUC 0.9332**. Model ini memberikan trade-off yang optimal antara tingkat presisi dan sensitivitas dalam mendeteksi pasien yang berisiko penyakit jantung.

## Evaluation
Metrik evaluasi yang digunakan untuk mengukur kinerja klasifikasi biner ini meliputi **Accuracy**, **Precision**, **Recall**, **F1-Score**, dan **ROC-AUC**.

### 1. penjelasan dan Formula Metrik

* **Accuracy**: Mengukur proporsi total prediksi yang benar terhadap keseluruhan data uji.
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Precision**: Mengukur ketepatan model dalam memprediksi pasien yang benar-benar sakit dari seluruh pasien yang diprediksi positif sakit.
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall (Sensitivity)**: Mengukur kemampuan model dalam menangkap seluruh pasien yang sebenarnya menderita penyakit jantung (*sangat krusial dalam konteks medis untuk meminimalkan False Negative*).
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **F1-Score**: Rata-rata harmonik antara *Precision* dan *Recall* untuk memberikan gambaran kinerja yang seimbang.
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **ROC-AUC**: Mengukur kemampuan model dalam membedakan antara kelas positif dan negatif pada berbagai batas *threshold*.

---

### 2. Hasil Evaluasi pada Data Uji
Berikut adalah ringkasan perbandingan metrik evaluasi dari model-model yang diuji:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | **87.50%** | **89.11%** | **88.24%** | **0.8867** | **0.9332** |
| Random Forest (Tuned) | 86.96% | 89.00% | 87.25% | 0.8812 | 0.9296 |
| XGBoost | 85.87% | 88.00% | 86.27% | 0.8713 | 0.9142 |
| KNN | 84.24% | 87.63% | 83.33% | 0.8543 | 0.9244 |

---

### 3. Analisis Confusion Matrix
Berdasarkan hasil pengujian Random Forest pada 184 sampel data uji:
* **True Positive (TP)**: 90 pasien terdeteksi memiliki penyakit jantung dengan tepat.
* **True Negative (TN)**: 71 pasien normal terdeteksi sehat dengan tepat.
* **False Positive (FP)**: 11 pasien sehat salah didiagnosis berpenyakit jantung.
* **False Negative (FN)**: 12 pasien yang harusnya berpenyakit jantung didiagnosis normal.

---

### 4. Dampak terhadap Business Understanding

* **Menjawab Problem Statement 1**: Analisis korelasi dan distribusi menunjukkan bahwa fitur `Oldpeak`, `MaxHR`, `ExerciseAngina`, `ChestPainType` (tipe ASY), dan bentuk `ST_Slope` (Flat/Down) merupakan indikator klinis paling dominan terhadap risiko gagal jantung.
* **Mencapai Goals**: Model *Random Forest* berhasil melampaui target performa dengan mencapai **Akurasi 87.50%** dan **F1-Score 0.8667**.
* **Dampak Solusi**: Tingkat *Recall* sebesar $88.24\%$ dan *ROC-AUC* sebesar $0.9332$ membuktikan bahwa sistem pendukung keputusan ini memiliki sensitivitas tinggi untuk menyaring pasien berisiko tinggi secara akurat, membantu tenaga medis melakukan intervensi dini secara cepat, serta mengurangi risiko fatalitas akibat keterlambatan penanganan.
