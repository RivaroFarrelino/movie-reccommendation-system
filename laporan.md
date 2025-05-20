# Laporan Proyek Machine Learning - \[Muhammad Rivaro Farrelino Gozali]

## Project Overview

Rekomendasi film merupakan salah satu aplikasi machine learning yang sangat populer di platform streaming dan e-commerce media hiburan. Dengan jutaan film dan pengguna, memberikan rekomendasi yang relevan dan personal menjadi tantangan utama dalam meningkatkan pengalaman pengguna dan retensi pelanggan.

Dalam proyek ini, saya menggunakan dataset MovieLens 25 juta rating film yang mencakup data rating, tag, dan metadata film, untuk membangun sistem rekomendasi film menggunakan dua pendekatan utama: content-based filtering dan collaborative filtering berbasis embedding menggunakan deep learning.

Sistem rekomendasi dapat membantu mengatasi masalah overload informasi (information overload) dengan menyajikan film yang paling sesuai preferensi pengguna berdasarkan histori rating dan kemiripan konten film. Studi sebelumnya menunjukkan bahwa gabungan metode content-based dan collaborative filtering dapat meningkatkan akurasi rekomendasi (Ricci et al., 2011).

Referensi:

* Ricci, F., Rokach, L., Shapira, B., & Kantor, P. B. (2011). *Recommender Systems Handbook*. Springer.

## Business Understanding

### Problem Statements

* Bagaimana memberikan rekomendasi film yang relevan dan personal kepada pengguna berdasarkan rating dan preferensi mereka?
* Bagaimana mengatasi data film dan pengguna yang sangat besar dan sparse (banyak nilai hilang)?
* Bagaimana membangun model rekomendasi yang efisien dengan memanfaatkan metadata film (seperti genre) dan histori rating pengguna?

### Goals

* Membangun sistem rekomendasi film dengan pendekatan content-based filtering yang memanfaatkan genre film menggunakan TF-IDF dan cosine similarity.
* Mengembangkan model collaborative filtering menggunakan embedding dan neural network untuk menangkap pola preferensi pengguna secara non-linear.
* Melakukan evaluasi model menggunakan metrik root mean squared error (RMSE) pada data validasi.
* Menyediakan rekomendasi film yang personalized bagi pengguna berdasarkan prediksi rating film yang belum mereka tonton.

### **Solution Approach**

Untuk mencapai tujuan di atas, proyek ini akan menggunakan dua pendekatan utama dalam pengembangan sistem rekomendasi:

#### **1. Content-Based Filtering**

* **Deskripsi**: Sistem akan merekomendasikan film berdasarkan kemiripan konten dengan film yang sebelumnya disukai pengguna.
* **Teknik**:

  * Menggunakan representasi teks dari genre dan tag film dengan metode **TF-IDF vectorization**.
  * Mengukur kemiripan antar film dengan **cosine similarity**.
  * Menghasilkan daftar film dengan skor kemiripan tertinggi terhadap film yang telah dinilai tinggi oleh pengguna.

#### **2. Collaborative Filtering dengan Deep Learning**

* **Deskripsi**: Sistem belajar dari pola rating pengguna terhadap film, untuk memprediksi preferensi pengguna terhadap film yang belum ditonton.
* **Teknik**:

  * Menggunakan pendekatan **Matrix Factorization** berbasis **Embedding Layer** untuk pengguna dan item.
  * Model dibangun dengan **Neural Network** yang menggabungkan embedding pengguna dan film, lalu memprediksi rating.
  * Dilatih menggunakan **loss function RMSE** atau **Mean Squared Error (MSE)**.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari situs MovieLens dan berjudul [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/#:~:text=MovieLens%2025M%20Dataset). Dataset ini tersedia dalam format ZIP dan berisi tujuh file terpisah, yaitu: `movies.csv`, `ratings.csv`, `genome-scores.csv`, `links.csv`, `tags.csv`, `genome-tags.csv`, dan `README.txt`. Berdasarkan informasi dalam file `README.txt`, MovieLens 25M (ml-25m) adalah kumpulan data yang mencakup informasi film dan rating yang diberikan oleh pengguna. Secara keseluruhan, terdapat 25.000.095 data rating yang diberikan oleh 162.541 pengguna terhadap berbagai film. Data ini dikumpulkan dari tanggal 9 Januari 1995 hingga 21 November 2019, yang juga merupakan tanggal terakhir pembaruan dataset ini.

* `movies.csv`: Informasi film termasuk movieId, judul, dan genre.
* `ratings.csv`: Rating film oleh pengguna dengan timestamp.
* `tags.csv`: Tag yang diberikan pengguna ke film.
* `links.csv`, `genome-tags.csv`, `genome-scores.csv`: Metadata tambahan terkait tag dan link film.

Setelah pembersihan, data difilter untuk menghilangkan film tanpa genre dan film dengan jumlah rating sangat sedikit (<50 rating), serta penghapusan duplikasi data film.

### Exploratory Data Analysis (EDA)

Proses EDA dilakukan untuk memahami struktur dan karakteristik dataset secara menyeluruh guna memperoleh insight dan pengetahuan yang berguna dalam pengembangan model.

**Univariate Analysis**

Dari enam file `.csv` yang tersedia dalam dataset, saya mengelompokkan data ke dalam tiga kategori utama berdasarkan ID-nya, yaitu data film, data pengguna (users), dan data skor relevansi film. Selanjutnya, saya melakukan analisis untuk melihat jumlah data pada masing-masing kategori.

* **Tag relevance scores**

  <p align='center'>
      <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/tag_relevance_score.png?raw=true" alt="relevance-score">
  </p>

* **Films**

  <p align='center'>
      <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/films.png?raw=true" alt="films">
  </p>

* **User ratings**

  <p align='center'>
      <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/user_ratings.png?raw=true" alt="users ratings">
  </p>

Berdasarkan ketiga visualisasi tersebut, terdapat 1.128 tag unik dalam data genome tag, 62.423 film unik, dan 162.541 pengguna unik. Saya menilai bahwa jumlah data pada skor relevansi berdasarkan `tagId` relatif kecil, hanya sekitar 1.000 data. Jumlah ini jauh lebih sedikit jika dibandingkan dengan data pengguna dan film yang masing-masing berjumlah sekitar 162 ribu dan 62 ribu. Oleh karena itu, saya memutuskan untuk fokus mengolah data pengguna dan film saja dalam pengembangan sistem rekomendasi.


## Data Preparation

Berikut adalah teknik-teknik yang digunakan dalam tahap Data Preparation:

Dalam proses persiapan data, langkah pertama yang saya lakukan adalah *data cleaning* terhadap file `movies.csv` dan `ratings.csv`. Setelah kedua dataset tersebut dibersihkan dan layak untuk digunakan, saya menggabungkannya menjadi satu *dataframe* utama sebagai dasar pengembangan sistem rekomendasi. Adapun tahapan yang dilakukan adalah sebagai berikut:

1. **Pembersihan Data pada `movies.csv`**
   Pada file `movies.csv`, informasi tahun rilis film dicantumkan bersamaan dengan judul film dalam satu kolom. Hal ini perlu ditangani agar tidak menyebabkan gangguan saat pelatihan model. Oleh karena itu, saya memisahkan tahun rilis ke kolom tersendiri untuk memperjelas struktur data.

2. **Pembersihan Data pada `ratings.csv`**
   Kolom *rating* dalam file `ratings.csv` menunjukkan nilai yang tersebar dalam interval 0.5 hingga 5.0. Karena distribusi ini tidak merata, saya memutuskan untuk membulatkan nilai rating agar terstandardisasi dalam skala 1 hingga 5.
   Selain itu, kolom *timestamp* yang tersimpan dalam bentuk kode UNIX juga diubah menjadi format *datetime* agar lebih mudah dipahami dan digunakan dalam analisis lanjutan.

3. **Penggabungan Data `movies` dan `ratings`**
   Proses *merging* dilakukan dengan menggabungkan kedua *dataframe* (`movies` dan `ratings`) menjadi satu kesatuan yang dinamai `films`. Data gabungan ini akan menjadi basis utama dalam pengembangan model rekomendasi.

4. **Pembersihan Data pada `films`**
   Setelah penggabungan, saya melakukan beberapa langkah pembersihan tambahan, antara lain:

   * **Menangani Missing Values**
     Data yang hilang atau tidak terbaca seperti *NaN* atau *null* dapat mengganggu kinerja model. Misalnya, pada kolom genre ditemukan nilai **'(no genres listed)'**. Nilai-nilai seperti ini dihapus agar tidak mempengaruhi hasil pelatihan model.

   * **Reduksi Data**
     Karena jumlah data yang sangat besar bisa memperlambat proses pelatihan, saya menyaring film-film yang jumlah rating-nya kurang dari 50. Film dengan jumlah ulasan sedikit dianggap tidak cukup representatif dan penghapusannya akan membantu efisiensi proses pelatihan model tanpa mengurangi kualitas rekomendasi secara signifikan.

   * **Menghapus Duplikasi**
     Duplikasi data dapat menyebabkan bias pada model. Oleh karena itu, saya menghapus entri yang terduplikasi berdasarkan ID dan judul film untuk menjaga konsistensi data.

   * **Perbaikan Data pada Kolom Genre**
     Di kolom genre, terdapat label seperti ***Sci-Fi*** yang merupakan singkatan dari *Science Fiction*. Tanda pemisah (dash) perlu dihapus karena dapat menyebabkan kesalahan saat proses tokenisasi dan vektorisasi TF-IDF, di mana kata ***Sci-Fi*** akan dipecah menjadi dua token berbeda: ***Sci*** dan ***Fi***. Pembersihan ini bertujuan agar genre dikenali sebagai satu kesatuan makna.

## Modeling

Tahapan ini berfokus pada pengembangan dan implementasi dua pendekatan sistem rekomendasi, yaitu **Content-Based Filtering** dan **Collaborative Filtering dengan Neural Network**. Keduanya digunakan untuk menghasilkan *Top-N Movie Recommendations* bagi pengguna berdasarkan data yang telah diproses sebelumnya.

### 1. Content-Based Filtering

**Pendekatan:**

* Menggunakan representasi fitur film berdasarkan kolom **genre** menggunakan teknik **TF-IDF Vectorization**.
* Setelah itu, dihitung **cosine similarity** antar semua film untuk mengukur kemiripan kontennya.
* Berdasarkan skor kemiripan tersebut, sistem akan merekomendasikan film yang paling mirip dengan film yang disukai atau diberi rating tinggi oleh pengguna.

**Langkah-langkah:**

1. Ekstraksi fitur dari kolom `genres` menggunakan `TfidfVectorizer`.
2. Hitung matriks cosine similarity antar semua film.
3. Untuk setiap input film yang disukai pengguna, pilih Top-N film dengan skor similarity tertinggi sebagai rekomendasi.

**Output:**

* Sistem menghasilkan **Top-N rekomendasi film** berdasarkan kemiripan genre dengan film yang pernah ditonton/disukai pengguna.

**Kelebihan:**

* Tidak memerlukan data pengguna lain, cukup berdasarkan konten film.
* Dapat memberikan rekomendasi bahkan untuk pengguna baru yang belum banyak memberi rating (*cold-start for users*).

**Kekurangan:**

* Kurang mampu menangkap selera kompleks pengguna karena hanya berbasis genre.
* Tidak mempertimbangkan popularitas atau selera kolektif dari komunitas pengguna lain.

### 2. Collaborative Filtering dengan Neural Network (Model: RecommenderNet)

**Pendekatan:**

* Menggunakan arsitektur neural network sederhana yang disebut **RecommenderNet**.
* Model memanfaatkan dua **embedding layer** untuk merepresentasikan pengguna dan film dalam bentuk vektor laten berdimensi rendah.
* Dilanjutkan dengan **dot product** antara kedua embedding dan penambahan bias sebagai prediksi rating.
* Model dilatih menggunakan fungsi loss **binary crossentropy** dan optimizer **Adam**.

**Langkah-langkah:**

1. Buat embedding untuk user dan movie ID dari dataset `ratings.csv`.
2. Bangun model RecommenderNet menggunakan TensorFlow/Keras:

   * Dua input: `user_input` dan `movie_input`.
   * Embedding + dot product → sigmoid activation untuk prediksi skor relevansi.
3. Latih model dengan data training dan validasi.
4. Prediksi skor relevansi film untuk pengguna tertentu, kemudian tampilkan Top-N film dengan skor tertinggi sebagai rekomendasi.

**Output:**

* Sistem menghasilkan **Top-N film rekomendasi** bagi pengguna berdasarkan preferensi kolektif pengguna lain yang mirip.

**Kelebihan:**

* Mampu menangkap pola preferensi pengguna yang kompleks.
* Memberikan hasil yang lebih personal karena berdasarkan interaksi nyata antar user dan film.

**Kekurangan:**

* Membutuhkan cukup banyak data historis agar model dapat belajar dengan baik.
* Mengalami masalah cold-start untuk pengguna atau film baru yang belum punya cukup interaksi.

## Evaluation

Evaluasi dilakukan untuk menilai seberapa baik performa dari sistem rekomendasi yang dikembangkan. Setiap pendekatan evaluasi menggunakan metrik yang sesuai dengan jenis metode rekomendasi yang diterapkan.


### **1. Content-Based Filtering**

Untuk pendekatan **Content-Based Filtering**, performa model diukur menggunakan metrik **Precision** yang dikaitkan dengan kemiripan konten menggunakan **Cosine Similarity**. Cosine Similarity merupakan ukuran yang mengukur tingkat kesamaan antar dua vektor berdasarkan sudut di antara mereka.

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/cosine-sim.png?raw=true" alt="cosine-sim">
</p>

**Precision** digunakan untuk mengukur seberapa relevan item yang direkomendasikan oleh sistem terhadap item yang sebenarnya sesuai dengan preferensi pengguna. Precision sangat cocok dalam konteks sistem rekomendasi karena fokus pada kualitas item yang disarankan.

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/precision-formula.png?raw=true" alt="precision">
</p>

**Keterangan:**

* **True Positive (TP):** Item direkomendasikan dan memang relevan.
* **False Positive (FP):** Item direkomendasikan, tetapi tidak relevan.
* **True Negative (TN):** Item tidak direkomendasikan dan memang tidak relevan.
* **False Negative (FN):** Item tidak direkomendasikan padahal relevan.

Dalam sistem rekomendasi, rumus Precision disederhanakan seperti pada ilustrasi berikut:

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/recommender-system-precision.png?raw=true" alt="precision-for-recommendations">
</p>

Untuk pengujian, digunakan film **The Raid 2: Berandal** sebagai referensi untuk mencari rekomendasi film serupa. Hasilnya dapat dilihat berikut:

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/prediksi-film.png?raw=true" alt="output-cb">
</p>

Film tersebut memiliki genre *Adventure, Fantasy, IMAX*. Dari lima film yang direkomendasikan:

* Tiga film memiliki genre yang sama (Adventure, Fantasy, IMAX)
* Dua film lainnya tidak relevan karena bergenre Drama dan Thriller

Dengan demikian, perhitungan Precision sebagai berikut:

* **Precision = (Jumlah rekomendasi relevan) / (Jumlah total rekomendasi)**
* **Precision = 3 / 5 = 0.6 (60%)**

Artinya, sistem berhasil memberikan rekomendasi yang relevan sebanyak 60% dari lima film yang disarankan.

### **2. Collaborative Filtering (Neural Network)**

Untuk pendekatan **Collaborative Filtering** menggunakan model neural network, evaluasi dilakukan dengan metrik **Root Mean Squared Error (RMSE)**. RMSE digunakan untuk mengukur rata-rata kesalahan antara prediksi model dan nilai aktual rating yang diberikan oleh pengguna.

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/collaborative-formula.png?raw=true" width=40% alt="rmse">
</p>

RMSE dihitung dengan mengkuadratkan selisih antara nilai prediksi dan observasi, menjumlahkan seluruh nilai kuadrat tersebut, membaginya dengan jumlah data, lalu mengambil akar kuadrat dari hasilnya.

Selama proses pelatihan model, nilai RMSE digunakan untuk memantau penurunan kesalahan pada tiap epoch. Berikut adalah visualisasi hasil pelatihan model berdasarkan metrik RMSE:

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/hasil-visualisasi-rmse.png?raw=true" alt="metrics">
</p>

Berdasarkan plot tersebut:

* Nilai RMSE pada data **training** mencapai **0.17**
* Nilai RMSE pada data **validasi** adalah **0.26**

Hal ini menunjukkan bahwa model mampu belajar dengan baik dan memberikan hasil prediksi rating yang cukup akurat. Selisih yang tidak terlalu jauh antara training dan validation loss juga menunjukkan bahwa model tidak mengalami overfitting secara signifikan.
