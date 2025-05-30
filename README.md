# Laporan Proyek Machine Learning - \[Muhammad Rivaro Farrelino Gozali]

## Project Overview

Rekomendasi film merupakan salah satu aplikasi machine learning yang sangat populer di platform streaming dan e-commerce media hiburan. Dengan jutaan film dan pengguna, memberikan rekomendasi yang relevan dan personal menjadi tantangan utama dalam meningkatkan pengalaman pengguna dan retensi pelanggan.

Dalam proyek ini, digunakan menggunakan dataset MovieLens 25 juta rating film yang mencakup data rating, tag, dan metadata film, untuk membangun sistem rekomendasi film menggunakan dua pendekatan utama: content-based filtering dan collaborative filtering berbasis embedding menggunakan deep learning.

Sistem rekomendasi dapat membantu mengatasi masalah overload informasi (information overload) dengan menyajikan film yang paling sesuai preferensi pengguna berdasarkan histori rating dan kemiripan konten film. Studi sebelumnya menunjukkan bahwa gabungan metode content-based dan collaborative filtering dapat meningkatkan akurasi rekomendasi (Ricci et al., 2011).

Referensi:

* Ricci, F., Rokach, L., Shapira, B., & Kantor, P. B. (2011). *Recommender Systems Handbook*. Springer.

## Business Understanding

### Problem Statements

* Bagaimana membangun sistem rekomendasi film yang dapat menyesuaikan dengan preferensi, minat, atau perilaku pengguna?
* Bagaimana mengevaluasi kinerja dan hasil dari model dalam mengembangkan sistem rekomendasi film yang disesuaikan dengan preferensi, minat, atau perilaku pengguna?

### Goals

* Memahami langkah-langkah pengembangan sistem rekomendasi film yang mampu menyesuaikan dengan preferensi, minat, atau perilaku pengguna.
* Mengetahui hasil serta mengevaluasi performa model dalam membangun sistem rekomendasi yang disesuaikan dengan preferensi, minat, atau perilaku pengguna.

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

Dataset yang digunakan dalam proyek ini adalah **MovieLens 25M Dataset**, yang tersedia di situs [MovieLens](https://grouplens.org/datasets/movielens/#:~:text=MovieLens%2025M%20Dataset). Dataset ini berbentuk arsip ZIP yang memuat 7 file terpisah, yaitu `movies.csv`, `ratings.csv`, `genome-scores.csv`, `links.csv`, `tags.csv`, `genome-tags.csv`, dan `README.txt`. Berdasarkan keterangan dalam file `README.txt`, dataset ini (ml-25m) menyajikan data mengenai film dan rating yang diberikan oleh pengguna. Secara keseluruhan, terdapat 25.000.095 penilaian terhadap film yang diberikan oleh 162.541 pengguna dalam periode dari 9 Januari 1995 hingga 21 November 2019. Dataset ini terakhir diperbarui pada 21 November 2019.

**Informasi Dataset:**

| Kategori | Deskripsi                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------- |
| Judul    | MovieLens 25M Dataset                                                                          |
| Sumber   | [GroupLens](https://grouplens.org/datasets/movielens/#:~:text=MovieLens%2025M%20Dataset)       |
| Pemilik  | [GroupLens Research](https://grouplens.org/about/what-is-grouplens/)                           |
| Tautan   | [https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/) |

**Missing value dalam dataset:**
* **File `tags.csv`** berisi 16 missing value pada kolom `tag`.
* **File `links.csv`** berisi 107 missing value pada kolom `tmdbId`.  

### Deskripsi Variabel dalam Dataset:

* **File `movies.csv`** berisi daftar 62.423 film dengan 3 atribut:

  * `movieId`: ID unik film
  * `title`: Judul film
  * `genres`: Genre film

* **File `ratings.csv`** memuat 25.000.095 data rating dengan 4 atribut:

  * `userId`: ID unik pengguna
  * `movieId`: ID unik film
  * `rating`: Nilai rating film dalam skala 0,5–5 bintang (dengan kenaikan 0,5)
  * `timestamp`: Waktu pemberian rating dalam format timestamp

* **File `tags.csv`** berisi 1.093.360 data tag yang diberikan pengguna pada film dengan 4 atribut:

  * `userId`: ID unik pengguna
  * `movieId`: ID unik film
  * `tag`: Tag atau kata kunci yang diberikan
  * `timestamp`: Waktu pemberian tag dalam format timestamp

* **File `links.csv`** menyajikan 62.423 tautan ke halaman film dengan 3 atribut:

  * `movieId`: ID unik film di MovieLens
  * `imdbId`: ID film di IMDb
  * `tmdbId`: ID film di TMDB

* **File `genome-tags.csv`** memuat 1.128 deskripsi tag yang diacu oleh `tagId` dengan 2 atribut:

  * `tagId`: ID tag unik
  * `tag`: Deskripsi tag

* **File `genome-scores.csv`** berisi 15.584.448 data relevansi tag untuk film dengan 3 atribut:

  * `movieId`: ID film
  * `tagId`: ID tag unik
  * `relevance`: Skor relevansi tag terhadap film

### Exploratory Data Analysis (EDA)

Proses EDA dilakukan untuk memahami struktur dan karakteristik dataset secara menyeluruh guna memperoleh insight dan pengetahuan yang berguna dalam pengembangan model.

**Univariate Analysis**

Dari enam file `.csv` yang tersedia dalam dataset, data dikelompokkan ke dalam tiga kategori utama berdasarkan ID-nya, yaitu data film, data pengguna (users), dan data skor relevansi film. Selanjutnya, dilakukan analisis untuk melihat jumlah data pada masing-masing kategori.

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

Berdasarkan ketiga visualisasi tersebut, terdapat 1.128 tag unik dalam data genome tag, 62.423 film unik, dan 162.541 pengguna unik. Diilai bahwa jumlah data pada skor relevansi berdasarkan `tagId` relatif kecil, hanya sekitar 1.000 data. Jumlah ini jauh lebih sedikit jika dibandingkan dengan data pengguna dan film yang masing-masing berjumlah sekitar 162 ribu dan 62 ribu. Oleh karena itu, diputuskan untuk fokus mengolah data pengguna dan film saja dalam pengembangan sistem rekomendasi.


## Data Preparation

Berikut adalah teknik-teknik yang digunakan dalam tahap Data Preparation:

1. **Penggabungan dan Pembersihan Data `genome_tags` dan `genome_scores`**
   Data `tagId` unik dari kedua file `genome_tags` dan `genome_scores` digabungkan menjadi satu array. Selanjutnya, duplikasi dihapus untuk memastikan hanya tag unik yang dihitung, lalu data disortir agar lebih terstruktur. Langkah ini menghasilkan jumlah total `tagId` unik dari kedua file setelah penggabungan dan pembersihan.

2. **Penggabungan dan Pembersihan Data `movieId` dari `movies.csv`, `tags.csv`, `ratings.csv`, dan `links.csv`**
   Seluruh `movieId` unik dari file `movies`, `tags`, `ratings`, dan `links` digabungkan untuk memastikan cakupan data film yang lengkap. Setelah penggabungan, duplikasi dihapus dan data disortir untuk mendapatkan struktur data yang rapi. Hasil ini menunjukkan jumlah total film unik yang akan digunakan dalam analisis.

3. **Penggabungan dan Pembersihan Data `userId` dari `ratings.csv` dan `tags.csv`**
   Data `userId` unik yang terdapat pada file `ratings` dan `tags` digabungkan menjadi satu array. Proses penggabungan ini dilanjutkan dengan penghapusan duplikasi dan penyortiran untuk memperoleh data yang terorganisir. Hasil akhirnya memberikan total jumlah pengguna unik yang terlibat dalam memberikan rating dan tag pada film.

4. **Ekstraksi Tahun Rilis dari `movies.csv`**
   Tahun rilis film diekstrak dari kolom `title` ke dalam kolom baru bernama `year_of_release`. Ekstraksi ini menggunakan pola 4 digit angka dalam tanda kurung yang biasanya menunjukkan tahun rilis film. Langkah ini membantu memisahkan informasi judul film dan tahun rilis agar data lebih terstruktur.

5. **Pembersihan Judul Film di `movies.csv`**
   Judul film dibersihkan dengan menghapus informasi tambahan, termasuk tahun rilis yang berada dalam tanda kurung. Proses ini dilakukan dengan memisahkan string pada tanda kurung pertama dan mengambil bagian sebelum tanda kurung. Selain itu, tipe data kolom `title` diubah menjadi string agar lebih konsisten.

6. **Pembulatan Nilai Rating di `ratings.csv`**
   Nilai rating yang awalnya berupa bilangan desimal (contoh: 3.5, 4.0) dibulatkan ke atas menjadi bilangan bulat (contoh: 4, 5) menggunakan metode `ceil`. Hal ini menyederhanakan data rating menjadi skala integer 1–5 yang lebih mudah diolah dalam pemodelan.

7. **Konversi Kolom Timestamp di `ratings.csv`**
   Kolom `timestamp` yang semula berupa integer (dalam format Unix time) diubah ke format `datetime`. Perubahan format ini mempermudah analisis berbasis waktu, misalnya tren rating dari waktu ke waktu.

8. **Penggabungan Data `movies` dan `ratings`**
   Data `movies` dan `ratings` digabungkan menggunakan kolom `movieId` sebagai kunci utama. Penggabungan ini menggunakan metode `left join` agar setiap data film tetap tercakup, meskipun tidak memiliki rating. Hasil penggabungan kemudian diperiksa untuk mendeteksi nilai kosong (missing values).

9. **Pembersihan Nilai Kosong (Missing Values)**
   Setelah penggabungan data, dilakukan penghapusan baris yang memiliki nilai kosong untuk memastikan data bersih dan siap digunakan dalam proses analisis atau pemodelan. Langkah ini penting untuk menghindari error dan bias saat membangun model.

10. **Penghapusan Film dengan Genre `(no genres listed)`**
  Pada beberapa entri di kolom `genres`, terdapat film dengan genre yang ditandai sebagai `(no genres listed)`. Film-film tersebut dihapus dari dataset karena tidak memiliki informasi genre yang valid, sehingga tidak akan berguna untuk analisis genre ataupun rekomendasi berbasis genre.

11. **Penyaringan (Filtering) Film Berdasarkan Jumlah Kemunculan `movieId` Minimal 50 Kali**
  Untuk memastikan kualitas data, hanya film-film yang muncul minimal 50 kali pada dataset yang disimpan. Hal ini dilakukan agar model yang akan dibangun memiliki data yang cukup representatif dan tidak bias terhadap film-film dengan jumlah rating yang sangat sedikit.

12. **Penghapusan Data Duplikat Berdasarkan `movieId` dan `title`**
  Setelah proses penggabungan dan penyaringan awal, masih mungkin terdapat entri yang duplikat, baik berdasarkan `movieId` maupun `title`. Oleh karena itu, dilakukan penghapusan data duplikat agar setiap film hanya muncul sekali dan terhindar dari penghitungan ganda.

13. **Penggantian String `[nS]ci-Fi` Menjadi `Scifi` pada Kolom `genres`**
  Untuk memastikan konsistensi dalam penulisan genre, string yang sesuai dengan pola `[nS]ci-Fi` diubah menjadi `Scifi`. Perubahan ini bertujuan untuk menyeragamkan data genre sehingga memudahkan analisis selanjutnya.

### **Feature Engineering**

Langkah awal dilakukan dengan membuat salinan data hasil penggabungan `movies` dan `ratings` ke dalam variabel baru `preparation`. Data ini kemudian diurutkan berdasarkan `movieId`.

* Kolom `movieId`, `title`, dan `genres` pada data preparation diubah menjadi list terpisah (`film_id`, `film_name`, dan `film_genre`) agar dapat digunakan untuk pembuatan dataframe baru bernama `df_film`.
* Pembuatan dataframe `df_film` ini menyusun kembali data `film_id`, `film_name`, dan `genre` ke dalam format yang rapi, yang mempermudah proses berikutnya, terutama pada content-based filtering.

Selain itu, dilakukan encoding (penyandian) terhadap `userId` dan `movieId`:

* Nilai unik `userId` diubah menjadi list, lalu dibuat dictionary penyandian `user_to_user_encoded` dan `user_encoded_to_user` untuk memetakan `userId` ke indeks angka dan sebaliknya.
* Hal serupa juga dilakukan untuk `movieId`, membentuk `films_to_films_encoded` dan `films_encoded_to_films`.
* Dataframe kemudian diperbarui dengan menambahkan kolom `user` dan `films` yang berisi hasil penyandian.
* Jumlah user dan film unik dihitung, dan kolom `rating` diubah ke format `float32`. Nilai rating minimum dan maksimum dicatat untuk digunakan dalam normalisasi.


### **Vektorisasi TF-IDF**

Pada tahap ini dilakukan proses ekstraksi fitur berbasis konten (content-based filtering) dengan menggunakan **TF-IDF** pada kolom `genres`.

* Objek `TfidfVectorizer` dibuat untuk memproses teks genre. Data genre dari dataframe `df_film` dilatih (fit) dan diubah menjadi representasi numerik matriks **TF-IDF**.
* Matriks TF-IDF disimpan dalam bentuk sparse matrix (untuk menghemat memori) dan juga dalam bentuk dense matrix untuk keperluan analisis.
* Matriks ini memiliki dimensi baris sebanyak jumlah film dan kolom sebanyak jumlah fitur kata dari genre.
* Karena keterbatasan memori, subset data diambil dengan memilih 10.000 indeks acak untuk menghitung **cosine similarity** antar film.
* Hasil similarity ini disimpan dalam dataframe `cosine_sim_df` dengan nama film sebagai indeks dan kolom. Sebuah fungsi `film_recommendations()` juga dibuat untuk mengambil rekomendasi film berdasarkan kemiripan genre.


### **Splitting Data untuk Pelatihan Model Collaborative Filtering**

Untuk membangun model rekomendasi berbasis collaborative filtering menggunakan neural network, data disiapkan sebagai berikut:

* Data `df` yang telah dilengkapi penyandian dan normalisasi rating diacak dengan sampling acak menggunakan `sample(frac=1)`.
* Fitur input `x` diambil dari kolom `user` dan `films`, sementara target `y` diperoleh dari kolom `rating` yang dinormalisasi ke rentang 0-1 menggunakan rumus normalisasi min-max.
* Data dibagi menjadi data **train** dan **validation** menggunakan `train_test_split` dengan proporsi 80% data untuk training dan 20% untuk validasi. Pembagian ini memastikan data latih dan uji terdistribusi acak dan seimbang.


## Modeling

Tahapan ini berfokus pada pengembangan dan implementasi dua pendekatan sistem rekomendasi, yaitu **Content-Based Filtering** dan **Collaborative Filtering dengan Neural Network**. Keduanya digunakan untuk menghasilkan *Top-N Movie Recommendations* bagi pengguna berdasarkan data yang telah diproses sebelumnya.

### 1. Content-Based Filtering

#### Pendekatan:

* Menggunakan representasi fitur film berdasarkan kolom **genre** menggunakan teknik **TF-IDF Vectorization**.
* Menghitung **cosine similarity** antar semua film untuk mengukur tingkat kemiripan kontennya.
* Berdasarkan skor kemiripan, sistem merekomendasikan film yang mirip dengan yang disukai/diberi rating tinggi oleh pengguna.

#### Penjelasan Cosine Similarity:

* Cosine similarity mengukur kemiripan antar dua vektor (dalam hal ini vektor TF-IDF dari genre film) dengan menghitung **cosine dari sudut antar vektor** tersebut.
* Formula:

  <p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/cosine-sim.png?raw=true" alt="cosine-sim">
</p>

  Di mana:

  * A dan B adalah vektor fitur dua film,
  * A dot B adalah dot product kedua vektor,
  * |A| dan |B| adalah norma (panjang) dari vektor.
* Nilai cosine similarity berkisar dari 0 (tidak mirip) hingga 1 (identik).

#### Langkah-langkah:

1. Ekstraksi fitur dari kolom `genres` menggunakan `TfidfVectorizer`.
2. Hitung matriks cosine similarity antar semua film.
3. Untuk setiap input film yang disukai pengguna, pilih **Top-N** film dengan skor similarity tertinggi sebagai rekomendasi.

#### Hasil Top-N Recommendation:

Contoh rekomendasi untuk film **"The Raid 2: Berandal"**:

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/prediksi-film.png?raw=true" alt="output-cb">
</p>

#### Kelebihan:

* Tidak memerlukan data pengguna lain, cukup berdasarkan konten film.
* Dapat memberikan rekomendasi bahkan untuk pengguna baru (cold-start for users).

#### Kekurangan:

* Kurang mampu menangkap selera kompleks pengguna karena hanya berbasis genre.
* Tidak mempertimbangkan popularitas atau selera kolektif dari pengguna lain.


### 2. Collaborative Filtering dengan Neural Network (Model: RecommenderNet)

#### Pendekatan:

* Menggunakan arsitektur neural network **RecommenderNet** yang memanfaatkan dua **embedding layer** untuk merepresentasikan pengguna dan film dalam vektor berdimensi rendah.
* Menghitung **dot product** antar vektor embedding ditambah bias sebagai prediksi skor rating.
* Model dilatih menggunakan **binary crossentropy** dan optimizer **Adam**.

#### Langkah-langkah:

1. Encoding user dan film ID menjadi angka urut.
2. Membangun model neural network dengan dua input (`user_input` dan `film_input`), embedding layer, dot product, dan sigmoid activation.
3. Melatih model menggunakan data training dan validasi.
4. Prediksi skor rating untuk film yang belum ditonton user, dan menampilkan **Top-N** film dengan skor tertinggi.

#### Hasil Evaluasi:

* **Root Mean Squared Error (RMSE)**:

  * Data training: \~0.03
  * Data validasi: \~0.25
    Hal ini menunjukkan model mampu belajar dengan baik, meskipun validasi error masih cukup besar.

#### Hasil Top-N Recommendation:

Contoh rekomendasi untuk user ID **2601**:

<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/prediksi-film-collaborative.png?raw=true" alt="output-col">
</p>

#### Kelebihan:

* Mampu menangkap pola preferensi pengguna yang kompleks.
* Memberikan hasil yang lebih personal karena berdasarkan interaksi pengguna.

#### Kekurangan:

* Membutuhkan data historis yang memadai agar model dapat belajar dengan baik.
* Mengalami masalah cold-start untuk pengguna atau film baru.

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

Film tersebut memiliki genre *Action, Crime, Thriller*. Dari lima film yang direkomendasikan:

* Lima film memiliki genre yang sama (Action, Crime, Thriller)

Dengan demikian, perhitungan Precision sebagai berikut:

* **Precision = (Jumlah rekomendasi relevan) / (Jumlah total rekomendasi)**
* **Precision = 5 / 5 = 1.0 (100%)**

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

* Nilai RMSE pada data **training** mencapai **0.0321**
* Nilai RMSE pada data **validasi** adalah **0.2550**

Hal ini menunjukkan bahwa model mampu belajar dengan baik dan memberikan hasil prediksi rating yang cukup akurat. Selisih yang tidak terlalu jauh antara training dan validation loss juga menunjukkan bahwa model tidak mengalami overfitting secara signifikan.

Untuk pengujian acak didapatkan satu dari **ID user 2601:**
<p align='center'>
    <img src ="https://github.com/RivaroFarrelino/movie-reccommendation-system/blob/main/images/prediksi-film-collaborative.png?raw=true" alt="output-col">
</p>