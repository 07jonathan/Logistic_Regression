Dalam kode di atas:
  - Fungsi sigmoid digunakan untuk menghitung nilai sigmoid dari z.
  - Fungsi train_logistic_regression melakukan training model menggunakan gradient descent untuk mengoptimalkan bobot (w) dan bias (b).
  - Fungsi predict_logistic_regression digunakan untuk melakukan prediksi berdasarkan model yang sudah dilatih.
  - Dataset contoh (tinggi_badan dan jenis_kelamin) disiapkan dan diteruskan ke fungsi training dan prediksi.
  - Akurasi prediksi sederhana dihitung dengan membandingkan prediksi dengan nilai sebenarnya (y).
  - split_train_test adalah fungsi untuk membagi dataset menjadi data latih (train) dan data uji (test). Fungsi ini menggunakan permutasi acak dari indeks data untuk memastikan pembagian yang acak.
  - Setelah membagi dataset, kita melatih model regresi logistik dengan data latih dan kemudian melakukan prediksi dengan data uji.
  - Akurasi prediksi dihitung dan ditampilkan sebagai persentase.
