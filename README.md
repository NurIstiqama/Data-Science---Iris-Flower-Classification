Analisis Data Eksplorasi Portofolio Proyek
Analisis Data Eksplorasi (EDA) – Dataset Iris

Exploratory Data Analysis (EDA) adalah proses analisis eksplorasi untuk memahami pola dalam data sebelum membangun model Machine Learning. Saya menggunakan dataset Iris dari Scikit-Learn, yang dapat diakses melalui tautan berikut:  
🔗 [Dataset Iris - Scikit-Learn](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)  

Dataset ini berisi 150 sampel dari tiga spesies bunga Iris (Setosa, Versicolor, dan Virginica), dengan empat fitur utama:  
- Sepal Length (cm)
- Sepal Width (cm) 
- Petal Length (cm)
- Petal Width (cm)

Analisis ini saya kerjakan sebagai bagian dari exercise di Dibimbing kelas Data Science untuk memahami bagaimana teknik eksplorasi data dan klasifikasi bekerja dalam dataset sederhana.  

📌 Tujuan Analisis
1️⃣ Memahami distribusi data setiap fitur dalam dataset Iris.  
2️⃣ Mengetahui hubungan antara fitur dan spesies bunga.  
3️⃣ Mengidentifikasi pola atau korelasi antar variabel.  
4️⃣ Membangun model klasifikasi menggunakan K-Nearest Neighbors (KNN) untuk mengelompokkan spesies bunga berdasarkan fitur yang diberikan.  


📊 Wawasan dari Analisis 
🔹 Distribusi Data
- Petal length dan petal width memiliki distribusi yang paling berbeda antar spesies, yang menjadikannya fitur penting untuk klasifikasi.  
- Setosa memiliki karakteristik yang unik dibandingkan dengan Versicolor dan Virginica, sehingga lebih mudah dikenali.  

🔹 Korelasi Antar Fitur 
- Petal length vs Petal width memiliki korelasi tinggi, yang membantu dalam membedakan spesies bunga.  
- Sepal length vs Sepal width memiliki korelasi lebih lemah dibandingkan fitur lainnya.  

🔹 Hasil Model KNN
- Saya menggunakan algoritma K-Nearest Neighbors (KNN) untuk klasifikasi dengan parameter terbaik (nilai k=3).  
- Model berhasil mengklasifikasikan spesies bunga dengan akurasi tinggi (~97%).  
- Confusion matrix menunjukkan bahwa model dapat mengenali Setosa dengan sempurna, sedangkan Versicolor dan Virginica memiliki sedikit kesalahan klasifikasi.  

💡 Kesimpulan dan Saran 
✅ Petal length dan petal width adalah fitur paling signifikan untuk membedakan spesies.  
✅ Model KNN dengan k=3 memberikan hasil yang baik untuk klasifikasi dataset Iris.  
✅ Normalisasi data bisa meningkatkan performa model pada dataset yang lebih kompleks.  
✅ Teknik lain seperti Logistic Regression atau Decision Tree dapat dicoba untuk melihat perbandingan hasil klasifikasi.  

📩 Diskusi dan Feedback
Analisis ini saya kerjakan untuk memenuhi exercise dari Dibimbing kelas Data Science. Jika ada saran atau ingin berdiskusi lebih lanjut, silakan hubungi saya melalui LinkedIn atau Email.  

#Python #EDA #IrisDataset #KNN #MachineLearning #DataScience
