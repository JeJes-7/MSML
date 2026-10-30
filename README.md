# MSML – Prediksi Kematian Gagal Jantung dengan MLflow, Prometheus & Grafana

Proyek akhir *Sistem Machine Learning* (SML) oleh **Jesika Yudiani Putri**. Proyek ini mencakup alur MLOps end-to-end: eksperimen dan preprocessing data, pelatihan model dengan pelacakan MLflow, penyajian model lewat REST API, hingga monitoring dan alerting menggunakan Prometheus dan Grafana.

**Tujuan model:** memprediksi `DEATH_EVENT` (pasien meninggal atau tidak selama masa tindak lanjut) berdasarkan data klinis pasien gagal jantung, memakai algoritma **Random Forest Classifier**.

## Repositori Terkait

| Tahap | Repositori |
|---|---|
| Eksperimen & preprocessing data | https://github.com/JeJes-7/Eksperimen_SML_Jesika |
| Workflow CI | https://github.com/JeJes-7/Workflow-CI |


## 1. Data

- Dataset: **Heart Failure Clinical Records**, dengan target `DEATH_EVENT` (0 = bertahan, 1 = meninggal).
- Fitur klinis: `age`, `anaemia`, `creatinine_phosphokinase`, `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, `smoking`.
- Hasil preprocessing: fitur numerik telah distandarisasi dan umur dikelompokkan menjadi kategori one-hot (`age_Adult`, `age_Senior`, `age_Elderly`), sehingga model menerima **14 fitur**.
- Pembagian data: **216 sampel latih** dan **54 sampel uji**.

## 2. Membangun Model

Dua skrip pelatihan tersedia, keduanya menggunakan `RandomForestClassifier` dan dilacak dengan MLflow:

- `modelling.py` – pelatihan dasar (100 pohon) dengan `mlflow.autolog()`.
- `modelling_tuning.py` – pelatihan dengan *manual logging*: parameter, metrik (accuracy, precision, recall), dan model dicatat sendiri. Konfigurasi default skrip: `n_estimators=200`, `max_depth=10`.

Eksperimen dicatat pada experiment MLflow `Eksperimen_Jesika_Heart_Failure`.

**Hasil evaluasi pada data uji (run tercatat di MLflow):**

| Metrik | Nilai |
|---|---|
| Accuracy | 0,759 |
| Precision | 0,636 |
| Recall | 0,438 |
| F1-score | 0,519 |

Artefak seperti *confusion matrix* dan *feature importance* tersimpan bersama setiap run.

**Menjalankan pelatihan:**

```bash
pip install -r Membangun_model/requirements.txt

cd Membangun_model
python modelling.py --data_path data_train.csv
python modelling_tuning.py --data_path data_train.csv

# Melihat hasil eksperimen
mlflow ui
```

## 3. Serving Model

`Monitoring_dan_Logging/7.Inference.py` menyajikan model lewat API Flask pada `http://127.0.0.1:5050`.

**Endpoint:** `POST /predict`

```json
{
  "data": [ ...nilai fitur... ]
}
```

Contoh respons:

```json
{
  "prediction": [0],
  "latency": 0.0123
}
```

API memvalidasi payload (JSON harus memuat `data` berupa list) dan mengembalikan kode `400` untuk input tidak valid serta `500` untuk kesalahan internal. Bukti pengujian tersedia pada `1.Bukti_serving.jpg` dan `1.Bukti_serving_postman.jpg`.

## 4. Monitoring & Logging

Stack monitoring terdiri dari **Prometheus** (pengumpul metrik) dan **Grafana** (visualisasi dan alerting).

- `3.prometheus_exporter.py` menjalankan exporter pada port `8000` dengan metrik:
  - `request_total` – jumlah request inferensi
  - `request_latency_seconds` – histogram latensi
  - `prediction_error_total` – jumlah error prediksi
  - `process_cpu_usage` – penggunaan CPU
  - `process_memory_usage` – penggunaan memori
- `2.prometheus.yml` – Prometheus melakukan scrape ke `localhost:8000` setiap 5 detik (job `ml_model_metrics`).
- Dashboard Grafana memantau status model, request rate, latensi, resource (CPU/memori), dan error.
- Alerting Grafana dikonfigurasi dengan rule berbasis threshold dan telah terbukti menghasilkan notifikasi.


**Menjalankan monitoring:**

```bash
# 1. Jalankan exporter
python Monitoring_dan_Logging/3.prometheus_exporter.py

# 2. Jalankan Prometheus dengan konfigurasi proyek
prometheus --config.file=Monitoring_dan_Logging/2.prometheus.yml

# 3. Jalankan Grafana (http://localhost:3000), tambahkan Prometheus sebagai data source
```

## 5. Bukti Pendukung

| Komponen | Lokasi |
|---|---|
| Dashboard & artefak MLflow | `Membangun_model/Screenshot_*.jpg` |
| Serving model | `Monitoring_dan_Logging/1.Bukti_serving*.jpg` |
| Monitoring Prometheus | `Monitoring_dan_Logging/4.bukti monitoring Prometheus/` |
| Monitoring Grafana | `Monitoring_dan_Logging/5.bukti monitoring Grafana/` |
| Alerting Grafana | `Monitoring_dan_Logging/6.bukti alerting Grafana/` |

## Teknologi

Python · scikit-learn · pandas · MLflow · Flask · Prometheus · Grafana

## Penulis

**Jesika Yudiani Putri**
