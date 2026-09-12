# Demand Forecasting Bahan Baku - Streamlit

Aplikasi Python Streamlit untuk membaca, membersihkan, menganalisis, dan melakukan demand forecasting pada dataset manufaktur.

## Struktur project

- `app.py` — aplikasi Streamlit
- `Demand_Forecasting .csv` — dataset
- `requirements.txt` — library Python
- `README.md` — panduan deployment

## Menjalankan di komputer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy gratis dengan URL publik

1. Buat repository baru di GitHub.
2. Upload `app.py`, `requirements.txt`, `README.md`, dan dataset CSV.
3. Buka Streamlit Community Cloud.
4. Pilih repository GitHub tersebut.
5. Pilih branch `main`.
6. Isi file utama dengan `app.py`.
7. Deploy.

Setelah deployment selesai, Streamlit akan memberikan URL publik. Siapa pun yang memiliki URL tersebut dapat membuka aplikasi selama aplikasi/repository deployment masih aktif.

## Catatan dataset

Dataset utama harus memiliki kolom:
- Date
- Warehouse
- Product_Code
- Product_Category
- Order_Demand

Nilai `Order_Demand` seperti `(13)` akan otomatis dibersihkan menjadi `13`.
