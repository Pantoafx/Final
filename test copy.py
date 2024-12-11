import pandas as pd

# Langkah 1: Membaca file CSV
csv_file = 'DataAlaska.csv'  # Ganti dengan nama file CSV Anda
df = pd.read_csv(csv_file)

# Tampilkan nama kolom
print("Nama kolom dalam DataFrame:")
print(df.columns)
