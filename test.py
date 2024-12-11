import pandas as pd
import sqlite3

# Langkah 1: Membaca file CSV
csv_file = 'DataAlaska.csv'  # Ganti dengan nama file CSV Anda
df = pd.read_csv(csv_file)

# Membersihkan nama kolom
df.columns = df.columns.str.strip()  # Menghapus spasi di depan dan belakang
df.columns = df.columns.str.replace(';', '', regex=False)  # Menghapus tanda titik koma

# Tampilkan nama kolom setelah dibersihkan
print("Nama kolom setelah dibersihkan:")
print(df.columns)

# Langkah 2: Membuat koneksi ke database SQLite
conn = sqlite3.connect('data_alaska.db')  # Nama file database SQLite yang akan dibuat
cursor = conn.cursor()

# Langkah 3: Membuat tabel jika belum ada
cursor.execute('''
CREATE TABLE IF NOT EXISTS ulasan_alaska (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT,
    tipe_barang TEXT,
    varian_barang TEXT,
    ulasan TEXT,
    rating INTEGER,
    tanggal TEXT
)
''')

# Langkah 4: Menyimpan DataFrame ke SQLite
df.to_sql('ulasan_alaska', conn, if_exists='append', index=False)

# Langkah 5: Menutup koneksi
conn.commit()  # Simpan perubahan
conn.close()

print("Data berhasil diimpor ke SQLite!")
