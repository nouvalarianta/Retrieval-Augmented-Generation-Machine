# Machine Assistant

**Machine Assistant** adalah asisten teknis troubleshooting lapangan untuk peralatan industri berbasis *Retrieval-Augmented Generation* (RAG) lokal. Sistem ini mengintegrasikan panduan resmi pabrikan (**Bühler MEAG Manual**) dengan riwayat empiris penanganan teknisi (**Excel Laporan Harian Maintenance**) secara aman dan privat tanpa ketergantungan API pihak ketiga (cloud).

---

## 1. Arsitektur Sistem

Aplikasi menggunakan pendekatan **Dual-Track RAG**:

```text
                                [ Pertanyaan Teknisi ]
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Entity Detector & Guardrail OOS   │
                      │ (Alarm, Mesin, Area, Blok Merek Luar) │
                      └───────────────────┬───────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             [ Track 1: Manual PDF ]             [ Track 2: Excel History ]
           5 Manual Resmi Bühler MEAG              2.300+ Record Perbaikan
                        │                                   │
                        ▼                                   ▼
                BM25 + Glosarium ID-EN             BM25 + Filter FuncLoc
                        │                                   │
                        ▼                                   ▼
               Local LLM (llama-server)            Python Aggregator Lokal
             (Qwen2.5-1.5B-Instruct GGUF)         (100% Deterministik / No LLM)
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                              [ Jawaban Terstruktur ]
                               Bagian 1: Dari Manual
                               Bagian 2: Dari Riwayat
```

1. **Track 1 — Manual Resmi Pabrikan (PDF)**:
   - Sumber: Manual kontroler Bühler MEAG (*Checkweigher, Bagging, Dump Scale, Differential Dosing, Flowbalancer*).
   - Dijawab menggunakan **Local LLM** (`Qwen2.5-1.5B-Instruct` via `llama-server`) untuk menghasilkan penjelasan lugas dan langkah perbaikan sesuai kutipan buku panduan.

2. **Track 2 — Riwayat Perbaikan Lapangan (Excel)**:
   - Sumber: Log penanganan kerusakan harian teknisi.
   - Diproses secara **100% deterministik lokal dengan Python (tanpa LLM)** untuk menjamin privasi data internal operasional pabrik.

---

## 2. Cara Kerja Aplikasi

1. **Ingest & Normalisasi (`ma ingest`)**:
   - Membaca dokumen manual teknis PDF dan sheet Excel.
   - Melakukan ekstraksi struktur, pembersihan header/footer, pemecahan teks menjadi chunk beranotasi, serta normalisasi nama mesin dan status perbaikan.
   - Menyimpan hasil ke dalam `index/chunks.jsonl` dan melacak integritas via `index/manifest.json`.
2. **Retrieval & Guardrail (`ma ask`)**:
   - **Guardrail Out-of-Scope**: Mendeteksi dan menolak pertanyaan di luar sistem (seperti merek Danfoss, Siemens, Wonderware, data privasi karyawan/NIK, atau modifikasi ilegal) secara instan.
   - **Entity Detection**: Mengenali kode alarm (`A012`, `W003`, `E d`), nama mesin (`Automatic Weigher`, `Dump Scale`), area (`CP`, `Packing 25 Kg`), dan lokasi fungsional (*FuncLoc*).
   - **BM25 Teknis & Glosarium Dwibahasa**: Memetakan keluhan teknisi dalam Bahasa Indonesia (*timbangan tidak stabil*, *baterai lemah*) ke padanan istilah manual bahasa Inggris (*standstill*, *battery*).
3. **Penyusunan Jawaban**:
   - Menghasilkan jawaban dua bagian: ringkasan tindakan resmi dari buku manual serta statistik empiris dari riwayat penanganan nyata di lapangan.

---

## 3. Kebutuhan Sistem & Dependensi

### Dependensi Software
- **Python**: Versi `3.10` atau lebih baru.
- **Pustaka Python**:
  - `PyMuPDF` (ekstraksi teks PDF manual)
  - `openpyxl` (pembaca berkas laporan Excel)
  - `pydantic` (validasi skema data dan konfigurasi)
  - `PyYAML` (pengelolaan konfigurasi sistem)
- **Inference Server**: `llama.cpp` (`llama-server`) untuk menjalankan model bahasa secara lokal.
- **Model LLM**: `Qwen2.5-1.5B-Instruct` (format GGUF, misal `qwen2.5-1.5b-instruct-q4_k_m.gguf`).

---

## 4. Panduan Instalasi

### Langkah 1: Siapkan Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Langkah 2: Install Dependensi & Paket CLI
```bash
pip install -r requirements/base.txt
pip install -e .
```

### Langkah 3: Siapkan Dokumen Sumber
Pastikan dokumen teknis sudah berada di dalam folder `docs/`:
- `MEAG_BAG.pdf`
- `MEAG_CHECK.pdf`
- `MEAG_DUMP_82_pages.pdf`
- `MEAG-DIFF.pdf`
- `MEAG-FBAL.pdf`
- `Laporan Harian 1.xlsx`

### Langkah 4: Jalankan Local LLM Server
Jalankan server inferensi lokal menggunakan model Qwen2.5 di port `8080`:
```bash
llama-server \
  -m /path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096
```

---

## 5. Panduan Penggunaan CLI

### 1. Membangun Indeks Dokumen
Jalankan proses ekstraksi dan pembuatan indeks (hanya perlu dijalankan sekali di awal atau saat berkas manual/excel diperbarui):
```bash
ma ingest --rebuild
```

### 2. Mengajukan Pertanyaan Troubleshooting
Ajukan pertanyaan kendala mesin langsung melalui terminal:
```bash
ma ask "Apa penyebab munculnya alarm A004 WATCHDOG pada kontroler MEAG?"
```

### 3. Opsi Argumen CLI
- `--top-k <N>`: Mengatur jumlah kutipan maksimum per sumber (default: `5`).
  ```bash
  ma ask "Bagaimana cara kalibrasi zero pada timbangan bagging?" --top-k 3
  ```
- `--no-llm`: Menampilkan kutipan asli buku manual secara langsung tanpa melalui pemrosesan bahasa oleh LLM.
  ```bash
  ma ask "Apa fungsi dari parameter SYS.REM?" --no-llm
  ```

---

## 6. Struktur Direktori Proyek

```text
machine-assistant/
├── config/
│   ├── profiles/                    # Profil mapping kolom Excel & aturan PDF
│   ├── glossary_id_en.example.yaml  # Glosarium istilah keluhan ID ke istilah manual EN
│   ├── settings.example.yaml        # Konfigurasi aplikasi & endpoint LLM lokal
│   └── sources.yaml                 # Registrasi dokumen sumber (PDF & Excel)
├── docs/                            # Berkas sumber (PDF manual & Excel laporan)
├── index/
│   ├── chunks.jsonl                 # Database teks hasil chunking
│   └── manifest.json                # Hash pelacakan versi incremental ingest
├── scripts/
│   ├── pre-commit                   # Hook proteksi kebocoran data
│   ├── run_eval_30.py               # Runner evaluasi pertanyaan uji
│   └── setup_macos.sh               # Skrip instalasi otomatis macOS
├── src/
│   └── machine_assistant/
│       ├── answer/
│       │   ├── aggregator.py        # Agregasi data Excel deterministik (No LLM)
│       │   └── local_llm.py         # Penghubung prompt & inferensi llama-server
│       ├── ingest/
│       │   ├── base_parser.py       # Kelas dasar dan registry parser
│       │   ├── excel_parser.py      # Parser laporan Excel dengan filter privasi
│       │   ├── pdf_parser.py        # Parser manual teknis PDF Bühler
│       │   └── pipeline.py          # Orkestrasi alur ingest dan pembuatan manifest
│       ├── retrieval/
│       │   ├── engine.py            # Mesin pencari BM25, glosarium, & guardrail OOS
│       │   └── entity_detector.py   # Ekstraktor kode alarm, mesin, & lokasi fungsional
│       ├── cli.py                   # Titik masuk perintah `ma ingest` dan `ma ask`
│       └── config.py                # Konfigurasi Pydantic & path direktori
├── pyproject.toml                   # Konfigurasi instalasi paket Python
└── requirements/
    ├── base.txt                     # Dependensi esensial produksi
    └── dev.txt                      # Dependensi pengembangan tambahan
```
