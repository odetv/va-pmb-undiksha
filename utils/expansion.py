import re


CONTEXT_ABBREVIATIONS = {
    "UNDIKSHA": "Universitas Pendidikan Ganesha",
    "FTK": "Fakultas Teknik dan Kejuruan (FTK) Undiksha",
    "FOK": "Fakultas Olahraga dan Kesehatan (FOK) Undiksha",
    "FMIPA": "Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA) Undiksha",
    "FK": "Fakultas Kedokteran (FK) Undiksha",
    "FIP": "Fakultas Ilmu Pendidikan (FIP) Undiksha",
    "FHIS": "Hukum dan Ilmu Sosial (FHIS) Undiksha",
    "FBS": "Fakultas Bahasa dan Seni (FBS) Undiksha",
    "FE": "Fakultas Ekonomi (FE) Undiksha",
    "WR": "Wakil Rektor (WR)",
    "WD": "Wakil Dekan (WD)",
    "KEJUR": "Ketua Jurusan (Kejur)",
    "KAPRODI": "Kepala Program Studi (Kaprodi)",
    "KOORPRODI": "Koordinator Program Studi (Koorprodi)",
    "SNBP": "Seleksi Nasional Berbasis Prestasi (SNBP)",
    "SNBT": "Seleksi Nasional Berbasis Tes (SNBT)",
    "SMBJM": "Seleksi Mahasiswa Baru Jalur Mandiri (SMBJM)",
    "UKT": "Uang Kuliah Tunggal (UKT)",
    "KIP K": "Kartu Indonesia Pintar Kuliah Undiksha (KIP-K)",
    "KIP-K": "Kartu Indonesia Pintar Kuliah Undiksha (KIP-K)",
    "PMB": "Penerimaan Mahasiswa Baru (PMB)",
    "KRS": "Kartu Rencana Studi (KRS)",
    "KHS": "Kartu Hasil Studi (KHS)",
    "SKS": "Satuan Kredit Semester (SKS)",
    "IP": "Indeks Prestasi (IP)",
    "IPK": "Indeks Prestasi Kumulatif (IPK)",
    "SIAK": "Sistem Informasi Akademik (SIAK)",
    "SIAK-NG": "Sistem Informasi Akademik New Generation (SIAK-NG)",
    "UKM": "Unit Kegiatan Mahasiswa (UKM)",
    "KKN": "Kuliah Kerja Mahasiswa (KKN)",
    "PKL": "Praktik Kerja Lapangan (PKL)",
    "BEM": "Badan Eksekutif Mahasiswa (BEM)",
    "HMJ": "Himpunan Mahsiswa Jurusan (HMJ)",
    "BAN PT": "Badan Akreditasi Nasional Perguruan Tinggi (BAN-PT)",
    "BAN-PT": "Badan Akreditasi Nasional Perguruan Tinggi (BAN-PT)",
    "TA": "Tugas Akhir (TA)",
    "MABA": "Mahasiswa Baru (MABA)",
    "PKKMB": "Pengenalan Kehidupan Kampus bagi Mahasiswa Baru (PKKMB)",
    "SEMPRO": "Seminar Proposal (Sempro)",
    "SEMHAS": "Seminar Hasil (Semhas)",
    "UPA-TIK": "UPA TIK Undiksha",
    "UPA TIK": "UPA TIK Undiksha",
    "UPT-TIK": "UPA TIK Undiksha",
    "UPTTIK": "UPA TIK Undiksha",
}


def query_expansion(question: str, abbreviations: dict) -> str:
    def replace_query(match):
        return abbreviations.get(match.group(0).upper(), match.group(0))
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b', re.IGNORECASE)
    expanded_question = pattern.sub(replace_query, question)
    return expanded_question