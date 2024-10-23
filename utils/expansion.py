import re


CONTEXT_ABBREVIATIONS = {
    "FTK": "Fakultas Teknik dan Kejuruan (FTK)",
    "FOK": "Fakultas Olahraga dan Kesehatan (FOK)",
    "FMIPA": "Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA)",
    "FK": "Fakultas Kedokteran (FK)",
    "FIP": "Fakultas Ilmu Pendidikan (FIP)",
    "FHIS": "Hukum dan Ilmu Sosial (FHIS)",
    "FBS": "Fakultas Bahasa dan Seni (FBS)",
    "FE": "Fakultas Ekonomi (FE)",
    "WR": "Wakil Rektor (WR)",
    "WD": "Wakil Dekan (WD)",
    "Kejur": "Ketua Jurusan (Kejur)",
    "Kaprodi": "Kepala Program Studi (Kaprodi)",
    "Koorprodi": "Koordinator Program Studi (Koorprodi)",
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
    "Sempro": "Seminar Proposal (Sempro)",
    "Semhas": "Seminar Hasil (Semhas)",
}


def query_expansion(question: str, abbreviations: dict) -> str:
    def replace_query(match):
        return abbreviations.get(match.group(0).upper(), match.group(0))
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviations.keys()) + r')\b', re.IGNORECASE)
    expanded_question = pattern.sub(replace_query, question)
    return expanded_question