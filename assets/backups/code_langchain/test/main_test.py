import pytest
import warnings
import sys
import os
import logging
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import query_rag
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Setup logging
log_dir = "test/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, datetime.now().strftime("test_log_%Y%m%d_%H%M%S.log"))
# Pastikan untuk menghapus handlers yang sudah ada untuk menghindari duplikasi
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Mengatur format tanggal tanpa milidetik
    handlers=[
        logging.FileHandler(log_filename),
    ]
)
logger = logging.getLogger()
logger.info("Logging initialized")


# Fungsi clean teks menghapus spasi di awal dan akhir serta mengubah semua teks menjadi case huruf kecil
def clean_text(text):
    return text.strip().lower()


# Fungsi mendapatkan respons dari API chatbot dan memvalidasi apakah sesuai dengan respons yang diharapkan
def query_and_validate(question: str, expected_response: str):
    response = query_rag(question)
    actual_response = response['answer'].strip()

    # Membersihkan teks untuk memastikan pencocokan lebih tepat
    cleaned_expected_response = clean_text(expected_response)
    cleaned_actual_response = clean_text(actual_response)

    # Logging to file
    logger.info(f"Question: {question}")
    logger.info(f"Actual Response: {cleaned_actual_response}")
    logger.info(f"Expected Response: {cleaned_expected_response}")

    # Menampilkan respons dan ekspektasi yang telah dibersihkan
    print(f"\nActual Response (cleaned): {cleaned_actual_response}")
    print(f"Expected Response (cleaned): {cleaned_expected_response}")

    # Validasi apakah cleaned_expected_response ada di dalam cleaned_actual_response
    if cleaned_expected_response in cleaned_actual_response:
        print("\033[92m" + "Expected response found within the actual response. Test passed." + "\033[0m")
        logger.info("Test result: Passed")
        return True
    else:
        print("\033[91m" + "Expected response not found in actual response. Test failed." + "\033[0m")
        logger.error("Test result: Failed")
        return False


# Daftar pertanyaan dan respons yang diharapkan
test_cases = [
    ("Apa itu undiksha?", "Undiksha merupakan perguruan tinggi yang berlokasi di Jalan Udayana No.11 Singaraja - Bali 81116 Buleleng Bali. dikembangkan berdasarkan Pancasila dan Undang-Undang Dasar 1945 yang menjunjung nilai-nilai kemanusiaan, menghasilkan tenaga kependidikan, dan tenaga non-kependidikan yang bertakwa kepada Tuhan Yang Maha Esa, memiliki kemampuan akademis-profesional yang tinggi, mengembangkan ilmu pengetahuan, teknologi, dan seni."),
    ("Siapa rektor undiksha?", "Prof. Dr. I Wayan Lasmawan, M.Pd."),
    ("Apakah ada video profil undiksha?", "https://www.youtube.com/watch?v=rkQBOhpXhwc"),
    ("Apa motto undiksha?", "dharmaning sajjana umerdhyaken widyaguna."),
    ("Apa arti motto undiksha?", "kewajiban orang bijaksana adalah mengembangkan ilmu pengetahuan dan pekerti."),
    ("Bagaimana akreditasi undiksha?", "Universitas Pendidikan Ganesha (Undiksha) merupakan satu-satunya perguruan tinggi negeri di Bali Utara yang telah TERAKREDITASI A oleh Badan Akreditasi Nasional Perguruan Tinggi dan terakreditasi Internasional (AQAS) pada berbagai bidang studi."),
    ("Apakah undiksha sudah terakreditasi AQAS?", "Ya, telah mendapatkan akreditasi internasional dari AQAS."),
    ("Fakta undiksha seperti apa?", "Perkembangan yang terjadi sejak diubah statusnya menjadi Universitas Pendidikan Ganesha pada tahun 2006, Undiksha kini telah bersaing dengan berbagai perguruan tinggi Nasional. Capaian tidak hanya dalam bidang bidang akademik, namun dalam non akademik."),
    ("Apa visi undiksha?", "Menjadi Universitas Unggul Berlandaskan Falsafah Tri Hita Karana di Asia Pada Tahun 2045"),
    ("Apa misi undiksha?", "1. Menyelenggarakan pendidikan dan pengajaran yang bermartabat untuk menghasilkan sumber daya manusia yang kompetitif, kolaboratif, dan berkarakter. 2. Menyelenggarakan penelitian yang kompetitif, kolaboratif, dan inovatif untuk pengembangan dan penerapan ilmu pengetahuan dan teknologi. 3. Menyelenggarakan pengabdian kepada masyarakat yang kompetitif, kolaboratif, akomodatif, dan inovatif."),
    ("Apa tujuan undiksha?", """
        1. Menghasilkan lulusan yang mampu bersaing dengan lulusan universitas lain dalam mengisi pasar kerja.
        2. Menghasilkan lulusan yang mampu bekerja secara bersama-sama atau dalam bentuk tim di tempat kerja.
        3. Menghasilkan lulusan yang menjunjung tinggi nilai-nilai ketuhanan, kemanusiaan, dan kelestarian alam dalam menjalankan tugas.
        4. Menghasilkan karya penelitian yang mampu bersaing dengan karya-karya penelitian yang dihasilkan oleh sivitas akademika universitas lain.
        5. Menghasilkan karya penelitian yang dilakukan secara bersama-sama dengan sivitas akademika lain dan/atau masyarakat, baik yang berasal dari dalam maupun luar Undiksha.
        6. Menghasilkan karya penelitian yang memiliki kebaruan.
        7. Menghasilkan karya pengabdian kepada masyarakat yang mampu bersaing dengan karya pengabdian kepada masyarakat yang dilakukan oleh universitas lain.
        8. Menghasilkan karya pengabdian kepada masyarakat yang dilakukan secara bersama-sama antar sivitas akademika dan/atau pegawai, baik yang berasal dari dalam maupun luar Undiksha.
        9. Menghasilkan karya pengabdian kepada masyarakat yang dilakukan atas permintaan masyarakat.
        10. Menghasilkan karya pengabdian masyarakat yang memiliki kebaruan.
        11. Menghasilkan kerjasama nasional, regional, dan internasional yang saling menguntungkan.
    """),
    ("Bagaimana sasaran undiksha?", """
        1. Meningkatnya kualitas sistem tata kelola kelembagaan secara terpadu, transparan, akuntabel, adil, dan bertanggung jawab berlandaskan falsafah Tri Hita Karana.
        2. Diperolehnya masukan (input) yang berkualitas dan dihasilkannya lulusan (output) yang memiliki daya saing yang tinggi dalam memasuki dunia kerja, mampu bekerja sama, dan memiliki nilai-nilai ketuhanan, kemanusiaan, dan mampu menjaga kelestarian lingkungan.
        3. Meningkatnya sumber daya manusia yang berkualitas yang mampu memenuhi kebutuhan Undiksha menjadi universitas unggul berlandaskan falsafah Tri Hita Karana.
        4. Dihasilkannya kurikulum yang adaptif yang mampu memenuhi tuntutan dunia usaha dan industri dan terlaksananya pembelajaran yang inovatif, berpusat pada siswa, berbasis teknologi informasi, serta pembelajaran bilingual dan berbahasa asing penuh berlandaskan falsafah Tri Hita Karana.
        5. Tersedianya sarana dan prasarana yang lengkap, berkualitas, dan bertaraf nasional dan internasional berlandaskan falsafah Tri Hita Karana.
        6. Terwujudnya pengelolaan keuangan yang transparan, akuntabel dan transparan, serta unit-unit bisnis yang mampu menghasilkan generating avenue bagi Undiksha sebagai perguruan tinggi Badan Layanan Umum.
        7. Dihasilkannya karya penelitian yang kompetitif, inovatif, dan kolaboratif, baik pada tingkat nasional, regional maupun internasional dan publikasi hasil-hasil penelitian dalam jurnal nasional terakreditasi dan jurnal internasional bereputasi berlandaskan falsafah Tri Hita Karana.
        8. Dihasilkannya karya pengabdian kepada masyarakat yang unggul yang mampu meningkatkan kesejahteraan masyarakat.
        9. Meningkatnya kerjasama dan kemitraan dalam dan luar negeri yang saling menguntungkan yang mampu meningkatkan proses pendidikan dan pengajaran, penelitian dan publikasi, serta pengabdian kepada masyarakat.
    """),
    ("Mengapa memilih undiksha?", "Karena sebagai salah satu universitas pendidikan terbesar di Bali, Undiksha menawarkan beberapa kesempatan untuk belajar tentang masing-masing disiplin pengetahuan dan sistem pendidikan. Terdapat beberapa hal yang menjadi alasan mahasiswa memilih Undiksha (titik pandang mahasiswa). Video mengapa memilih Undiksha dapat di tonton pada YouTube https://www.youtube.com/watch?v=BrMLWo0E8-0&t=1s"),
    ("Bagaimana kerjasama undiksha?", "Kerjasama Universitas Pendidikan Ganesha (Undiksha) untuk meningkatkan mutu pelaksanaan tridharma perguruan tinggi, Undiksha melakukan kerja sama dengan perguruan tinggi lain, dunia usaha dan industri, dan atau pihak lain, baik dari dalam maupun luar negeri. Kerjasama dilakukan dalam bentuk kerja sama akademik dan/atau non-akademik."),
    ("Prestasi dan capaian apa yang sudah didapatkan undiksha?", "Prestasi dan Capaian Universitas Pendidikan Ganesha (Undiksha) terus meningkatkan kualitas lembaga sehingga mampu bersaing tak hanya nasional, melainkan internasional. Melalui upaya tersebut, saat ini Undiksha berhasil menorehkan capaian kinerja yang positif dan membanggakan. Langkah Undiksha untuk meningkatkan kualitas terus berlanjut. Penguatan implementasi Tri Dharma Perguruan Tinggi terus dilakukan, demikian juga dengan akreditasi dan peningkatan kualitas sumber daya. Hal ini sebagai salah satu strategi untuk meningkatkan jumlah student body."),
    ("Bagaimana statistik capaian undiksha?", "Statistik dan Capaian Universitas Pendidikan Ganesha (Undiksha) Sejak perubahan status IKIP Negeri Singaraja menjadi Universitas Pendidikan Ganesha (Undiksha) pada 11 Mei 2006, Undiksha telah berhasil meraih berbagai prestasi dengan pencapaian gemilang. Bebagai prestasi Nasional dan Internasional ditorehkan oleh seluruh sivitas akademika untuk mencapai visi dan misi lembaga menuju Universitas UNGGUL di Asia. Terakreditasi A “Unggul” oleh BAN-PT serta meraih peringkat 11 Research and Innovation Ranking berdasarkan data yang dirilis oleh Scimago Institutions Rankings (SIR)."),
    ("Dimana lokasi undiksha?", "Universitas Pendidikan Ganesha (Undiksha) sebagai salah satu kampus pendidikan terbaik di Bali Utara, Undiksha memiliki beberapa lokasi kampus yang tersebar di dua Kabupaten. Masing-masing dapat digunakan sebagai tempat perkuliahan dan praktikum oleh prodi sesuai ketentuan lembaga. Penyelenggarakan layanan pendidikan dilakukan terpusat melalui Kampus Tengah Undiksha yang berlokasi di Jalan Udayana No.11 Singaraja - Bali 81116 Kota Singaraja, Kabupaten Buleleng, Bali."),
    ("Kontak undiksha yang dapat dihubungi?", """
        - Alamat: Jalan Udayana No.11 Singaraja - Bali 81116
        - Telepon: (0362) 22570
        - Email: humas@undiksha.ac.id
        - Website: www.undiksha.ac.id
        - Facebook: https://www.facebook.com/undiksha.bali
        - Instagram: https://www.instagram.com/undiksha.bali
        - Twitter: https://twitter.com/undikshabali
        - YouTube: https://www.youtube.com/universitaspendidikanganesha
        - Tiktok: https://www.tiktok.com/@undiksha.bali
        - LinkedIn: https://www.linkedin.com/school/universitas-pendidikan-ganesha
    """),
    ("Ada berapa fakultas di undiksha?", "Terdapat 9 Daftar Pilihan Fakultas di Universitas Pendidikan Ganesha."),
    ("Fakultas apa saja yang ada di undiksha?", "Universitas Pendidikan Ganesha (Undiksha) menawarkan sembilan fakultas untuk mahasiswa. Daftar fakultas tersebut meliputi Fakultas Teknik dan Kejuruan (FTK), Fakultas Olahraga dan Kesehatan (FOK), Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA), Fakultas Ilmu Pendidikan (FIP), Fakultas Hukum dan Ilmu Sosial (FHIS), Fakultas Ekonomi (FE), Fakultas Bahasa dan Seni (FBS), Fakultas Kedokteran (FK), dan Fakultas Pascasarjana. Masing-masing fakultas memiliki berbagai program studi yang dirancang untuk memenuhi beragam minat dan tujuan akademik mahasiswa."),
    ("Ada berapa jalur penerimaan mahasiswa baru di undiksha?", "Jalur Penerimaan Mahasiswa Universitas Pendidikan Ganesha (Undiksha) menyediakan tiga jalur penerimaan mahasiswa untuk tahun akademik 2024/2025 dengan mekanisme seleksi yang berbeda."),
    ("Dimana bisa melihat sumber informasi penerimaan mahasiswa baru undiksha?", "Ayo Kuliah di Universitas Pendidikan Ganesha (Undiksha): https://undiksha.ac.id/pmb/"),
    ("Apakah ada grup PMB undiksha?", "Untuk informasi lebih lanjut, yuk bergabung di group PMB 2024 di telegram https://go.undiksha.ac.id/ayookuliahdiundiksha."),
    ("Bagaimana proses pendaftaran di undiksha?", "Jalur Penerimaan Mahasiswa Universitas Pendidikan Ganesha (Undiksha) menyediakan tiga jalur penerimaan mahasiswa untuk tahun akademik 2024/2025 dengan mekanisme seleksi yang berbeda. Jalur pertama adalah Seleksi Nasional Berbasis Prestasi (SNBP), yang menilai calon mahasiswa berdasarkan prestasi akademik mereka, termasuk nilai rapor dan prestasi lainnya. Jalur kedua adalah Seleksi Nasional Berbasis Tes (SNBT), yang memilih siswa berdasarkan hasil tes UTBK yang mengukur kemampuan penalaran dan pemecahan masalah. Jalur ketiga adalah Seleksi Mandiri (SMBJM), di mana Undiksha mengadakan seleksi dengan tiga metode, yaitu penilaian prestasi, skor UTBK SNBT, dan ujian mandiri berbasis CBT."),
    ("Kapan jadwal SNBP undiksha?", "Mulai tanggal 28 Desember 2023"),
    ("Kapan jadwal SNBT undiksha?", "Mulai tanggal 08 Januari - 15 Februari 2024"),
    ("Kapan jadwal SMBJM undiksha?", "Mulai tanggal 14-22 Juni 2024")
]

@pytest.mark.parametrize("question, expected_response", test_cases)
def test_chatbot(question, expected_response):
    result = query_and_validate(
        question=question,
        expected_response=expected_response
    )
    assert result

# Run: pytest -s test/api_test.py