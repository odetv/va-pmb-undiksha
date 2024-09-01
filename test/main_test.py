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
    ("Siapa rektor undiksha?", "Prof. Dr. I Wayan Lasmawan, M.Pd."),
    ("Apa visi undiksha?", "Menjadi Universitas Unggul Berlandaskan Falsafah Tri Hita Karana di Asia Pada Tahun 2045"),
    ("Apa saja fakultas yang ada di undiksha?", "Fakultas Bahasa dan Seni, Fakultas Teknik dan Kejuruan, Fakultas Olahraga dan Kesehatan, Fakultas Matematika dan Ilmu Pengetahuan Alam, Fakultas Ekonomi dan Bisnis, Fakultas Hukum dan Ilmu Sosial")
]

@pytest.mark.parametrize("question, expected_response", test_cases)
def test_chatbot(question, expected_response):
    result = query_and_validate(
        question=question,
        expected_response=expected_response
    )
    assert result

# Run: pytest -s test/api_test.py