import os
import asyncio
from crawl4ai import AsyncWebCrawler
from docx import Document
from docx2pdf import convert
from dotenv import load_dotenv


load_dotenv()
DATASETS_DIR = os.getenv("APP_DATASETS_DIR")


def save_to_word(url, content):
    filename = os.path.join(DATASETS_DIR, url.replace("https://", "").replace("/", "_") + ".docx")
    doc = Document()
    for line in content.splitlines():
        doc.add_paragraph(line)
    doc.save(filename)
    convert(filename)
    # os.remove(filename)


async def scrape_and_save(url):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
        save_to_word(url, result.markdown)


async def main():
    urls = [
        "https://undiksha.ac.id/tentang-undiksha/",
        "https://undiksha.ac.id/tentang-undiksha/pimpinan/",
        "https://undiksha.ac.id/tentang-undiksha/fasilitas/",
        "https://undiksha.ac.id/kemahasiswaan/beasiswa/",
        "https://undiksha.ac.id/akademik/",
        "https://undiksha.ac.id/akademik/kurikulum/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-bahasa-dan-seni/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-ekonomi/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-hukum-dan-ilmu-sosial/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-ilmu-pendidikan/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-kedokteran/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-matematika-dan-ilmu-pengetahuan-alam/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-olahraga-dan-kesehatan/",
        "https://undiksha.ac.id/akademik/fakultas/fakultas-teknik-dan-kejuruan/",
        "https://undiksha.ac.id/akademik/program/pascasarjana/",
        "https://undiksha.ac.id/kemahasiswaan/ukm/",
        "https://upttik.undiksha.ac.id/faq/",
        "https://upttik.undiksha.ac.id/profil/",
        "https://upttik.undiksha.ac.id/",
        "https://undiksha.ac.id/pmb/",
        "https://undiksha.ac.id/pmb/tahun2024/daya-tampung/",
        "https://undiksha.ac.id/pmb/tahun2024/snbp/",
        "https://undiksha.ac.id/snbt-2024/",
        "https://undiksha.ac.id/pmb/tahun2024/smbjm/",
        "https://undiksha.ac.id/pmb/tahun2024/seleksi-mandiri-jalur-prestasi/",
        "https://undiksha.ac.id/pmb/tahun2024/seleksi-mandiri-jalur-skor-utbk/",
        "https://undiksha.ac.id/pmb/tahun2024/seleksi-mandiri-cbt/",
        "https://penerimaan.undiksha.ac.id/",
        "https://undiksha.ac.id/panduan-pendaftaran-kembali-calon-mahasiswa-baru-undiksha-hasil-smbjm-dengan-skor-utbk-snbt-dan-prestasi-tahun-akademik-2024-2025/",
        "https://undiksha.ac.id/panduan-daftar-kembali-smbjm-cbt-2024/",
        "https://undiksha.ac.id/panduan-daftar-kembali-snbp-2024/",
        "https://undiksha.ac.id/daftar-kembali-snbt-2024/"
    ]
    data_folder = DATASETS_DIR
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    tasks = [scrape_and_save(url) for url in urls]
    await asyncio.gather(*tasks)


asyncio.run(main())
