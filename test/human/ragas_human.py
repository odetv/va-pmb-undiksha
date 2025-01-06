import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data.test_case import questions, ground_truths
from main import rag_adaptive


def run_opsi1():
    # Opsi 1 (Evaluasi dibedakan per agent yang ada):
    # - General Agent (44 Test Case)
    # - KTM Agent (2 Test Case)
    # - Kelulusan Agent (2 Test Case)
    # - Out of Context Agent (2 Test Case)

    # - Buat fungsi untuk menangkap hasil context dan answer secara berulangan berdasarkan jumlah question pada endpoint terakhir masing-masing agent
    # - Jalankan fungsi tersebut
    # - Simpan ke file excel dengan kolom question, ground truth, context, answer
    # - Lakukan perhitungan RAGAS manual dengan excel dari hasil masing-masing agent setiap masing-masing metrik
    return


def run_opsi2():
    # Opsi 2 (Evaluasi hanya pada agent yang menggunakan RAG):
    # - General Agent (50 Test Case)

    # - Buat fungsi untuk menangkap hasil context dan answer secara berulangan berdasarkan jumlah question pada endpoint terakhir general agent
    # - Jalankan fungsi tersebut
    # - Simpan ke file excel dengan kolom question, ground truth, context, answer
    # - Lakukan perhitungan RAGAS manual dengan excel dari hasil general agent setiap masing-masing metrik
    return


def run_opsi3():
    # Opsi 3 (Evaluasi pada hasil kompilasi model multi-agent atau gabungan agent-agent oleh LangGraph):
    # - Multi-Agent (50 Test Case)

    # - Buat fungsi untuk menangkap hasil context dan answer secara berulangan berdasarkan jumlah question pada endpoint terakhir multi-agent yaitu rag_adaptive pada langgraph
    # - Jalankan fungsi tersebut
    # - Simpan ke file excel dengan kolom question, ground truth, context, answer
    # - Lakukan perhitungan RAGAS manual dengan excel dari hasil multi-agent setiap masing-masing metrik
    return


def main():
    run_opsi1()
    run_opsi2()
    run_opsi3()