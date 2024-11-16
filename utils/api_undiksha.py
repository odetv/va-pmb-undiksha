import sys
import os
import requests
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.agent_state import AgentState
from datetime import datetime


load_dotenv()
API_KTM_UNDIKSHA_AUTH_URL = os.getenv("API_KTM_UNDIKSHA_AUTH_URL")
API_KTM_UNDIKSHA_USERNAME = os.getenv("API_KTM_UNDIKSHA_USERNAME")
API_KTM_UNDIKSHA_PASSWORD = os.getenv("API_KTM_UNDIKSHA_PASSWORD")
API_KTM_UNDIKSHA_RESPONSE_URL = os.getenv("API_KTM_UNDIKSHA_RESPONSE_URL")
API_KELULUSAN_UNDIKSHA_AUTH_URL = os.getenv("API_KELULUSAN_UNDIKSHA_AUTH_URL")
API_KELULUSAN_UNDIKSHA_USERNAME = os.getenv("API_KELULUSAN_UNDIKSHA_USERNAME")
API_KELULUSAN_UNDIKSHA_PASSWORD = os.getenv("API_KELULUSAN_UNDIKSHA_PASSWORD")
API_KELULUSAN_UNDIKSHA_RESPONSE_URL = os.getenv("API_KELULUSAN_UNDIKSHA_RESPONSE_URL")


def get_current_year():
    return datetime.now().year


def get_auth_token_ktm():
    body = {
        "username": API_KTM_UNDIKSHA_USERNAME,
        "password": API_KTM_UNDIKSHA_PASSWORD
    }
    try:
        response = requests.post(API_KTM_UNDIKSHA_AUTH_URL, data=body)
        response.raise_for_status()
        # print("Respon autentikasi:", response.text)
        result = response.json()
        if result["status"] == "success":
            return result["data"]
        else:
            raise Exception(f"Autentikasi gagal: {result['message']}")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Terjadi kesalahan saat mengakses API autentikasi: {e}")
    

def get_auth_token_kelulusan():
    body = {
        "username": API_KELULUSAN_UNDIKSHA_USERNAME,
        "password": API_KELULUSAN_UNDIKSHA_PASSWORD
    }
    try:
        response = requests.post(API_KELULUSAN_UNDIKSHA_AUTH_URL, data=body)
        response.raise_for_status()
        # print("Respon autentikasi:", response.text)
        result = response.json()
        if result["token"] == result["token"]:
            return result["token"]
        else:
            raise Exception(f"Autentikasi gagal: {result['message']}")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Terjadi kesalahan saat mengakses API autentikasi: {e}")


def show_ktm_mhs(state: AgentState):
    id_nim_mhs = state.get("idNIMMhs")
    token = get_auth_token_ktm()
    headers = {
        "token": token
    }
    url_get_ktm = f"{API_KTM_UNDIKSHA_RESPONSE_URL}/{id_nim_mhs}?token={token}"
    try:
        response = requests.get(url_get_ktm, headers=headers)
        response.raise_for_status()
        url_ktm_mhs = url_get_ktm
        state["urlKTMMhs"] = url_ktm_mhs
        return url_ktm_mhs
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Terjadi kesalahan saat mengakses API KTM: {e}")


def show_kelulusan_pmb(state: AgentState):
    noPendaftaran = state.get("noPendaftaran")
    tglLahirPendaftar = state.get("tglLahirPendaftar")
    token = get_auth_token_kelulusan()
    tahun = get_current_year()
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    body = {
        "tahun": tahun,
        "no-pendaftaran": noPendaftaran,
        "tgl-lahir": tglLahirPendaftar
    }

    try:
        response = requests.post(API_KELULUSAN_UNDIKSHA_RESPONSE_URL, headers=headers, data=body)
        response.raise_for_status()
        result = response.json()
        if result["status"]:
            data_siswa = result["data"][0]
            informasi_kelulusan = {
                "nomor_pendaftaran": data_siswa["nomor_pendaftaran"],
                "nama_siswa": data_siswa["nama_siswa"],
                "tgl_lahir": data_siswa["tgl_lahir"],
                "tahun": data_siswa["tahun"],
                "program_studi": data_siswa["program_studi"],
                "status_kelulusan": data_siswa["status_kelulusan"]
            }
            # print(f"Informasi Kelulusan: {informasi_kelulusan}")
            return informasi_kelulusan
        else:
            raise Exception(f"Data tidak ditemukan: {result.get('message', 'Tidak ada pesan kesalahan')}")
    
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            print("Detail kesalahan:", e.response.text)
        raise SystemExit(f"Terjadi kesalahan saat mengakses API kelulusan: {e}")
        # return "Maaf sedang terjadi kesalahan pada sistem."



# DEBUG AUTH
# print(get_auth_token_ktm())
# print(get_auth_token_kelulusan())