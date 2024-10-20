import sys
import os
import requests
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.agent_state import AgentState


load_dotenv()
API_UNDIKSHA_AUTH_URL = os.getenv('API_UNDIKSHA_AUTH_URL')
API_UNDIKSHA_USERNAME = os.getenv('API_UNDIKSHA_USERNAME')
API_UNDIKSHA_PASSWORD = os.getenv('API_UNDIKSHA_PASSWORD')
API_UNDIKSHA_KTM_URL = os.getenv('API_UNDIKSHA_KTM_URL')


def get_auth_token():
    body = {
        "username": API_UNDIKSHA_USERNAME,
        "password": API_UNDIKSHA_PASSWORD
    }
    try:
        response = requests.post(API_UNDIKSHA_AUTH_URL, data=body)
        response.raise_for_status()
        # print("Respon autentikasi:", response.text)
        result = response.json()
        if result["status"] == "success":
            return result["data"]
        else:
            raise Exception(f"Autentikasi gagal: {result['message']}")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Terjadi kesalahan saat mengakses API autentikasi: {e}")


def cetak_ktm_mhs(state: AgentState):
    id_nim_mhs = state.get("idNIMMhs")
    token = get_auth_token()
    headers = {
        "token": token
    }
    url_get_ktm = f"{API_UNDIKSHA_KTM_URL}/{id_nim_mhs}?token={token}"
    try:
        response = requests.get(url_get_ktm, headers=headers)
        response.raise_for_status()
        url_ktm_mhs = url_get_ktm
        state["urlKTMMhs"] = url_ktm_mhs
        return url_ktm_mhs
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Terjadi kesalahan saat mengakses API KTM: {e}")


# DEBUG
# state = {"idNIMMhs": "2115101014"}
# print(cetak_ktm_mhs(state))