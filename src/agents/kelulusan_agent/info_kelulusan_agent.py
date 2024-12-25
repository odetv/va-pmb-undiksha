import re
from langchain_core.messages import SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.api_undiksha import show_kelulusan_pmb
from utils.debug_time import time_check


@time_check
def infoKelulusanAgent(state: AgentState):
    info = "\n--- Info Kelulusan SMBJM ---"
    print(info)

    noPendaftaran_match = re.search(r"\b(?:nmr|no|nomor|nmr.|no.|nomor.|nmr. |no. |nomor. )\s*pendaftaran.*?(\b\d{10}\b)(?!\d)", state["kelulusanQuestion"], re.IGNORECASE)
    tglLahirPendaftar_match = re.search(r"(?:ttl|tanggal lahir|tgl lahir|lahir|tanggal-lahir|tgl-lahir|lhr|tahun|tahun lahir|thn lahir|thn|th lahir)[^\d]*(\d{4}-\d{2}-\d{2})", state["kelulusanQuestion"], re.IGNORECASE)
    state["noPendaftaran"] = noPendaftaran_match.group(1)
    state["tglLahirPendaftar"] = tglLahirPendaftar_match.group(1)
    kelulusan_info = show_kelulusan_pmb(state)

    try:
        no_pendaftaran = kelulusan_info.get("nomor_pendaftaran", "")
        nama_siswa = kelulusan_info.get("nama_siswa", "")
        tgl_lahir = kelulusan_info.get("tgl_lahir", "")
        tgl_daftar = kelulusan_info.get("tahun", "")
        pilihan_prodi = kelulusan_info.get("program_studi", "")
        status_kelulusan = kelulusan_info.get("status_kelulusan", "")
        response = f"""
            Berikut informasi Kelulusan Peserta SMBJM di Undiksha (Universitas Pendidikan Ganesha).
            - Nomor Pendaftaran: {no_pendaftaran}
            - Nama Siswa: {nama_siswa}
            - Tanggal Lahir: {tgl_lahir}
            - Tahun Daftar: {tgl_daftar}
            - Pilihan Program Studi: {pilihan_prodi}
            - Status Kelulusan: {status_kelulusan}
            Berdasarkan informasi, berikan ucapan selamat bergabung di menjadi bagian dari Universitas Pendidikan Ganesha jika {nama_siswa} lulus, atau berikan motivasi {nama_siswa} jika tidak lulus.
        """
        agentOpinion = {
            "question": state["kelulusanQuestion"],
            "answer": response
        }
        state["finishedAgents"].add("infoKelulusan_agent")

        return {"answerAgents": [agentOpinion]}

    except Exception as e:
        # print("Error retrieving graduation information:", e)
        prompt = f"""
            Anda adalah agen pengirim pesan informasi Undiksha.
            Tugas Anda untuk memberitahu pengguna bahwa:
            Terjadi kesalahan dalam mengecek informasi kelulusan.
            - Ini pesan kesalahan dari sistem coba untuk diulas lebih lanjut agar lebih sederhana untuk diberikan ke pengguna (Jika terdapat informasi yang bersifat penting atau rahasia maka ganti menjadi "Tidak disebutkan"): {kelulusan_info}
        """
        messages = [
            SystemMessage(content=prompt)
        ]
        response = chat_llm(messages)
        agentOpinion = {
            "answer": response
        }
        state["finishedAgents"].add("infoKelulusan_agent")

        return {"answerAgents": [agentOpinion]}