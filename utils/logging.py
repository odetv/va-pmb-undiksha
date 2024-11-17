import os
import openpyxl
from openpyxl import Workbook
from datetime import datetime
import random
import string

def generate_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def log(log_data):
    log_dir = "api/logs"
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "logs_data.xlsx")
    try:
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
    except FileNotFoundError:
        wb = Workbook()
        ws = wb.active
        ws.append(["ID", "Timestamp", "Method", "Status Code", "Success", "Description"])
    ws.append([
        log_data["id"],
        log_data["timestamp"],
        log_data["method"],
        log_data["status_code"],
        log_data["success"],
        log_data["description"]
    ])
    wb.save(file_path)