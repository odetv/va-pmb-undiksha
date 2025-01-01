import os
import random
import string
import nest_asyncio
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes


# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate({
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY"),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN")
})
firebase_admin.initialize_app(cred)
db = firestore.client()


# Mendefinisikan zona waktu GMT+8 untuk WITA
wita_timezone = timezone(timedelta(hours=8))


# Daftar admin ID
ADMIN_IDS = [int(x) for x in os.getenv("ID_TELEGRAM_USER_ADMIN").split(",")] if os.getenv("ID_TELEGRAM_USER_ADMIN") else []


# Command handler untuk start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_name = update.effective_user.username
    user_ref = db.collection("users").document(str(user_id))
    if not user_ref.get().exists:
        role = 'admin' if user_id in ADMIN_IDS else 'registered'
        user_ref.set({
            'user_id': user_id,
            'username': user_name,
            'role': role,
            'created_at': datetime.now(wita_timezone),
            'updated_at': datetime.now(wita_timezone)
        })
        await update.message.reply_text(f"Selamat bergabung {user_ref.get().to_dict()['username']}!\nAnda terdaftar dengan role {user_ref.get().to_dict()['role']} dalam bot ini!\n\nGunakan /help untuk melihat panduan penggunaan bot.")
    else:
        await update.message.reply_text(f"Selamat datang kembali {user_ref.get().to_dict()['username']}!\n\nGunakan /help untuk melihat panduan penggunaan bot.")


# Command handler untuk help
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_message = (
        "Panduan Penggunaan Bot\n\n"
        "Berikut adalah daftar perintah yang dapat digunakan:\n"
        "/start - Mulai bot\n"
        "/help - Panduan bot\n"
        "/auth_on - Hidupkan fitur auth token\n"
        "/auth_off - Matikan fitur auth token\n"
        "/check_auth - Cek status fitur auth token\n"
        "/list_users - Lihat pengguna yang terdaftar\n"
        "/set_user_role [id_user] [role] - Ubah role akses pengguna (admin, member, atau registered)\n"
        "/create_token - Buat token baru\n"
        "/list_tokens - Lihat daftar token yang ada\n"
        "/set_token_status [token] [1/0] - Ubah status token (1 untuk valid, 0 untuk invalid)\n"
        "/revoke_token [token] - Hapus token tertentu\n"
        "/revoke_token_all - Hapus semua token\n"
    )
    await update.message.reply_text(start_message)


# Command handler untuk mengaktifkan autentikasi
async def auth_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Command accessed: /auth_on")
    auth_ref = db.collection("settings").document("auth")
    auth_ref.set({"status": 1})
    await update.message.reply_text("Fitur autentikasi telah dihidupkan.")


# Command handler untuk menonaktifkan autentikasi
async def auth_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("Command accessed: /auth_off")
    auth_ref = db.collection("settings").document("auth")
    auth_ref.set({"status": 0})
    await update.message.reply_text("Fitur autentikasi telah dimatikan.")


# Command handler untuk memeriksa status autentikasi
async def check_auth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    auth_ref = db.collection("settings").document("auth")
    auth_doc = auth_ref.get()
    if auth_doc.exists:
        auth_status = auth_doc.to_dict().get("status", "unknown")
        status_message = "hidup" if auth_status == 1 else "mati"
        await update.message.reply_text(f"Status autentikasi token saat ini {status_message}.")
    else:
        await update.message.reply_text("Pengaturan autentikasi belum diatur.")


# Fungsi untuk memeriksa peran pengguna
async def check_user_role(update: Update, required_role: str):
    user_id = update.effective_user.id
    user_doc = db.collection("users").document(str(user_id)).get()
    if not user_doc.exists:
        await update.message.reply_text("Anda tidak terdaftar untuk menggunakan bot ini.")
        return False
    user_info = user_doc.to_dict()
    if user_info.get('role') not in [required_role, 'admin']:
        await update.message.reply_text("Anda tidak memiliki izin untuk mengakses perintah ini.")
        return False
    return True


# Command handler untuk mengubah peran pengguna
async def set_user_role(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_user_role(update, 'admin'):
        return
    if len(context.args) < 2:
        await update.message.reply_text(
            "Format perintah salah!\n\nGunakan /set_user_role [id_user] [role]\n"
            "Contoh: /set_user_role 123456789 member\n\n"
            "Keterangan:\n"
            "   - id_user = ID pengguna\n"
            "   - role = admin, member, atau registered\n"
            "   - admin = Akses penuh\n"
            "   - member = Akses terbatas\n"
            "   - registered = Tidak dapat mengakses bot"
        )
        return
    user_id = context.args[0]
    new_role = context.args[1]
    current_user_id = update.effective_user.id
    if str(current_user_id) == str(user_id):
        await update.message.reply_text("Anda tidak bisa mengubah peran Anda sendiri.")
        return
    if new_role not in ['admin', 'member', 'registered']:
        await update.message.reply_text("Role harus admin, member, atau registered.")
        return
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        await update.message.reply_text(f"Pengguna dengan ID {user_id} tidak ditemukan.")
        return
    user_ref.update({
        "role": new_role,
        "updated_at": datetime.now(wita_timezone)
    })
    await update.message.reply_text(f"Role pengguna {user_id} berhasil diubah menjadi {new_role}.")


# Command handler untuk daftar pengguna
async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_user_role(update, 'member'):
        return
    users_ref = db.collection("users").order_by("created_at", direction=firestore.Query.DESCENDING)
    docs = users_ref.stream()
    user_list = ""
    user_found = False
    index = 1
    for doc in docs:
        user_found = True
        user_info = doc.to_dict()
        user_id = user_info.get('user_id', 'N/A')
        username = user_info.get('username', 'N/A')
        role = user_info.get('role', 'N/A')
        created_time = user_info.get('created_at', 'N/A')
        updated_time = user_info.get('updated_at', 'N/A')
        if isinstance(created_time, datetime):
            created_time = created_time.astimezone(wita_timezone).strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(updated_time, datetime):
            updated_time = updated_time.astimezone(wita_timezone).strftime('%Y-%m-%d %H:%M:%S')
        user_list += (
            f"{index}. ID User: {user_id}\n"
            f"   - Username: {username}\n"
            f"   - Role: {role}\n"
            f"   - Created At: {created_time}\n"
            f"   - Updated At: {updated_time}\n\n"
        )
        index += 1
    if user_found:
        await update.message.reply_text(f"Daftar Pengguna\n\n{user_list.strip()}")
    else:
        await update.message.reply_text("Tidak ada pengguna yang terdaftar.")


# Command handler untuk buat token
async def create_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_user_role(update, 'member'):
        return
    token = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    current_time = datetime.now(wita_timezone)
    data = {
        'token': token,
        'created_at': current_time,
        'updated_at': current_time,
        'status': 1
    }
    doc_ref = db.collection("tokens").add(data)
    doc_snapshot = doc_ref[1].get()
    if doc_snapshot.exists:
        token_info = doc_snapshot.to_dict()
        created_time = token_info['created_at'].astimezone(wita_timezone).strftime('%Y-%m-%d %H:%M:%S') if 'created_at' in token_info else 'N/A'
        token_message = (
            "Token berhasil dibuat!\n\n"
            f"- Token: {token_info['token']}\n"
            f"- Status: {'Valid' if token_info['status'] == 1 else 'Invalid'}\n"
            f"- Created At: {created_time}"
        )
        await update.message.reply_text(token_message)
    else:
        await update.message.reply_text("Gagal membuat token. Silakan coba lagi!")


# Command handler untuk daftar token
async def list_tokens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_user_role(update, 'member'):
        return
    tokens_ref = db.collection("tokens").order_by("created_at", direction=firestore.Query.DESCENDING)
    docs = tokens_ref.stream()
    token_list = ""
    token_found = False
    index = 1
    for doc in docs:
        token_found = True
        token_info = doc.to_dict()
        created_time = token_info.get('created_at', 'N/A')
        if isinstance(created_time, datetime):
            created_time = created_time.astimezone(wita_timezone).strftime('%Y-%m-%d %H:%M:%S')
        updated_time = token_info.get('updated_at', 'N/A')
        if isinstance(updated_time, datetime):
            updated_time = updated_time.astimezone(wita_timezone).strftime('%Y-%m-%d %H:%M:%S')
        token_list += (
            f"{index}. Token: {token_info['token']}\n"
            f"      - Status: {'Valid' if token_info['status'] == 1 else 'Invalid'}\n"
            f"      - Created At: {created_time}\n"
            f"      - Updated At: {updated_time}\n\n"
        )
        index += 1
    if token_found:
        await update.message.reply_text(f"Daftar Token\n\n{token_list.strip()}")
    else:
        await update.message.reply_text("Tidak ada token yang tersedia.")


# Command handler untuk set status
async def set_token_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_user_role(update, 'member'):
        return
    if len(context.args) < 2:
        await update.message.reply_text("Format perintah salah!\n\nGunakan /set_token_status [token] [1/0]\nContoh: /set_token_status hU90f4 1\n\nKeterangan:\n   - token = Pilihan token\n   - 1 = Valid\n   - 0 = Invalid")
        return
    token, status = context.args[0], context.args[1]
    if status not in ['0', '1']:
        await update.message.reply_text("Status harus 0 atau 1.")
        return
    tokens_ref = db.collection("tokens").where("token", "==", token)
    docs = tokens_ref.stream()
    found = False
    for doc in docs:
        found = True
        doc.reference.update({"status": int(status), "updated_at": datetime.now(wita_timezone)})
    if found:
        await update.message.reply_text(f"Status token {token} diubah menjadi {status}.")
    else:
        await update.message.reply_text(f"Token {token} tidak ditemukan.")


# Command handler untuk revoke token
async def revoke_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_user_role(update, 'member'):
        return
    if not context.args:
        await update.message.reply_text("Format perintah salah!\n\nGunakan /revoke [token]\nContoh: /revoke hU90f4\n\nKeterangan:\n   - token = Pilihan token")
        return
    token = context.args[0]
    tokens_ref = db.collection("tokens").where("token", "==", token)
    docs = tokens_ref.stream()
    found = False
    for doc in docs:
        found = True
        doc.reference.delete()
    if found:
        await update.message.reply_text(f"Token {token} berhasil dihapus.")
    else:
        await update.message.reply_text(f"Token {token} tidak ditemukan.")


# Command handler untuk revoke semua token
async def revoke_token_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_user_role(update, 'member'):
        return
    tokens_ref = db.collection("tokens")
    docs = tokens_ref.stream()
    token_found = False
    for doc in docs:
        token_found = True
        doc.reference.delete()
    if token_found:
        await update.message.reply_text("Semua token telah dihapus.")
    else:
        await update.message.reply_text("Tidak ada token yang tersedia.")


# Fungsi untuk menangani perintah yang tidak dikenal
async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Perintah salah!\n\nGunakan /help untuk melihat panduan penggunaan bot.")


# Menghubungkan handler untuk setiap command
async def main():
    application = ApplicationBuilder().token(os.getenv("TOKEN_BOT_TELEGRAM")).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("auth_on", auth_on))
    application.add_handler(CommandHandler("auth_off", auth_off))
    application.add_handler(CommandHandler("check_auth", check_auth))
    application.add_handler(CommandHandler("list_users", list_users))
    application.add_handler(CommandHandler("set_user_role", set_user_role))
    application.add_handler(CommandHandler("create_token", create_token))
    application.add_handler(CommandHandler("list_tokens", list_tokens))
    application.add_handler(CommandHandler("set_token_status", set_token_status))
    application.add_handler(CommandHandler("revoke_token", revoke_token))
    application.add_handler(CommandHandler("revoke_token_all", revoke_token_all))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown_command))
    application.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    await application.run_polling()


# Jalankan bot
if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()
    import asyncio
    asyncio.run(main())