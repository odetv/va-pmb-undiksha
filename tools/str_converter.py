import ast

def str_to_list(string):
    # Jika response adalah string tanpa tanda kurung, buat manual jadi list
    if isinstance(string, str) and ',' not in string:
        return [string]
    try:
        # Bersihkan string dari karakter yang tidak perlu
        cleaned_string = string.strip()
        # Ubah string menjadi list Python
        converted_list = ast.literal_eval(cleaned_string)
        return converted_list
    except (ValueError, SyntaxError):
        print(f"Error converting string to list: {string}")
        return []
