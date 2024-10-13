from IPython.display import Image, display
import os

def get_graph_image(graph):
# Menghasilkan gambar dari graf
    img_data = graph.get_graph().draw_mermaid_png()

    # Cek apakah folder graph ada
    if not os.path.exists('graph'):
        os.makedirs('graph')

    # Menentukan nama file dan path folder tempat menyimpan gambar
    file_path = 'graph/shavira_graph.png'

    # Menyimpan gambar dalam format PNG ke folder yang diinginkan
    with open(file_path, 'wb') as file:
        file.write(img_data)

    # # Menampilkan gambar di notebook
    # display(Image(img_data))