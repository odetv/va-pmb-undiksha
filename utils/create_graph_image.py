import os


GRAPH_IMAGE_PATH = "src/graph"

def get_graph_image(graph):
    img_data = graph.get_graph().draw_mermaid_png()
    if not os.path.exists(GRAPH_IMAGE_PATH):
        os.makedirs(GRAPH_IMAGE_PATH)
    file_path = GRAPH_IMAGE_PATH+"/graph-va-pmb-undiksha.png"
    with open(file_path, "wb") as file:
        file.write(img_data)