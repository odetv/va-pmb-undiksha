import os


def get_graph_image(graph):
    img_data = graph.get_graph().draw_mermaid_png()
    if not os.path.exists("src/graph"):
        os.makedirs("src/graph")
    file_path = "src/graph/graph-va-pmb-undiksha.png"
    with open(file_path, "wb") as file:
        file.write(img_data)