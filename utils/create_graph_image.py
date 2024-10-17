import os

def get_graph_image(graph):
    img_data = graph.get_graph().draw_mermaid_png()
    if not os.path.exists('graph'):
        os.makedirs('graph')
    file_path = 'graph/graph-va-pmb-undiksha.png'
    with open(file_path, 'wb') as file:
        file.write(img_data)