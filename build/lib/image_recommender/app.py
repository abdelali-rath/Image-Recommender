import PySimpleGUI as sg
from PIL import Image
import tempfile
import os

from image_recommender.pipeline.search_pipeline import combined_similarity_search

# Paths to your CLIP index and mapping
CLIP_INDEX_PATH = os.path.join('data', 'out', 'clip_index.ann')
CLIP_MAPPING_PATH = os.path.join('data', 'out', 'index_to_id.json')

# GUI layout
sg.theme('DarkBlue3')  # A friendly theme
layout = [
    [sg.Text('📷 Image Recommender', font=('Helvetica', 16), justification='center', expand_x=True)],
    [sg.Text('Select an image to find similar pictures:'),
     sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse(file_types=(('Image Files', '*.png;*.jpg;*.jpeg'),))],
    [sg.Image(key='-QUERY-')],
    [sg.Text('Top 3 Results:', font=('Helvetica', 14))],
    [sg.Column([[sg.Image(key=f'-RES{i}-'), sg.Text('', key=f'-SCORE{i}-')]
               for i in range(3)], scrollable=True, size=(600, 200))],
    [sg.Button('Search', bind_return_key=True), sg.Button('Exit')]
]

window = sg.Window('Image Recommender', layout, size=(700, 600))

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    if event == 'Search' and values['-FILE-']:
        query_path = values['-FILE-']
        try:
            # Display query image (scaled)
            img = Image.open(query_path)
            img.thumbnail((300, 300))
            bio = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img.save(bio.name)
            window['-QUERY-'].update(filename=bio.name)

            # Run search pipeline
            results = combined_similarity_search(
                query_path,
                CLIP_INDEX_PATH,
                CLIP_MAPPING_PATH,
                top_k_result=3
            )
            # Update result images and scores
            for i, (path, score) in enumerate(results):
                res_img = Image.open(path)
                res_img.thumbnail((150, 150))
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                res_img.save(tmp.name)
                window[f'-RES{i}-'].update(filename=tmp.name)
                window[f'-SCORE{i}-'].update(f"Score: {score:.4f}")
        except Exception as e:
            sg.popup_error(f"Error processing image:\n{e}")

# Clean up temp files
window.close()
temp_dir = tempfile.gettempdir()
for f in os.listdir(temp_dir):
    if f.startswith('tmp') and f.endswith('.png'):
        try:
            os.remove(os.path.join(temp_dir, f))
        except:
            pass
