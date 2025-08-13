import FreeSimpleGUI as sg
from PIL import Image
from PIL.ImageTk import PhotoImage  # For Tkinter icon handlin
import tempfile
import os

from image_recommender.pipeline.search_pipeline import combined_similarity_search

# CLIP index and mapping PAths
CLIP_INDEX_PATH = os.path.join('data', 'out', 'clip_index.ann')
CLIP_MAPPING_PATH = os.path.join('data', 'out', 'index_to_id.json')

# Logo
LOGO_PATH = os.path.join(os.path.dirname(__file__), '..', 'logo.png')

# Theme
sg.theme('LightGrey1')

# Preload and resize logo for UI header
logo_img = Image.open(LOGO_PATH)
logo_img = logo_img.resize((150, 150), Image.LANCZOS)
temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
logo_img.save(temp_logo.name)


# UI layout

layout = [
    [sg.Push(),
     sg.Image(temp_logo.name, key='-HEADER_LOGO-'),
     sg.Text(
         'Image Recommender',
         font=('Helvetica', 48, 'bold'),
         text_color='#e60028',
         background_color='#ffffff'
     ),
     sg.Push()],
    [sg.HorizontalSeparator(color='#e60028')],
    [sg.Text(
        'Select an image to find similar pictures:',
        font=('Helvetica', 16),
        background_color='#ffffff'
    )],
    [sg.Input(
        key='-FILE-',
        enable_events=True,
        size=(50, 1),
        background_color='#f0f0f0',
        text_color='#000000'
    ),
     sg.FileBrowse(
         button_text='Browse',
         file_types=(('Image Files', '*.png;*.jpg;*.jpeg'),),
         button_color=('#ffffff', '#e60028')
     )],
    [sg.Column(
        [[sg.Image(key='-QUERY-', size=(300, 300), background_color='#ffffff')]],
        justification='center',
        element_justification='center',
        background_color='#ffffff'
    )],
    [sg.Text(
        'Top 5 Results:',
        font=('Helvetica', 20),
        pad=((0, 0), (10, 0)),
        background_color='#ffffff',
        text_color='#e60028'
    )],
    [
      sg.Column([
        [sg.Image(key=f'-RES{i}-', size=(200, 200), background_color='#ffffff'),
         sg.Text(
             '',
             key=f'-SCORE{i}-',
             font=('Helvetica', 14),
             text_color='#e60028',
             background_color='#ffffff'
         )]
        for i in range(3)
      ], justification='center', element_justification='center', background_color='#ffffff')
    ],
    [sg.Push(),
     sg.Button('Search', bind_return_key=True, button_color=('#ffffff', '#e60028')),
     sg.Button('Exit', button_color=('#ffffff', '#a60e2f')),
     sg.Push()]
]


# Create window with white background
window = sg.Window(
    'Image Recommender',
    layout,
    resizable=True,
    finalize=True,
    element_justification='center',
    background_color='#ffffff',
    margins=(20, 20)
)


# Maximize window
try:
    window.Maximize()
except Exception:
    pass

# Set window icon via Tkinter for correct taskbar/icon scaling (optional)
try:
    tk_root = window.TKroot
    icon_img = Image.open(LOGO_PATH).resize((32, 32), Image.LANCZOS)
    tk_icon = PhotoImage(icon_img)
    tk_root.iconphoto(False, tk_icon)
except Exception:
    pass

# Track temp files for cleanup
temp_files = [temp_logo.name]


# Event loop
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    if event == 'Search' and values['-FILE-']:
        query_path = values['-FILE-']
        try:
            # Display query image
            img = Image.open(query_path)
            img.thumbnail((300, 300), Image.LANCZOS)
            temp_q = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img.save(temp_q.name)
            temp_files.append(temp_q.name)
            window['-QUERY-'].update(filename=temp_q.name)

            # Run search pipeline
            results = combined_similarity_search(
                query_path,
                CLIP_INDEX_PATH,
                CLIP_MAPPING_PATH,
                top_k_result=5
            )
            # Update result images and scores
            for i, (path, score) in enumerate(results):
                res_img = Image.open(path)
                res_img.thumbnail((200, 200), Image.LANCZOS)
                temp_r = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                res_img.save(temp_r.name)
                temp_files.append(temp_r.name)
                window[f'-RES{i}-'].update(filename=temp_r.name)
                window[f'-SCORE{i}-'].update(f"Score: {score:.4f}")
        except Exception as e:
            sg.popup_error(f"Error processing image:\n{e}")


# Cleanup and close
window.close()
for f in temp_files:
    try:
        os.remove(f)
    except:
        pass
