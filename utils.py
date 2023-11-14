import gradio as gr
from collections import defaultdict
import os
import sys
import subprocess as sp
import platform
import json
import html
import requests
import base64
import io
from PIL import Image
import boto3
from botocore.exceptions import ClientError
import traceback
import time

use_webui = os.environ.get('use_webui', False)
sagemaker_endpoint = os.environ.get('sagemaker_endpoint', None)
if sagemaker_endpoint:
    from sagemaker.predictor import Predictor
    from sagemaker.predictor_async import AsyncPredictor
    from sagemaker.serializers import JSONSerializer
    from sagemaker.deserializers import JSONDeserializer
    
    predictor = Predictor(endpoint_name=sagemaker_endpoint)
    async_predictor = AsyncPredictor(predictor=predictor, name=sagemaker_endpoint)

    s3_client = boto3.client('s3')
    s3_resource = boto3.resource("s3")

    def get_bucket_and_key(s3uri):
        pos = s3uri.find('/', 5)
        bucket = s3uri[5 : pos]
        key = s3uri[pos + 1 : ]
        return bucket, key

    def invoke_async_inference(payload):
        global async_predictor
        async_predictor.serializer = JSONSerializer()
        async_predictor.deserializer = JSONDeserializer()
        prediction = async_predictor.predict_async(payload)

        try:
            while True:
                body = handle_aysnc_inference(prediction.output_path)
                if body:
                    return {'status_code': 200, 'text': body}
                else:
                    time.sleep(1)
        except Exception as e:
            traceback.print_exc()
            print(e)
    
    def handle_aysnc_inference(s3uri):    
        try:
            output_bucket, output_key = get_bucket_and_key(s3uri)
            output_obj = s3_resource.Object(output_bucket, output_key)
            body = output_obj.get()["Body"].read().decode("utf-8")
            return body
        except ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                return None
            else:
                raise

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐

sd_samplers = [
    'DPM++ 2M Karras',
    'DPM++ SDE Karras',
    'DPM++ 2M SDE Exponential',
    'DPM++ 2M SDE Karras',
    'Euler a', 
    'Euler', 
    'LMS', 
    'Heun', 
    'DPM2', 
    'DPM2 a', 
    'DPM++ 2S a', 
    'DPM++ 2M', 
    'DPM++ SDE', 
    'DPM++ 2M SDE', 
    'DPM++ 2M SDE Heun', 
    'DPM++ 3M SDE', 
    'DPM++ 3M SDE Karras', 
    'DPM++ 3M SDE Exponential', 
    'DPM fast', 
    'DPM adaptive', 
    'LMS Karras', 
    'DPM2 Karras', 
    'DPM2 a Karras',
    'DPM++ 2S a Karras',
    'Restart',
    'DDIM',
    'PPIM',
    'UniPC'
]

sd_upscalers = [
    'None', 
    'Lanczos', 
    'Nearest', 
    'ESRGAN_4x', 
    'LDSR', 
    'R-ESRGAN 4x+', 
    'R-ESRGAN 4x+ Anime6B', 
    'ScuNET GAN', 
    'ScuNET PSNR', 
    'SwinIR 4x'
]

latent_upscale_modes = [
    "Latent",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "None",
    "Lanczos",
    "Nearest",
    "ESRGAN_4x",
    "LDSR",
    "R-ESRGAN 4x+",
    "R-ESRGAN 4x+ Anime6B",
    "ScuNET GAN",
    "ScuNET PSNR",
    "SwinIR 4x"
]

def handle_response(response):
    status_code = response['status_code'] if isinstance(response, dict) else response.status_code
    text = response['text'] if isinstance(response, dict) else response.text

    print(status_code)
    return status_code, text

max_controlnet_models = 1

api_endpoint = os.environ.get('api_endpoint', None)

sd_models = []

def refresh_sd_models():
    global sd_models
    payload = {'task': '/sdapi/v1/sd-models'}
    if sagemaker_endpoint:
        response = invoke_async_inference(payload)
    elif use_webui:
        response = requests.get(url=f'{api_endpoint}/sdapi/v1/sd-models')
    else:
        response = requests.post(url=f'{api_endpoint}/invocations', json=payload)

    status_code, text = handle_response(response)

    if status_code == 200:
        sd_models = [x["title"] for x in json.loads(text)]
    else:
        sd_models = [
            "v1-5-pruned-emaonly.safetensors [6ce0161689]"
        ]  

refresh_sd_models()

controlnet_models = []

def refresh_controlnet_models():
    global controlnet_models, use_webui
    payload = {'task': '/controlnet/model_list'}
    if sagemaker_endpoint:
        response = invoke_async_inference(payload)
    elif use_webui:
        response = requests.get(url=f'{api_endpoint}/controlnet/model_list')
    else:
        response = requests.post(url=f'{api_endpoint}/invocations', json=payload)

    status_code, text = handle_response(response)

    if status_code == 200:
        controlnet_models = ['None'] + json.loads(text)['model_list']
    else:
        controlnet_models = ['None']
    print(controlnet_models)

refresh_controlnet_models()

class FormComponent:
    def get_expected_parent(self):
        return gr.components.Form

class FormRow(FormComponent, gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"

class FormHTML(FormComponent, gr.HTML):
    """Same as gr.HTML but fits inside gradio forms"""

    def get_block_name(self):
        return "html"

class FormGroup(FormComponent, gr.Group):
    """Same as gr.Group but fits inside gradio forms"""

    def get_block_name(self):
        return "group"

class InputAccordion(gr.Checkbox):
    """A gr.Accordion that can be used as an input - returns True if open, False if closed.

    Actaully just a hidden checkbox, but creates an accordion that follows and is followed by the state of the checkbox.
    """

    global_index = 0

    def __init__(self, value, **kwargs):
        self.accordion_id = kwargs.get('elem_id')
        if self.accordion_id is None:
            self.accordion_id = f"input-accordion-{InputAccordion.global_index}"
            InputAccordion.global_index += 1

        kwargs_checkbox = {
            **kwargs,
            "elem_id": f"{self.accordion_id}-checkbox",
            "visible": False,
        }
        super().__init__(value, **kwargs_checkbox)

        self.change(fn=None, _js='function(checked){ inputAccordionChecked("' + self.accordion_id + '", checked); }', inputs=[self])

        kwargs_accordion = {
            **kwargs,
            "elem_id": self.accordion_id,
            "label": kwargs.get('label', 'Accordion'),
            "elem_classes": ['input-accordion'],
            "open": value,
        }
        self.accordion = gr.Accordion(**kwargs_accordion)

    def extra(self):
        """Allows you to put something into the label of the accordion.

        Use it like this:

        ```
        with InputAccordion(False, label="Accordion") as acc:
            with acc.extra():
                FormHTML(value="hello", min_width=0)

            ...
        ```
        """

        return gr.Column(elem_id=self.accordion_id + '-extra', elem_classes='input-accordion-extra', min_width=0)

    def __enter__(self):
        self.accordion.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accordion.__exit__(exc_type, exc_val, exc_tb)

    def get_block_name(self):
        return "checkbox"

class ToolButton(FormComponent, gr.Button):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, *args, **kwargs):
        classes = kwargs.pop("elem_classes", [])
        super().__init__(*args, elem_classes=["tool", *classes], **kwargs)

    def get_block_name(self):
        return "button"

class ResizeHandleRow(gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.elem_classes.append("resize-handle-row")

    def get_block_name(self):
        return "row"

originals = defaultdict(dict)

def patch(key, obj, field, replacement):
    """Replaces a function in a module or a class.

    Also stores the original function in this module, possible to be retrieved via original(key, obj, field).
    If the function is already replaced by this caller (key), an exception is raised -- use undo() before that.

    Arguments:
        key: identifying information for who is doing the replacement. You can use __name__.
        obj: the module or the class
        field: name of the function as a string
        replacement: the new function

    Returns:
        the original function
    """

    patch_key = (obj, field)
    if patch_key in originals[key]:
        raise RuntimeError(f"patch for {field} is already applied")

    original_func = getattr(obj, field)
    originals[key][patch_key] = original_func

    setattr(obj, field, replacement)

    return original_func

def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """

    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]

    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    self.webui_tooltip = kwargs.pop('tooltip', None)

    res = original_IOComponent_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    return res


def Block_get_config(self):
    config = original_Block_get_config(self)

    webui_tooltip = getattr(self, 'webui_tooltip', None)
    if webui_tooltip:
        config["webui_tooltip"] = webui_tooltip

    config.pop('example_inputs', None)

    return config


def BlockContext_init(self, *args, **kwargs):
    res = original_BlockContext_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    return res


def Blocks_get_config_file(self, *args, **kwargs):
    config = original_Blocks_get_config_file(self, *args, **kwargs)

    for comp_config in config["components"]:
        if "example_inputs" in comp_config:
            comp_config["example_inputs"] = {"serialized": []}

    return config


original_IOComponent_init = patch(__name__, obj=gr.components.IOComponent, field="__init__", replacement=IOComponent_init)
original_Block_get_config = patch(__name__, obj=gr.blocks.Block, field="get_config", replacement=Block_get_config)
original_BlockContext_init = patch(__name__, obj=gr.blocks.BlockContext, field="__init__", replacement=BlockContext_init)
original_Blocks_get_config_file = patch(__name__, obj=gr.blocks.Blocks, field="get_config_file", replacement=Blocks_get_config_file)

def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, 'label', None)
        if label is not None:
            break

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id, tooltip=f"{label}: refresh" if label else "Refresh")
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=refresh_components
    )
    return refresh_button

def open_folder(f):
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
        return
    elif not os.path.isdir(f):
        print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
        return

    path = os.path.normpath(f)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        sp.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        sp.Popen(["wsl-open", path])
    else:
        sp.Popen(["xdg-open", path])

def update_generation_info(generation_info, html_info, img_index):
    try:
        generation_info = json.loads(generation_info)
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info, gr.update()
        return plaintext_to_html(generation_info["infotexts"][img_index]), gr.update()
    except Exception:
        pass
    # if the json parse or anything else fails, just return the old html_info
    return html_info, gr.update()


def encode_image_to_base64(image):
    with io.BytesIO() as output_bytes:
        if type(image) == dict:
            image = image['image']
        format = "PNG" if image.mode == 'RGBA' else "JPEG"
        image.save(output_bytes, format=format)
        bytes_data = output_bytes.getvalue()

    encoded_string = base64.b64encode(bytes_data)

    base64_str = str(encoded_string, "utf-8")
    mimetype = "image/jpeg" if format == 'JPEG' else 'image/png'
    image_encoded_in_base64 = (
        "data:" + (mimetype if mimetype is not None else "") + ";base64," + base64_str
    )
    return image_encoded_in_base64

def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        print(e)

def plaintext_to_html(text, classname=None):
    print(text)
    print(type(text))
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))

    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"

def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "no image selected"

    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"

def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image

def webpath(fn):
    web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'

def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    head += stylesheet('style.css')

    return head

def javascript_html():
    head = ""
    script_js = ["script.js", 'ui.js']

    for script in script_js:
        head += f'<script type="text/javascript" src="{webpath(script)}"></script>\n'

    return head

GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

def reload_javascript():
    global gradio_theme
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response

    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )

    gradio_theme = gr.themes.Default(**default_theme_args)



controlnet_preprocessor_filters_aliases = {
    'instructp2p': ['ip2p'],
    'segmentation': ['seg'],
    'normalmap': ['normal'],
    't2i-adapter': ['t2i_adapter', 't2iadapter', 't2ia'],
    'ip-adapter': ['ip_adapter', 'ipadapter'],
    'scribble/sketch': ['scribble', 'sketch'],
    'tile/blur': ['tile', 'blur']
}

controlnet_preprocessor_filters = {
    "All": "none",
    "Canny": "canny",
    "Depth": "depth_midas",
    "NormalMap": "normal_bae",
    "OpenPose": "openpose_full",
    "MLSD": "mlsd",
    "Lineart": "lineart_standard (from white bg & black line)",
    "SoftEdge": "softedge_pidinet",
    "Scribble/Sketch": "scribble_pidinet",
    "Segmentation": "seg_ofade20k",
    "Shuffle": "shuffle",
    "Tile/Blur": "tile_resample",
    "Inpaint": "inpaint_only",
    "InstructP2P": "none",
    "Reference": "reference_only",
    "Recolor": "recolor_luminance",
    "Revision": "revision_clipvision",
    "T2I-Adapter": "none",
    "IP-Adapter": "ip-adapter_clip_sd15",
}

controlnet_preprocessor_aliases = {
    "invert": "invert (from white bg & black line)",
    "lineart_standard": "lineart_standard (from white bg & black line)",
    "lineart": "lineart_realistic",
    "color": "t2ia_color_grid",
    "clip_vision": "t2ia_style_clipvision",
    "pidinet_sketch": "t2ia_sketch_pidi",
    "depth": "depth_midas",
    "normal_map": "normal_midas",
    "hed": "softedge_hed",
    "hed_safe": "softedge_hedsafe",
    "pidinet": "softedge_pidinet",
    "pidinet_safe": "softedge_pidisafe",
    "segmentation": "seg_ufade20k",
    "oneformer_coco": "seg_ofcoco",
    "oneformer_ade20k": "seg_ofade20k",
    "pidinet_scribble": "scribble_pidinet",
    "inpaint": "inpaint_global_harmonious",
}

controlnet_ui_preprocessor_keys = ['none', 'invert (from white bg & black line)', 'blur_gaussian', 'canny', 'depth_leres', 'depth_leres++', 'depth_midas', 'depth_zoe', 'dw_openpose_full', 'inpaint_global_harmonious', 'inpaint_only', 'inpaint_only+lama', 'ip-adapter_clip_sd15', 'ip-adapter_clip_sdxl', 'lineart_anime', 'lineart_anime_denoise', 'lineart_coarse', 'lineart_realistic', 'lineart_standard (from white bg & black line)', 'mediapipe_face', 'mlsd', 'normal_bae', 'normal_midas', 'openpose', 'openpose_face', 'openpose_faceonly', 'openpose_full', 'openpose_hand', 'recolor_intensity', 'recolor_luminance', 'reference_adain', 'reference_adain+attn', 'reference_only', 'revision_clipvision', 'revision_ignore_prompt', 'scribble_hed', 'scribble_pidinet', 'scribble_xdog', 'seg_ofade20k', 'seg_ofcoco', 'seg_ufade20k', 'shuffle', 'softedge_hed', 'softedge_hedsafe', 'softedge_pidinet', 'softedge_pidisafe', 't2ia_color_grid', 't2ia_sketch_pidi', 't2ia_style_clipvision', 'threshold', 'tile_colorfix', 'tile_colorfix+sharp', 'tile_resample']

controlnet_control_modes = ["Balanced", "My prompt is more important", "ControlNet is more important"]

sam_padding_mode_names = [
        "constant",
        "edge",
        "reflect",
        "mean",
        "median",
        "maximum",
        "minimum",
    ]

sam_sampler_names = [
    "DDIM",
    "Euler",
    "Euler a",
    "DPM2 Karras",
    "DPM2 a Karras",
]
