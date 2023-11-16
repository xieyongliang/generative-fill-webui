import gradio as gr
import utils
from utils import ResizeHandleRow, FormRow, FormHTML, FormGroup, InputAccordion, ToolButton, create_refresh_button
import traceback
from typing import Optional
import json
from pydantic import BaseModel, Field
import requests
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Tuple, List
import cv2
from starlette.middleware.gzip import GZipMiddleware

progress = {}

txt2img_interrupted = False
img2img_interrupted = False

def txt2img_summit(*args):
    print(args)
    
    id_task = args[0]

    prompt = args[1]
    negative_prompt = args[2]
    steps = args[3]
    sampler_name = args[4]
    batch_count = args[5]
    batch_size = args[6]
    cfg_scale = args[7]
    height = args[8]
    width = args[9]
    enable_hr = args[10]
    denoising_strength = args[11]
    hr_scale = args[12]
    hr_upscaler = args[13]
    hr_second_pass_steps = args[14]
    hr_resize_x = args[15]
    hr_resize_y = args[16]

    seed = args[17]
    refiner_checkpoint = args[18]
    refiner_switch_at = args[19]

    cn_enabled = args[20]
    cn_module = args[21]
    cn_model = args[22]
    cn_image = args[23]
    cn_resize_mode = args[24]
    cn_low_vram = args[25]
    cn_processor_res = args[26]
    cn_threshold_a = args[27]
    cn_threshold_b = args[28]
    cn_guidance_start = args[29]
    cn_guidance_end = args[30]
    cn_pixel_perfect = args[31]
    cn_control_mode = args[32]

    global progress
    progress[id_task] = 0

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "styles": [],
        "steps": steps,
        "sampler_name": sampler_name,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "cfg_scale": cfg_scale,
        "height": height,
        "width": width,
        "enable_hr": enable_hr,
        "denoising_strength": denoising_strength,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "hr_second_pass_steps": hr_second_pass_steps,
        "hr_resize_x": hr_resize_x,
        "hr_resize_y": hr_resize_y,
        "seed": seed,
        "refiner_checkpoint": refiner_checkpoint,
        "refiner_switch_at": refiner_switch_at,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": cn_enabled,
                        "module": cn_module,
                        "model": cn_model,
                        "image": utils.encode_image_to_base64(cn_image) if cn_image else cn_image,
                        "resize_mode": cn_resize_mode,
                        "low_vram": cn_low_vram,
                        "processor_res": cn_processor_res,
                        "threshold_a": cn_threshold_a,
                        "threshold_b": cn_threshold_b,
                        "guidance_start": cn_guidance_start,
                        "guidance_end": cn_guidance_end,
                        "pixel_perfect": cn_pixel_perfect,
                        "control_mode": cn_control_mode
                    }
                ]
            }
        }
    }

    try:
        inputs = {'task': 'text-to-image', 'txt2img_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/sdapi/v1/txt2img', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            if txt2img_interrupted:
                return gr.update(), gr.update()
            else:
                images = []
                payload = json.loads(text)
                for image in payload['images']:
                    image = utils.decode_base64_to_image(image)
                    images.append(image)
                progress[id_task] = 1
                return gr.update(value=images), gr.update(value=utils.plaintext_to_html(json.loads(payload['info'])['infotexts'][0]))
        else:
            print(text)
            return gr.update(), gr.update()            
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update(), gr.update()

def img2img_summit(*args):
    print(args)

    id_task = args[0]
    mode = args[1]

    prompt = args[2]
    negative_prompt = args[3]
    init_img = args[4]
    sketch = args[5]
    init_img_with_mask = args[6]
    inpaint_color_sketch = args[7]
    inpaint_color_sketch_orig = args[8]
    init_img_inpaint = args[9]
    init_mask_inpaint = args[10]
    steps = args[11]
    sampler_name = args[12]
    mask_blur = args[13]
    mask_alpha = args[14]
    inpainting_fill = args[15]
    batch_count = args[16]
    batch_size = args[17]
    cfg_scale = args[18]
    image_cfg_scale = args[19]
    denoising_strength = args[20]
    selected_scale_tab = args[21]
    height = args[22]
    width = args[23]
    scale_by = args[24]
    resize_mode = args[25]
    inpaint_full_res = args[26]
    inpaint_full_res_padding = args[27]
    inpainting_mask_invert = args[28]

    seed = args[29]
    refiner_checkpoint = args[30] 
    refiner_switch_at = args[31]

    cn_enabled = args[33]
    cn_module = args[33]
    cn_model = args[34]
    cn_image = args[35]
    cn_resize_mode = args[36]
    cn_low_vram = args[37]
    cn_processor_res = args[38]
    cn_threshold_a = args[39]
    cn_threshold_b = args[40]
    cn_guidance_start = args[41]
    cn_guidance_end = args[42]
    cn_pixel_perfect = args[43]
    cn_control_mode = args[44]

    global progress
    progress[id_task] = 0

    is_batch = mode == 5

    if mode == 0:  # img2img
        image = init_img
        mask = None
    elif mode == 1:  # img2img sketch
        image = sketch
        mask = None
    elif mode == 2:  # inpaint
        image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
        mask = utils.create_binary_mask(mask)
    elif mode == 3:  # inpaint sketch
        image = inpaint_color_sketch
        orig = inpaint_color_sketch_orig or inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
        blur = ImageFilter.GaussianBlur(mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
    elif mode == 4:  # inpaint upload mask
        image = init_img_inpaint
        mask = init_mask_inpaint
    else:
        image = None
        mask = None

    # Use the EXIF orientation of photos taken by smartphones.
    if image is not None:
        image = ImageOps.exif_transpose(image)

    if selected_scale_tab == 1 and not is_batch:
        assert image, "Can't scale by because no image is selected"

        width = int(image.width * scale_by)
        height = int(image.height * scale_by)

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "styles": [],
        "steps": steps,
        "sampler_name": sampler_name,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "cfg_scale": cfg_scale,
        "height": height,
        "width": width,
        "init_images": [utils.encode_image_to_base64(image)],
        "mask": utils.encode_image_to_base64(mask) if mask else mask,
        "mask_blur": mask_blur,
        "inpainting_fill": inpainting_fill,
        "resize_mode": resize_mode,
        "denoising_strength": denoising_strength,
        "image_cfg_scale": image_cfg_scale,
        "inpaint_full_res": inpaint_full_res,
        "inpaint_full_res_padding": inpaint_full_res_padding,
        "inpainting_mask_invert": inpainting_mask_invert,
        "seed": seed,
        "refiner_checkpoint": refiner_checkpoint,
        "refiner_switch_at": refiner_switch_at,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": cn_enabled,
                        "module": cn_module,
                        "model": cn_model,
                        "image": utils.encode_image_to_base64(cn_image) if cn_image else cn_image,
                        "resize_mode": cn_resize_mode,
                        "low_vram": cn_low_vram,
                        "processor_res": cn_processor_res,
                        "threshold_a": cn_threshold_a,
                        "threshold_b": cn_threshold_b,
                        "guidance_start": cn_guidance_start,
                        "guidance_end": cn_guidance_end,
                        "pixel_perfect": cn_pixel_perfect,
                        "control_mode": cn_control_mode
                    }
                ]
            }
        }
    }

    try:
        inputs = {'task': 'image-to-image', 'img2img_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/sdapi/v1/img2img', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            if img2img_interrupted:
                return gr.update(), gr.update()
            else:
                images = []
                payload = json.loads(text)
                for image in payload['images']:
                    image = utils.decode_base64_to_image(image)
                    images.append(image)
                progress[id_task] = 1
                return gr.update(value=images), gr.update(value=utils.plaintext_to_html(json.loads(payload['info'])['infotexts'][0]))
        else:
            print(text)
            return gr.update(), gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update(), gr.update()

def upscale_submit(*args):
    upscale_mode = args[0]
    image = args[1]
    imageList = args[2]
    resize_mode = args[3]
    upscaling_resize = args[4]
    upscaling_resize_w = args[5]
    upscaling_resize_h = args[6]
    upscaling_crop = args[7]
    extras_upscaler_1 = args[8]
    extras_upscaler_2 = args[9]
    extras_upscaler_2_visibility = args[10]
    gfpgan_visibility = args[11]
    codeformer_visibility = args[12]
    codeformer_weight = args[13]

    try:
        payload = {
            "upscale_mode": resize_mode,
            "upscaling_resize": upscaling_resize,
            "upscaling_resize_w": upscaling_resize_w,
            "upscaling_resize_h": upscaling_resize_h,
            "upscaling_crop": upscaling_crop,
            "extras_upscaler_1": extras_upscaler_1,
            "extras_upscaler_2": extras_upscaler_2,
            "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
            "gfpgan_visibility": gfpgan_visibility,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight
        }
        
        if upscale_mode == 0:
            payload["image"] = utils.encode_image_to_base64(image)
            inputs = {'task':'extras-single-image', 'extras_single_payload': payload}
            if utils.sagemaker_endpoint:
                response = utils.invoke_async_inference(inputs)
            elif utils.use_webui:
                response = requests.post(url=f'{utils.api_endpoint}/sdapi/v1/extra-single-image', json=payload)
            else:
                response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

            status_code, text = utils.handle_response(response)
            if status_code == 200:
                images = []
                payload = json.loads(text)
                image = payload['image']
                image = utils.decode_base64_to_image(image)
                images.append(image)
                return gr.update(value=images), gr.update(value=payload['html_info'])
            else:
                print(text)
                return gr.update(), gr.update()
        elif upscale_mode == 1:
            payload["imageList"] = [utils.encode_image_to_base64(x) for x in imageList]
            inputs = {'task': 'extras-batch-images', 'extras_batch_payload': payload}
            if utils.sagemaker_endpoint:
                response = utils.invoke_async_inference(inputs)
            elif utils.use_webui:
                response = requests.post(url=f'{utils.api_endpoint}/sdapi/v1/extra-batch-images', json=payload)
            else:
                response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

            status_code, text = utils.handle_response(response)
            if status_code == 200:
                images = []
                payload = json.loads(text)
                for image in payload['images']:
                    image = utils.decode_base64_to_image(image)
                    images.append(image)
                return gr.update(value=images), gr.update(value=payload['html_info'])
            else:
                print(text)
                return gr.update(), gr.update()
        else:
            return gr.update(), gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update(), gr.update()

def run_padding(input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode="edge"):
    global sam_orig_image, sam_pad_mask

    payload = {
        "input_image": utils.encode_image_to_base64(input_image),
        "orig_image": utils.encode_image_to_base64(sam_orig_image),
        "pad_scale_width": pad_scale_width,
        "pad_scale_height": pad_scale_height,
        "pad_lr_barance": pad_lr_barance,
        "pad_tb_barance": pad_tb_barance,
        "padding_mode": padding_mode
    }

    try:
        inputs = {'task': '/inpaint-anything/padding', 'extra_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/inpaint-anything/padding', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            payload = json.loads(text)
            pad_image = utils.decode_base64_to_image(payload['pad_image'])
            sam_pad_mask = payload['pad_mask']
            return gr.update(value=pad_image)
        else:
            print(text)
            return gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update()

def run_sam(input_image, anime_style_chk=False):
    global sam_masks, sam_pad_mask, sam_model_id

    payload = {
        "input_image": utils.encode_image_to_base64(input_image),
        "sam_model_id": sam_model_id,
        "anime_style_chk": anime_style_chk,
        "pad_mask": sam_pad_mask
    }

    try:
        inputs = {'task': '/inpaint-anything/sam', 'extra_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/inpaint-anything/sam', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            payload = json.loads(text)
            sam_image = utils.decode_base64_to_image(payload['seg_image'])

            sam_masks = payload['sam_masks']

            return gr.update(value=sam_image)
        else:
            print(text)
            return gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update()

def select_mask(input_image, sam_image, invert_chk, ignore_black_chk, sel_mask):
    global sam_mask_image

    payload = {
        "input_image": utils.encode_image_to_base64(input_image),
        "sam_image": {
            "image": utils.encode_image_to_base64(sam_image["image"]),
            "mask": utils.encode_image_to_base64(sam_image["mask"])
        },
        "invert_chk": invert_chk,
        "ignore_black_chk": ignore_black_chk,
        "sam_masks": sam_masks
    }

    try:
        inputs = {'task': '/inpaint-anything/mask', 'extra_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/inpaint-anything/mask', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            payload = json.loads(text)
            ret_image = utils.decode_base64_to_image(payload['sel_mask'])
            sam_mask_image = utils.decode_base64_to_image(payload['mask_image'])
            if sel_mask is None:
                return gr.update(value=ret_image)
            else:
                if np.array(sel_mask["image"]).shape == np.array(ret_image).shape and np.all(np.array(sel_mask["image"]) == np.array(ret_image)):
                    return gr.update()
                else:
                    return gr.update(value=ret_image)
        else:
            print(text)
            return gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update()

def expand_mask(input_image, sel_mask, expand_mask_iteration_count):
    global sam_mask_image

    payload = {
        "input_image": utils.encode_image_to_base64(input_image),
        "mask_image": utils.encode_image_to_base64(sam_mask_image),
        "expand_mask_iteration_count": expand_mask_iteration_count
    }

    try:
        inputs = {'task': '/inpaint-anything/expand-mask', 'extra_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/inpaint-anything/expand-mask', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            payload = json.loads(text)
            sam_mask_image = utils.decode_base64_to_image(payload['mask_image'])
            ret_image = utils.decode_base64_to_image(payload['sel_mask'])

            if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == np.array(ret_image)):
                return gr.update()
            else:
                return gr.update(value=ret_image)
        else:
            print(text)
            return gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update()

def apply_mask(input_image, sel_mask):
    global sam_mask_image
    
    payload = {
        "input_image": utils.encode_image_to_base64(input_image),
        "sel_mask": {
            "image": utils.encode_image_to_base64(sel_mask["image"]),
            "mask": utils.encode_image_to_base64(sel_mask["mask"])
        },
        "mask_image": utils.encode_image_to_base64(sam_mask_image)
    }

    try:
        inputs = {'task': '/inpaint-anything/apply-mask', 'extra_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/inpaint-anything/apply-mask', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            payload = json.loads(text)
            sam_mask_image = utils.decode_base64_to_image(payload['mask_image'])
            ret_image = utils.decode_base64_to_image(payload['sel_mask'])

            if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == np.array(ret_image)):
                return gr.update()
            else:
                return gr.update(value=ret_image)
        else:
            print(text)
            return gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update()

def add_mask(input_image, sel_mask):
    global sam_mask_image
    
    payload = {
        "input_image": utils.encode_image_to_base64(input_image),
        "sel_mask": {
            "image": utils.encode_image_to_base64(sel_mask["image"]),
            "mask": utils.encode_image_to_base64(sel_mask["mask"])
        },
        "mask_image": utils.encode_image_to_base64(sam_mask_image)
    }

    try:
        inputs = {'task': '/inpaint-anything/add-mask', 'extra_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/inpaint-anything/add-mask', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            payload = json.loads(text)
            sam_mask_image = utils.decode_base64_to_image(payload['mask_image'])
            ret_image = utils.decode_base64_to_image(payload['sel_mask'])

            if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == np.array(ret_image)):
                return gr.update()
            else:
                return gr.update(value=ret_image)
        else:
            print(text)
            return gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update()

def run_cn_inpaint(input_image, sel_mask,
                   cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                   cn_module_id, cn_model_id, 
                   cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count=1,
                   cn_ref_module_id=None, cn_ref_image=None, cn_ref_weight=1.0, cn_ref_mode="Balanced", cn_ref_resize_mode="resize",
                   cn_ipa_or_ref=None, cn_ipa_model_id=None):
    global sam_mask_image

    if sel_mask is None:
        return gr.update(), gr.update()

    payload = {
        "input_image": utils.encode_image_to_base64(input_image),
        "mask_image": utils.encode_image_to_base64(sam_mask_image),
        "cn_prompt": cn_prompt,
        "cn_n_prompt": cn_n_prompt,
        "cn_sampler_id": cn_sampler_id,
        "cn_ddim_steps": cn_ddim_steps,
        "cn_cfg_scale": cn_cfg_scale,
        "cn_strength": cn_strength,
        "cn_seed": cn_seed,
        "cn_module_id": cn_module_id,
        "cn_model_id": cn_model_id,
        "cn_low_vram_chk": cn_low_vram_chk,
        "cn_weight": cn_weight,
        "cn_mode": cn_mode,
        "cn_iteration_count": cn_iteration_count,
        "cn_ref_module_id": cn_ref_module_id,
        "cn_ref_image": cn_ref_image,
        "cn_ref_weight": cn_ref_weight,
        "cn_ref_mode": cn_ref_mode,
        "cn_ref_resize_mode": cn_ref_resize_mode,
        "cn_ipa_or_ref": cn_ipa_or_ref,
        "cn_ipa_model_id": cn_ipa_model_id
    }

    try:
        inputs = {'task': '/inpaint-anything/cninpaint', 'extra_payload': payload}
        if utils.sagemaker_endpoint:
            response = utils.invoke_async_inference(inputs)
        elif utils.use_webui:
            response = requests.post(url=f'{utils.api_endpoint}/inpaint-anything/cninpaint', json=payload)
        else:
            response = requests.post(url=f'{utils.api_endpoint}/invocations', json=inputs)

        status_code, text = utils.handle_response(response)
        if status_code == 200:
            payload = json.loads(text)
            iteration_count = payload['iteration_count']
            images = []
            for output_image in payload['output_images']:
                images.append(utils.decode_base64_to_image(output_image))
            return gr.update(value=images), gr.update(value=iteration_count)
        else:
            print(text)
            return gr.update(), gr.update()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.update()

def run_get_alpha_image(input_image, sel_mask):
    global sam_mask_image
    if input_image is None or sam_mask_image is None or sel_mask is None:
        print("The image or mask does not exist")
        return None

    mask_image = np.array(json.loads(sam_mask_image))
    
    if input_image.shape != mask_image.shape:
        print("The sizes of the image and mask do not match")
        return None

    alpha_image = input_image.convert("RGBA")
    mask_image = Image.fromarray(mask_image).convert("L")

    alpha_image.putalpha(mask_image)

    return alpha_image

def run_get_mask(sel_mask):
    global sam_mask_image
    if sam_mask_image is None or sel_mask is None:
        return None

    mask_image = np.array(json.loads(sam_mask_image))

    return mask_image

class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    def __init__(self, is_img2img):
        id_part = "img2img" if is_img2img else "txt2img"
        self.id_part = id_part

        with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
            with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])
                            self.prompt_img = gr.File(label="", elem_id=f"{id_part}_prompt_image", file_count="single", type="binary", visible=False)

                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])

            self.button_interrogate = None
            self.button_deepbooru = None
            if is_img2img:
                with gr.Column(scale=1, elem_classes="interrogate-col"):
                    self.button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate", visible=False)
                    self.button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru", visible=False)

            with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
                with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                    self.interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", elem_classes="generate-box-interrupt")
                    self.submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                with gr.Row(elem_id=f"{id_part}_tools"):
                    self.clear_prompt_button = ToolButton(value=utils.clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
                    self.restore_progress_button = ToolButton(value=utils.restore_progress_symbol, elem_id=f"{id_part}_restore_progress", visible=False)

                    self.token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                    self.token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                    self.negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                    self.negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")

                    self.clear_prompt_button.click(
                        fn=lambda *x: x,
                        _js="confirm_clear_prompt",
                        inputs=[self.prompt, self.negative_prompt],
                        outputs=[self.prompt, self.negative_prompt],
                    )

class ControlnetUIGroup:
    refresh_symbol = "\U0001f504"  # 🔄
    switch_values_symbol = "\U000021C5"  # ⇅
    camera_symbol = "\U0001F4F7"  # 📷
    reverse_symbol = "\U000021C4"  # ⇄
    tossup_symbol = "\u2934"
    trigger_symbol = "\U0001F4A5"  # 💥
    open_symbol = "\U0001F4DD"  # 📝

    class ToolButton(gr.Button, gr.components.FormComponent):
        """Small button with single emoji as text, fits inside gradio forms"""

        def __init__(self, **kwargs):
            super().__init__(variant="tool", 
                            elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                            **kwargs)

        def get_block_name(self):
            return "button"

    def __init__(self, tabname, is_img2img, elem_id_tabname):
        with gr.Group(visible=not is_img2img) as image_upload_panel:
            with gr.Tabs():
                with gr.Tab(label="Single Image") as upload_tab:
                    with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                        with gr.Group(elem_classes=["cnet-input-image-group"]):
                            self.image = gr.Image(
                                source="upload",
                                brush_radius=20,
                                mirror_webcam=False,
                                type="pil",
                                tool="sketch",
                                elem_id=f"{elem_id_tabname}_{tabname}_input_image",
                                elem_classes=["cnet-image"],
                                brush_color="#ffffff"
                            )
                        with gr.Group(
                            visible=False, elem_classes=["cnet-generated-image-group"]
                        ) as generated_image_group:
                            self.generated_image = gr.Image(
                                value=None,
                                label="Preprocessor Preview",
                                elem_id=f"{elem_id_tabname}_{tabname}_generated_image",
                                elem_classes=["cnet-image"],
                                interactive=True,
                                height=242
                            )  # Gradio's magic number. Only 242 works.

                            with gr.Group(
                                elem_classes=["cnet-generated-image-control-group"]
                            ):
                                preview_check_elem_id = f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox"
                                preview_close_button_js = f"document.querySelector('#{preview_check_elem_id} input[type=\\'checkbox\\']').click();"
                                gr.HTML(
                                    value=f"""<a title="Close Preview" onclick="{preview_close_button_js}">Close</a>""",
                                    visible=True,
                                    elem_classes=["cnet-close-preview"],
                                )

                with gr.Tab(label="Batch", visible=False) as batch_tab:
                    self.batch_image_dir = gr.Textbox(
                        label="Input Directory",
                        placeholder="Leave empty to use img2img batch controlnet input directory",
                        elem_id=f"{elem_id_tabname}_{tabname}_batch_image_dir",
                    )

            with gr.Accordion(
                label="Open New Canvas", visible=False
            ) as create_canvas:
                self.canvas_width = gr.Slider(
                    label="New Canvas Width",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_width",
                )
                self.canvas_height = gr.Slider(
                    label="New Canvas Height",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_height",
                )
                with gr.Row():
                    self.canvas_create_button = gr.Button(
                        value="Create New Canvas",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_create_button",
                    )
                    self.canvas_cancel_button = gr.Button(
                        value="Cancel",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_cancel_button",
                    )

            with gr.Row(elem_classes="controlnet_image_controls"):
                gr.HTML(
                    value="<p>Set the preprocessor to [invert] If your image has white background and black lines.</p>",
                    elem_classes="controlnet_invert_warning",
                )
                open_new_canvas_button = self.ToolButton(
                    value=self.open_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_open_new_canvas_button",
                    interactive=False
                )
                webcam_enable = self.ToolButton(
                    value=self.camera_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_enable",
                    interactive=False
                )
                webcam_mirror = self.ToolButton(
                    value=self.reverse_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_mirror",
                    interactive=False
                )
                send_dimen_button = self.ToolButton(
                    value=self.tossup_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_send_dimen_button",
                    interactive=False
                )

        enabled: bool = False
        module: Optional[str] = None
        model: Optional[str] = None
        weight: float = 1.0
        low_vram: bool = False
        pixel_perfect: bool = False
        guidance_start: float = 0.0
        guidance_end: float = 1.0
        processor_res: int = -1
        threshold_a: float = -1
        threshold_b: float = -1    
        loopback: bool = False 

        with FormRow(elem_classes=["controlnet_main_options"]):
            self.enabled = gr.Checkbox(
                label="Enable",
                value=enabled,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_enable_checkbox",
                elem_classes=["cnet-unit-enabled"],
            )
            self.low_vram = gr.Checkbox(
                label="Low VRAM",
                value=low_vram,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_low_vram_checkbox",
            )
            self.pixel_perfect = gr.Checkbox(
                label="Pixel Perfect",
                value=pixel_perfect,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox",
            )
            self.preprocessor_preview = gr.Checkbox(
                label="Allow Preview",
                value=False,
                elem_id=preview_check_elem_id,
                visible=not is_img2img,
            )
            self.use_preview_as_input = gr.Checkbox(
                label="Preview as Input",
                value=False,
                elem_classes=["cnet-preview-as-input"],
                visible=False,
            )

        with gr.Row(elem_classes="controlnet_img2img_options"):
            if is_img2img:
                self.upload_independent_img_in_img2img = gr.Checkbox(
                    label="Upload independent control image",
                    value=False,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_same_img2img_checkbox",
                    elem_classes=["cnet-unit-same_img2img"],
                )
            else:
                self.upload_independent_img_in_img2img = None

        with gr.Row(elem_classes=["controlnet_control_type", "controlnet_row"]):
            self.type_filter = gr.Radio(
                list(utils.controlnet_preprocessor_filters.keys()),
                label=f"Control Type",
                value="All",
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_type_filter_radio",
                elem_classes="controlnet_control_type_filter_group",
            )

            def select_control_type(control_type: str) -> Tuple[List[str], List[str], str, str]:
                default_option = utils.controlnet_preprocessor_filters[control_type]
                pattern = control_type.lower()
                preprocessor_list = utils.controlnet_ui_preprocessor_keys
                model_list = utils.controlnet_models
                if pattern == "all":
                    return [
                        preprocessor_list,
                        model_list,
                        'none', #default option
                        "None"  #default model 
                    ]
                filtered_preprocessor_list = [
                    x
                    for x in preprocessor_list
                    if pattern in x.lower() or any(a in x.lower() for a in utils.controlnet_preprocessor_filters_aliases.get(pattern, [])) or x.lower() == "none"
                ]
                if pattern in ["canny", "lineart", "scribble/sketch", "mlsd"]:
                    filtered_preprocessor_list += [
                        x for x in preprocessor_list if "invert" in x.lower()
                    ]
                filtered_model_list = [
                    x for x in model_list if pattern in x.lower() or any(a in x.lower() for a in utils.controlnet_preprocessor_filters_aliases.get(pattern, [])) or x.lower() == "none"
                ]
                if default_option not in filtered_preprocessor_list:
                    default_option = filtered_preprocessor_list[0]
                if len(filtered_model_list) == 1:
                    default_model = "None"
                    filtered_model_list = model_list
                else:
                    default_model = filtered_model_list[1]
                    for x in filtered_model_list:
                        if "11" in x.split("[")[0]:
                            default_model = x
                            break
                
                return (
                    filtered_preprocessor_list,
                    filtered_model_list, 
                    default_option,
                    default_model
                )

            def filter_selected(k: str, pp):
                (
                    filtered_preprocessor_list,
                    filtered_model_list,
                    default_option,
                    default_model,
                ) = select_control_type(k)

                return [
                    gr.Dropdown.update(
                        value=default_option, choices=filtered_preprocessor_list
                    ),
                    gr.Dropdown.update(
                        value=default_model, choices=filtered_model_list
                    ),
                ]

        with gr.Row(elem_classes=["controlnet_preprocessor_model", "controlnet_row"]):
            self.module = gr.Dropdown(
                utils.controlnet_ui_preprocessor_keys,
                label=f"Preprocessor",
                value=module,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown",
            )
            trigger_preprocessor = ToolButton(
                value=self.trigger_symbol,
                visible=not is_img2img,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_trigger_preprocessor",
                elem_classes=["cnet-run-preprocessor"],
            )
            self.model = gr.Dropdown(
                utils.controlnet_models,
                label=f"Model",
                value=model,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_dropdown",
            )
            refresh_models = ToolButton(
                value=self.refresh_symbol,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_refresh_models",
            )
            def refresh_controlnet_models():
                utils.refresh_controlnet_models()
                return gr.update(choices=utils.controlnet_models)

            self.type_filter.change(
                filter_selected,
                inputs=[self.type_filter],
                outputs=[self.module, self.model],
                show_progress=False
            )

            refresh_models.click(
                fn=refresh_controlnet_models, 
                inputs=[], 
                outputs=[self.model], 
                show_progress=False
            )

        with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
            self.weight = gr.Slider(
                label=f"Control Weight",
                value=weight,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_weight_slider",
                elem_classes="controlnet_control_weight_slider",
            )
            self.guidance_start = gr.Slider(
                label="Starting Control Step",
                value=guidance_start,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_start_control_step_slider",
                elem_classes="controlnet_start_control_step_slider",
            )
            self.guidance_end = gr.Slider(
                label="Ending Control Step",
                value=guidance_end,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_ending_control_step_slider",
                elem_classes="controlnet_ending_control_step_slider",
            )

        # advanced options
        with gr.Column(visible=False) as advanced:
            self.processor_res = gr.Slider(
                label="Preprocessor resolution",
                value=processor_res,
                minimum=64,
                maximum=2048,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_resolution_slider",
            )
            self.threshold_a = gr.Slider(
                label="Threshold A",
                value=threshold_a,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_A_slider",
            )
            self.threshold_b = gr.Slider(
                label="Threshold B",
                value=threshold_b,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_B_slider",
            )

        self.control_mode = gr.Radio(
            choices=utils.controlnet_control_modes,
            value=utils.controlnet_control_modes[0],
            label="Control Mode",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_mode_radio",
            elem_classes="controlnet_control_mode_radio",
        )

        resize_modes = ["Just Resize", "Crop and Resize", "Resize and Fill"]
        self.resize_mode = gr.Radio(
            choices=resize_modes,
            value=resize_modes[0],
            label="Resize Mode",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_resize_mode_radio",
            elem_classes="controlnet_resize_mode_radio",
            visible=not is_img2img,
        )

        self.loopback = gr.Checkbox(
            label="[Loopback] Automatically send generated images to this ControlNet unit",
            value=loopback,
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_automatically_send_generated_images_checkbox",
            elem_classes="controlnet_loopback_checkbox",
            visible=not is_img2img,
        )

def create_top_row(is_img2img):
    return Toprow(is_img2img)

def create_controlnet_row(is_img2img):
    elem_id_tabname = ("img2img" if is_img2img else "txt2img") + "_controlnet"

    controlnet_ui_groups = []
    with gr.Group(elem_id=elem_id_tabname):
        with gr.Accordion(f"ControlNet 1.1.415", open = False, elem_id="controlnet"):
            if utils.max_controlnet_models > 1:
                with gr.Tabs(elem_id=f"{elem_id_tabname}_tabs"):
                    for i in range(utils.max_controlnet_models):
                        with gr.Tab(f"ControlNet Unit {i}", elem_classes=['cnet-unit-tab']):
                            controlnet_ui_groups.append(ControlnetUIGroup(f"ControlNet-{i}", is_img2img, elem_id_tabname))
            else:
                with gr.Column():
                    controlnet_ui_groups.append(ControlnetUIGroup(f"ControlNet", is_img2img, elem_id_tabname))
    return controlnet_ui_groups

sd_model_checkpoint_component = None

def create_sd_model_checkpoint_interface():
    global sd_model_checkpoint_component
    with gr.Blocks(analytics_enabled=False) as sd_model_checkpoint_interface:
        with gr.Row():
            with gr.Column(scale=1):
                sd_model_checkpoint_component = gr.Dropdown(label='Stable-Diffusion Checkpoint', elem_id="sd_checkpoint", choices=utils.sd_models, value=utils.sd_models[0])
            with gr.Column(scale=3):
                gr.HTML("")
    return sd_model_checkpoint_interface

def create_txt2img_interface():
    global sd_model_checkpoint_component

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        dummy_component = gr.Label(visible=False)
        txt2img_top_row = create_top_row(False)
        with gr.Tab("Generation", id="txt2img_generation") as txt2img_generation_tab, ResizeHandleRow(equal_height=False):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):
                with FormRow(elem_id=f"sampler_selection_txt2img"):
                    sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"txt2img_sampling", choices=utils.sd_samplers, value=utils.sd_samplers[0])
                    steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"txt2img_steps", label="Sampling steps", value=20)

                with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
                    with InputAccordion(False, label="Hires. fix", elem_id="txt2img_hr") as enable_hr:
                        with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                            hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=utils.latent_upscale_modes, value=utils.latent_upscale_modes[0])
                            hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                        with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                            hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                            hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                    with InputAccordion(False, label="Refiner", elem_id="enable") as enable_refiner:
                        with gr.Row():
                            refiner_checkpoint = gr.Dropdown(label='Checkpoint', elem_id="txt2img_checkpoint", choices=utils.sd_models, value='', tooltip="switch to another model in the middle of generation")
                            create_refresh_button(refiner_checkpoint, utils.refresh_sd_models, lambda: {"choices": utils.sd_models}, "txt2img_checkpoint_refresh")

                            refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01, elem_id="txt2img_switch_at", tooltip="fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation")

                with FormRow():
                    with gr.Column(elem_id="txt2img_column_size", scale=4):
                        width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="txt2img_width")
                        height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="txt2img_height")

                    with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                        res_switch_btn = ToolButton(value=utils.switch_values_symbol, elem_id="txt2img_res_switch_btn", label="Switch dims")

                    with gr.Column(elem_id="txt2img_column_batch"):
                        batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                with gr.Row():
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="txt2img_cfg_scale")

                with gr.Row():
                    seed = gr.Textbox(label='Seed', value="-1", elem_id="txt2img_seed", min_width=100)
                    random_seed = ToolButton(utils.random_symbol, elem_id="txt2img_random_seed", label='Random seed')
                    reuse_seed = ToolButton(utils.reuse_symbol, elem_id="txt2img_reuse_seed", label='Reuse seed')

                with gr.Row():
                    txt2img_controlnet_row = create_controlnet_row(False)

            with gr.Column(variant='panel', elem_id=f"txt2img_results"):
                with gr.Group(elem_id=f"txt2img_gallery_container"):
                    txt2img_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"txt2img_gallery", columns=4, preview=True, height=None)
                    txt2img_html_info = gr.HTML(elem_id=f'html_info_txt2img', elem_classes="infotext")

                generation_info = None
                with gr.Column():
                    with gr.Group():
                        html_info = gr.HTML(elem_id=f'html_info_txt2img', elem_classes="infotext")

                        generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_txt2img')
                        generation_info_button = gr.Button(visible=False, elem_id=f"txt2img_generation_info_button")
                        generation_info_button.click(
                            fn=utils.update_generation_info,
                            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                            inputs=[generation_info, html_info, html_info],
                            outputs=[html_info, html_info],
                            show_progress=False,
                        )

        def interrupt():
            global txt2img_interrupted
            txt2img_interrupted = True

        txt2img_top_row.interrupt.click(
            fn=lambda: interrupt(),
            inputs=[],
            outputs=[],
        )

        txt2img_controlnet_inputs = []
        for controlnet_ui_group in txt2img_controlnet_row:
            txt2img_controlnet_inputs += [
                controlnet_ui_group.enabled,
                controlnet_ui_group.module,
                controlnet_ui_group.model,
                controlnet_ui_group.image,
                controlnet_ui_group.resize_mode,
                controlnet_ui_group.low_vram,
                controlnet_ui_group.processor_res,
                controlnet_ui_group.threshold_a,
                controlnet_ui_group.threshold_b,
                controlnet_ui_group.guidance_start,
                controlnet_ui_group.guidance_end,
                controlnet_ui_group.pixel_perfect,
                controlnet_ui_group.control_mode
            ]

        txt2img_args = dict(
            fn=txt2img_summit,
            _js="txt2img_submit",
            inputs=[
                dummy_component,
                txt2img_top_row.prompt, 
                txt2img_top_row.negative_prompt,
                steps,
                sampler_name,
                batch_count,
                batch_size,
                cfg_scale,
                height,
                width,
                enable_hr,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                seed,
                refiner_checkpoint, 
                refiner_switch_at
            ] + txt2img_controlnet_inputs + [sd_model_checkpoint_component],
            outputs=[
                txt2img_gallery,
                txt2img_html_info
            ],
            show_progress=False,
        )

        txt2img_top_row.prompt.submit(**txt2img_args)
        txt2img_top_row.submit.click(**txt2img_args)

        res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)
        random_seed.click(fn=None, _js="function(){setRandomSeed('" + "txt2img_seed" + "')}", show_progress=False, inputs=[], outputs=[])

        def copy_seed(gen_info_string: str, index):
            res = -1

            try:
                gen_info = json.loads(gen_info_string)
                index -= gen_info.get('index_of_first_image', 0)

                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

            except json.decoder.JSONDecodeError:
                if gen_info_string:
                    print(f"Error parsing JSON generation info: {gen_info_string}")

            return [res, gr.update()]

        reuse_seed.click(
            fn=copy_seed,
            _js="(x, y) => [x, selected_gallery_index()]",
            show_progress=False,
            inputs=[generation_info, seed],
            outputs=[seed, seed]
        )

    return txt2img_interface

def create_img2img_interface():
    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        dummy_component = gr.Label(visible=False)
        dummy_component_2 = gr.Label(visible=False)
        img2img_top_row = create_top_row(is_img2img=True)
        with gr.Tab("Generation", id="img2img_generation") as img2img_generation_tab, ResizeHandleRow(equal_height=False):
            with gr.Column(variant='compact', elem_id="img2img_settings"):
                copy_image_buttons = []
                copy_image_destinations = {}

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}"):
                        gr.HTML("Copy image to: ", elem_id=f"img2img_label_copy_to_{tab_name}")

                        for title, name in zip(['img2img', 'sketch', 'inpaint', 'inpaint sketch'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue

                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                with gr.Tabs(elem_id="mode_img2img"):
                    img2img_selected_tab = gr.State(0)

                    with gr.TabItem('img2img', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA", height=720)
                        add_copy_image_controls('img2img', init_img)

                    with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                        sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=720, brush_color='#ffffff')
                        add_copy_image_controls('sketch', sketch)

                    with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=720, brush_color='#ffffff')
                        add_copy_image_controls('inpaint', init_img_with_mask)

                    with gr.TabItem('Inpaint sketch', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                        inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=720, brush_color='#ffffff')
                        inpaint_color_sketch_orig = gr.State(None)
                        add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                        def update_orig(image, state):
                            if image is not None:
                                same_size = state is not None and state.size == image.size
                                has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                edited = same_size and has_exact_match
                                return image if not edited or state is None else state

                        inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                    with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", image_mode="RGBA", elem_id="img_inpaint_mask")

                    img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload]

                    for i, tab in enumerate(img2img_tabs):
                        tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

                def copy_image(img):
                    if isinstance(img, dict) and 'image' in img:
                        return img['image']

                    return img

                for button, name, elem in copy_image_buttons:
                    button.click(
                        fn=copy_image,
                        inputs=[elem],
                        outputs=[copy_image_destinations[name]],
                    )
                    button.click(
                        fn=lambda: None,
                        _js=f"switch_to_{name.replace(' ', '_')}",
                        inputs=[],
                        outputs=[],
                    )

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"], type="index", value="Just resize")

                with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                    with FormRow():
                        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                        mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")

                    with FormRow():
                        inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")

                    with FormRow():
                        inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index", elem_id="img2img_inpainting_fill")

                    with FormRow():
                        with gr.Column():
                            inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")

                        with gr.Column(scale=4):
                            inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                    def select_img2img_tab(tab):
                        return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3),

                    for i, elem in enumerate(img2img_tabs):
                        elem.select(
                            fn=lambda tab=i: select_img2img_tab(tab),
                            inputs=[],
                            outputs=[inpaint_controls, mask_alpha],
                        )

                with FormRow():
                    with gr.Column(variant='compact', elem_id="img2img_settings"):
                        with FormRow(elem_id=f"sampler_selection_img2img"):
                            sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"img2img_sampling", choices=utils.sd_samplers, value=utils.sd_samplers[0])
                            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"img2img_steps", label="Sampling steps", value=20)

                        with FormRow(elem_id="img2img_accordions", elem_classes="accordions"):
                            with InputAccordion(False, label="Refiner", elem_id="enable") as enable_refiner:
                                with gr.Row():
                                    refiner_checkpoint = gr.Dropdown(label='Checkpoint', elem_id="img2img_checkpoint", choices=utils.sd_models, value='', tooltip="switch to another model in the middle of generation")
                                    create_refresh_button(refiner_checkpoint, utils.refresh_sd_models, lambda: {"choices": utils.sd_models}, "img2img_checkpoint_refresh")

                                    refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01, elem_id="img2img_switch_at", tooltip="fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation")

                        with FormRow():
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                selected_scale_tab = gr.State(value=0)

                                with gr.Tabs():
                                    with gr.Tab(label="Resize to", elem_id="img2img_tab_resize_to") as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="img2img_width")
                                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="img2img_height")
                                            with gr.Column(elem_id="img2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                                res_switch_btn = ToolButton(value=utils.switch_values_symbol, elem_id="img2img_res_switch_btn")
                                                detect_image_size_btn = ToolButton(value=utils.detect_image_size_symbol, elem_id="img2img_detect_image_size_btn")

                                    with gr.Tab(label="Resize by", elem_id="img2img_tab_resize_by") as tab_scale_by:
                                        scale_by = gr.Slider(minimum=0.05, maximum=4.0, step=0.05, label="Scale", value=1.0, elem_id="img2img_scale")

                                        with FormRow():
                                            scale_by_html = FormHTML(utils.resize_from_to_html(0, 0, 0.0), elem_id="img2img_scale_resolution_preview")
                                            gr.Slider(label="Unused", elem_id="img2img_unused_scale_by_slider")
                                            button_update_resize_to = gr.Button(visible=False, elem_id="img2img_update_resize_to")

                                    on_change_args = dict(
                                        fn=utils.resize_from_to_html,
                                        _js="currentImg2imgSourceResolution",
                                        inputs=[dummy_component, dummy_component, scale_by],
                                        outputs=scale_by_html,
                                        show_progress=False,
                                    )

                                    scale_by.release(**on_change_args)
                                    button_update_resize_to.click(**on_change_args)

                                    # the code below is meant to update the resolution label after the image in the image selection UI has changed.
                                    # as it is now the event keeps firing continuously for inpaint edits, which ruins the page with constant requests.
                                    # I assume this must be a gradio bug and for now we'll just do it for non-inpaint inputs.
                                    for component in [init_img, sketch]:
                                        component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress=False)

                            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
                            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])

                            with gr.Column(elem_id="img2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                            with gr.Column(elem_id="img2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=utils.switch_values_symbol, elem_id="img2img_res_switch_btn", label="Switch dims")

                            with gr.Column(elem_id="img2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="img2img_cfg_scale")
                            image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale', value=1.5, elem_id="img2img_image_cfg_scale", visible=False)

                        with gr.Row():
                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="img2img_denoising_strength")

                        with gr.Row():
                            seed = gr.Textbox(label='Seed', value="-1", elem_id="img2img_seed", min_width=100)
                            random_seed = ToolButton(utils.random_symbol, elem_id="img2img_random_seed", label='Random seed')
                            reuse_seed = ToolButton(utils.reuse_symbol, elem_id="img2img_reuse_seed", label='Reuse seed')

                        with gr.Row():
                            img2img_controlnet_row = create_controlnet_row(False)

            with gr.Column(variant='panel', elem_id=f"img2img_results"):
                with gr.Group(elem_id=f"img2img_gallery_container"):
                    img2img_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"img2img_gallery", columns=4, preview=True, height=None)
                    img2img_html_info = gr.HTML(elem_id=f'html_info_img2img', elem_classes="infotext")

                generation_info = None
                with gr.Column():
                    with gr.Group():
                        html_info = gr.HTML(elem_id=f'html_info_img2img', elem_classes="infotext")

                        generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_img2img')
                        generation_info_button = gr.Button(visible=False, elem_id=f"img2img_generation_info_button")
                        generation_info_button.click(
                            fn=utils.update_generation_info,
                            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                            inputs=[generation_info, html_info, html_info],
                            outputs=[html_info, html_info],
                            show_progress=False,
                        )

            def interrupt():
                global img2img_interrupted
                img2img_interrupted = True

            img2img_top_row.interrupt.click(
                fn=lambda: interrupt(),
                inputs=[],
                outputs=[],
            )


        img2img_controlnet_inputs = []
        for controlnet_ui_group in img2img_controlnet_row:
            img2img_controlnet_inputs += [
                controlnet_ui_group.enabled,
                controlnet_ui_group.module,
                controlnet_ui_group.model,
                controlnet_ui_group.image,
                controlnet_ui_group.resize_mode,
                controlnet_ui_group.low_vram,
                controlnet_ui_group.processor_res,
                controlnet_ui_group.threshold_a,
                controlnet_ui_group.threshold_b,
                controlnet_ui_group.guidance_start,
                controlnet_ui_group.guidance_end,
                controlnet_ui_group.pixel_perfect,
                controlnet_ui_group.control_mode
            ]

        img2img_args = dict(
            fn=img2img_summit,
            _js="img2img_submit",
            inputs=[
                dummy_component,
                dummy_component_2,
                img2img_top_row.prompt, 
                img2img_top_row.negative_prompt,
                init_img,
                sketch,
                init_img_with_mask,
                inpaint_color_sketch,
                inpaint_color_sketch_orig,
                init_img_inpaint,
                init_mask_inpaint,
                steps,
                sampler_name,
                mask_blur,
                mask_alpha,
                inpainting_fill,
                batch_count,
                batch_size,
                cfg_scale,
                image_cfg_scale,
                denoising_strength,
                selected_scale_tab,
                height,
                width,
                scale_by,
                resize_mode,
                inpaint_full_res,
                inpaint_full_res_padding,
                inpainting_mask_invert,
                seed,
                refiner_checkpoint, 
                refiner_switch_at
            ] + img2img_controlnet_inputs + [sd_model_checkpoint_component],
            outputs=[
                img2img_gallery,
                img2img_html_info
            ],
            show_progress=False,
        )

        img2img_top_row.prompt.submit(**img2img_args)
        img2img_top_row.submit.click(**img2img_args)

        res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)
        random_seed.click(fn=None, _js="function(){setRandomSeed('" + "txt2img_seed" + "')}", show_progress=False, inputs=[], outputs=[])

        def copy_seed(gen_info_string: str, index):
            res = -1

            try:
                gen_info = json.loads(gen_info_string)
                index -= gen_info.get('index_of_first_image', 0)

                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

            except json.decoder.JSONDecodeError:
                if gen_info_string:
                    print(f"Error parsing JSON generation info: {gen_info_string}")

            return [res, gr.update()]

        reuse_seed.click(
            fn=copy_seed,
            _js="(x, y) => [x, selected_gallery_index()]",
            show_progress=False,
            inputs=[generation_info, seed],
            outputs=[seed, seed]
        )

    return img2img_interface

def create_img_upscale_interface():
    with gr.Blocks(analytics_enabled=False) as upscale_interface:
        upscale_selected_tab = gr.State(value=0)

        with gr.Row(equal_height=False, variant='compact'):
            with gr.Column(variant='compact'):
                with gr.Tabs(elem_id="mode_extras"):
                    with gr.TabItem('Single Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                        single_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")

                    with gr.TabItem('Batch Process', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                        batch_image = gr.Files(label="Batch Process", interactive=True, elem_id="extras_image_batch")

                submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

                resize_selected_tab = gr.State(value=0)

                with FormRow():
                    with gr.Tabs(elem_id="extras_resize_mode"):
                        with gr.TabItem('Scale by', elem_id="extras_scale_by_tab") as tab_scale_by:
                            upscaling_resize = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Resize", value=4, elem_id="extras_upscaling_resize")

                        with gr.TabItem('Scale to', elem_id="extras_scale_to_tab") as tab_scale_to:
                            with FormRow():
                                with gr.Column(elem_id="upscaling_column_size", scale=4):
                                    upscaling_resize_w = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="extras_upscaling_resize_w")
                                    upscaling_resize_h = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="extras_upscaling_resize_h")
                                with gr.Column(elem_id="upscaling_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                    upscaling_res_switch_btn = ToolButton(value=utils.switch_values_symbol, elem_id="upscaling_res_switch_btn")
                                    upscaling_crop = gr.Checkbox(label='Crop to fit', value=True, elem_id="extras_upscaling_crop")

                with FormRow():
                    extras_upscaler_1 = gr.Dropdown(label='Upscaler 1', elem_id="extras_upscaler_1", choices=utils.sd_upscalers, value=utils.sd_upscalers[0])

                with FormRow():
                    extras_upscaler_2 = gr.Dropdown(label='Upscaler 2', elem_id="extras_upscaler_2", choices=utils.sd_upscalers, value=utils.sd_upscalers[0])
                    extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=0.0, elem_id="extras_upscaler_2_visibility")

                with FormRow():
                    gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN visibility", value=0, elem_id="extras_gfpgan_visibility")

                with FormRow():
                    codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer visibility", value=0, elem_id="extras_codeformer_visibility")
                    codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer weight (0 = maximum effect, 1 = minimum effect)", value=0, elem_id="extras_codeformer_weight")

            with gr.Column(variant='panel', elem_id=f"upscale_results"):
                with gr.Group(elem_id=f"upscale_gallery_container"):
                    upscale_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"upscale_gallery", columns=4, preview=True, height=None)
                    upscale_html_info = gr.HTML(elem_id=f'html_info_upscale', elem_classes="infotext")

                generation_info = None
                with gr.Column():
                    with gr.Group():
                        html_info = gr.HTML(elem_id=f'html_info_upscale', elem_classes="infotext")

                        generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_upscale')
                        generation_info_button = gr.Button(visible=False, elem_id=f"upscale_generation_info_button")
                        generation_info_button.click(
                            fn=utils.update_generation_info,
                            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                            inputs=[generation_info, html_info, html_info],
                            outputs=[html_info, html_info],
                            show_progress=False,
                        )

        upscaling_res_switch_btn.click(lambda w, h: (h, w), inputs=[upscaling_resize_w, upscaling_resize_h], outputs=[upscaling_resize_w, upscaling_resize_h], show_progress=False)
        tab_scale_by.select(fn=lambda: 0, inputs=[], outputs=[resize_selected_tab])
        tab_scale_to.select(fn=lambda: 1, inputs=[], outputs=[resize_selected_tab])

        tab_single.select(fn=lambda: 0, inputs=[], outputs=[upscale_selected_tab])
        tab_batch.select(fn=lambda: 1, inputs=[], outputs=[upscale_selected_tab])

        submit.click(
            fn=upscale_submit,
            inputs=[
                upscale_selected_tab,
                single_image,
                batch_image,
                resize_selected_tab,
                upscaling_resize,
                upscaling_resize_w,
                upscaling_resize_h,
                upscaling_crop,
                extras_upscaler_1,
                extras_upscaler_2,
                extras_upscaler_2_visibility,
                gfpgan_visibility,
                codeformer_visibility,
                codeformer_weight,              
            ],
            outputs=[
                upscale_gallery,
                upscale_html_info,
            ]
        )

    return upscale_interface

sam_orig_image = None
sam_pad_mask = None
sam_mask_image = None
sam_masks = None
sam_model_id = 'sam_vit_l_0b3195.pth'

def create_segment_anything_inferface():
    out_gallery_kwargs = dict(columns=2, height=520, object_fit="contain", preview=True)
    cn_ref_only = False
    cn_ip_adapter = False
    cn_ipa_model_ids = []
    cn_sampler_index = utils.sd_samplers.index("DDIM") if "DDIM" in utils.sd_samplers else 0
    cn_ref_module_ids = []
    cn_module_ids = utils.controlnet_ui_preprocessor_keys
    cn_module_index = cn_module_ids.index("inpaint_only") if "inpaint_only" in cn_module_ids else 0
    cn_model_ids = utils.controlnet_models[1 : len(utils.controlnet_models)]
    cn_modes = utils.controlnet_control_modes
    cn_sampler_ids = utils.sd_samplers
    padding_mode_names = utils.sam_padding_mode_names
    global sam_orig_image, sam_pad_mask, sam_mask_image, sam_masks, sam_model_id

    with gr.Blocks(analytics_enabled=False) as segment_anything_interface:
        with gr.Accordion("Step 1. Upload your image and create image segmentation using segment-anything", elem_id="step1", open=True):
            with gr.Row():
                    input_image = gr.Image(label="Input image", elem_id="ia_input_image", source="upload", type="pil", interactive=True, height=480)

            with gr.Row():
                with gr.Column(scale=2):
                    anime_style_chk = gr.Checkbox(label="Anime Style (Up Detection, Down mask Quality)", elem_id="anime_style_chk",
                                                show_label=True, interactive=True)
                with gr.Column(scale=4):
                    with gr.Accordion("Padding options", elem_id="padding_options", open=False):
                        with gr.Row():
                            with gr.Column():
                                pad_scale_width = gr.Slider(label="Scale Width", elem_id="pad_scale_width", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                            with gr.Column():
                                pad_scale_height = gr.Slider(label="Scale Height", elem_id="pad_scale_height", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                pad_lr_barance = gr.Slider(label="Left/Right Balance", elem_id="pad_lr_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                            with gr.Column():
                                pad_tb_barance = gr.Slider(label="Top/Bottom Balance", elem_id="pad_tb_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                padding_mode = gr.Dropdown(label="Padding Mode", elem_id="padding_mode", choices=padding_mode_names, value="edge")
                            with gr.Column():
                                padding_btn = gr.Button("Run Padding", elem_id="padding_btn")

            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML('')
                with gr.Column(scale=1):
                    sam_btn = gr.Button("Run Segment Anything", elem_id="sam_btn", variant="primary", interactive=False)
        with gr.Accordion("Step 2. Create image mask based on selected image segmentation", elem_id="step2", open=False):
            with gr.Row():
                sam_image = gr.Image(label="Segment Anything image", elem_id="ia_sam_image", type="pil", tool="sketch", brush_radius=8,
                                            show_label=False, interactive=True, height=480)
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                                invert_chk = gr.Checkbox(label="Invert mask", elem_id="invert_chk", show_label=True, interactive=True)
                        with gr.Column():
                                ignore_black_chk = gr.Checkbox(label="Ignore black area", elem_id="ignore_black_chk", value=True, show_label=True, interactive=True)                    
                with gr.Column(scale=1):
                    select_btn = gr.Button("Create Mask", elem_id="select_btn", variant="primary")

        with gr.Accordion("Step 3. Preview image mask and adjust if needed", elem_id="step3", open=False):
            with gr.Row():
                sel_mask = gr.Image(label="Selected mask image", elem_id="ia_sel_mask", type="pil", tool="sketch", brush_radius=12,
                                    show_label=False, interactive=True, height=480)

            with gr.Row():
                expand_mask_iteration_count = gr.Slider(label="Expand Mask Iterations",
                                                        elem_id="expand_mask_iteration_count", minimum=1, maximum=100, value=1, step=1)
            with gr.Row():
                with gr.Column(scale=1):
                    expand_mask_btn = gr.Button("Expand mask region", elem_id="expand_mask_btn")
                with gr.Column(scale=1):
                    apply_mask_btn = gr.Button("Trim mask by sketch", elem_id="apply_mask_btn")
                with gr.Column(scale=1):
                    add_mask_btn = gr.Button("Add mask by sketch", elem_id="add_mask_btn")

        with gr.Accordion("Step 4. Inpaint or mask only", elem_id="step4", open=False):
            with gr.Tab("ControlNet Inpaint", elem_id="cn_inpaint_tab"):
                with gr.Row():
                    with gr.Column():
                        cn_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_cn_sd_prompt")
                        cn_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_cn_sd_n_prompt")
                    with gr.Column(scale=0, min_width=128):
                        gr.Markdown("Get prompt from:")
                        cn_get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="cn_get_txt2img_prompt_btn")
                        cn_get_img2img_prompt_btn = gr.Button("img2img", elem_id="cn_get_img2img_prompt_btn")
                with gr.Accordion("Advanced options", elem_id="cn_advanced_options", open=False):
                    with gr.Row():
                        with gr.Column():
                            cn_sampler_id = gr.Dropdown(label="Sampling method", elem_id="cn_sampler_id",
                                                        choices=cn_sampler_ids, value=cn_sampler_ids[cn_sampler_index], show_label=True)
                        with gr.Column():
                            cn_ddim_steps = gr.Slider(label="Sampling steps", elem_id="cn_ddim_steps", minimum=1, maximum=150, value=30, step=1)
                    cn_cfg_scale = gr.Slider(label="Guidance scale", elem_id="cn_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                    cn_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Denoising strength", value=0.75, elem_id="cn_strength")
                    cn_seed = gr.Slider(
                        label="Seed",
                        elem_id="cn_sd_seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        value=-1,
                    )
                with gr.Accordion("ControlNet options", elem_id="cn_cn_options", open=False):
                    with gr.Row():
                        with gr.Column():
                            cn_low_vram_chk = gr.Checkbox(label="Low VRAM", elem_id="cn_low_vram_chk", value=True, show_label=True, interactive=True)
                            cn_weight = gr.Slider(label="Control Weight", elem_id="cn_weight", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                        with gr.Column():
                            cn_mode = gr.Dropdown(label="Control Mode", elem_id="cn_mode", choices=cn_modes, value=cn_modes[-1], show_label=True)

                    if cn_ref_only:
                        with gr.Row():
                            with gr.Column():
                                cn_md_text = "Reference Control (enabled with image below)"
                                if not cn_ip_adapter:
                                    cn_md_text = cn_md_text + ("<br><span style='color: gray;'>"
                                                                "[IP-Adapter](https://huggingface.co/lllyasviel/sd_control_collection/tree/main) "
                                                                "is not available. Reference-Only is used.</span>")
                                gr.Markdown(cn_md_text)
                                if cn_ip_adapter:
                                    cn_ipa_or_ref = gr.Radio(label="IP-Adapter or Reference-Only", elem_id="cn_ipa_or_ref",
                                                                choices=["IP-Adapter", "Reference-Only"], value="IP-Adapter", show_label=False)
                                cn_ref_image = gr.Image(label="Reference Image", elem_id="cn_ref_image", source="upload", type="pil",
                                                        interactive=True)
                            with gr.Column():
                                cn_ref_resize_mode = gr.Radio(label="Reference Image Resize Mode", elem_id="cn_ref_resize_mode",
                                                                choices=["resize", "tile"], value="resize", show_label=True)
                                if cn_ip_adapter:
                                    cn_ipa_model_id = gr.Dropdown(label="IP-Adapter Model ID", elem_id="cn_ipa_model_id",
                                                                    choices=cn_ipa_model_ids, value=cn_ipa_model_ids[0], show_label=True)
                                cn_ref_module_id = gr.Dropdown(label="Reference Type for Reference-Only", elem_id="cn_ref_module_id",
                                                                choices=cn_ref_module_ids, value=cn_ref_module_ids[-1], show_label=True)
                                cn_ref_weight = gr.Slider(label="Reference Control Weight", elem_id="cn_ref_weight",
                                                            minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                cn_ref_mode = gr.Dropdown(label="Reference Control Mode", elem_id="cn_ref_mode",
                                                            choices=cn_modes, value=cn_modes[0], show_label=True)
                    else:
                        with gr.Row():
                            gr.Markdown("The Multi ControlNet setting is currently set to 1.<br>"
                                        "If you wish to use the Reference-Only Control, "
                                        "please adjust the Multi ControlNet setting to 2 or more and restart the Web UI.")

                with gr.Row():
                    with gr.Column():
                        cn_module_id = gr.Dropdown(label="ControlNet Preprocessor", elem_id="cn_module_id",
                                                    choices=cn_module_ids, value=cn_module_ids[cn_module_index], show_label=True)
                    with gr.Column():
                        cn_model_id = gr.Dropdown(label="ControlNet Model ID", elem_id="cn_model_id",
                                                    choices=cn_model_ids, value=cn_model_ids[0], show_label=True)
                    with gr.Column():
                        cn_iteration_count = gr.Slider(label="Iterations", elem_id="cn_iteration_count",
                                                        minimum=1, maximum=10, value=1, step=1)
                    with gr.Column():
                            cn_inpaint_btn = gr.Button("Run ControlNet Inpaint", elem_id="cn_inpaint_btn", variant="primary")

                with gr.Row():
                    cn_out_image = gr.Gallery(label="Inpainted image", elem_id="ia_cn_out_image", show_label=False,
                                                **out_gallery_kwargs)

            with gr.Tab("Mask only", elem_id="mask_only_tab", render=True):
                with gr.Row():
                    with gr.Column():
                        get_alpha_image_btn = gr.Button("Generate image mask in RGBA", elem_id="get_alpha_image_btn")
                    with gr.Column():
                        get_mask_btn = gr.Button("Generate image mask in RGB", elem_id="get_mask_btn")

                with gr.Row():
                    with gr.Column():
                        cn_alpha_out_image = gr.Image(label="Alpha channel image", elem_id="alpha_out_image", type="pil", image_mode="RGBA", interactive=False)
                    with gr.Column():
                        cn_mask_out_image = gr.Image(label="Mask image", elem_id="mask_out_image", type="pil", interactive=False)

                with gr.Row(visible=False):
                    with gr.Column():
                        cn_mask_send_to_inpaint_btn = gr.Button("Send to img2img inpaint", elem_id="mask_send_to_inpaint_btn")

            def input_image_upload(input_image, sam_image, sel_mask):
                global sam_orig_image, sam_pad_mask, sam_mask_image, sam_masks

                sam_orig_image= input_image
                sam_pad_mask = None

                if sam_mask_image:
                    if isinstance(sam_mask_image, Image.Image):
                        sam_mask_image = np.array(sam_mask_image)
                    if isinstance(sam_mask_image, str):
                        sam_mask_image = np.array(utils.decode_base64_to_image(sam_mask_image))
                input_image = np.array(input_image)

                if (sam_mask_image is None or sam_mask_image.shape != input_image.shape):
                    sam_mask_image = np.zeros_like(input_image, dtype=np.uint8)

                ret_sel_image = cv2.addWeighted(input_image, 0.5, sam_mask_image, 0.5, 0)

                if sam_image is None or not isinstance(sam_image, dict) or "image" not in sam_image:
                    sam_masks = None
                    ret_sam_image = Image.fromarray(np.zeros_like(input_image, dtype=np.uint8))
                elif sam_image["image"].shape == input_image.shape:
                    ret_sam_image = gr.update()
                else:
                    sam_masks = None
                    ret_sam_image = gr.update(value=Image.fromarray(np.zeros_like(input_image, dtype=np.uint8)))

                if sel_mask is None or not isinstance(sel_mask, dict) or "image" not in sel_mask:
                    ret_sel_mask = Image.fromarray(ret_sel_image)
                elif np.array(sel_mask["image"]).shape == ret_sel_image.shape and np.all(np.array(sel_mask["image"]) == ret_sel_image):
                    ret_sel_mask = gr.update()
                else:
                    ret_sel_mask = gr.update(value=Image.fromarray(ret_sel_image))

                sam_mask_image = Image.fromarray(sam_mask_image)

                return ret_sam_image, ret_sel_mask, gr.update(interactive=True)

            input_image.upload(input_image_upload, inputs=[input_image, sam_image, sel_mask], outputs=[sam_image, sel_mask, sam_btn]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_initSamSelMask")
            padding_btn.click(run_padding, inputs=[input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode],
                              outputs=[input_image])
            sam_btn.click(run_sam, inputs=[input_image, anime_style_chk], outputs=[sam_image]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSamMask")
            select_btn.click(select_mask, inputs=[input_image, sam_image, invert_chk, ignore_black_chk, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            expand_mask_btn.click(expand_mask, inputs=[input_image, sel_mask, expand_mask_iteration_count], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            apply_mask_btn.click(apply_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            add_mask_btn.click(add_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            cn_get_txt2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getTxt2imgPrompt")
            cn_get_img2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getImg2imgPrompt")

            cn_inputs = [input_image, sel_mask,
                            cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                            cn_module_id, cn_model_id, 
                            cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count]
            if cn_ref_only:
                cn_inputs.extend([cn_ref_module_id, cn_ref_image, cn_ref_weight, cn_ref_mode, cn_ref_resize_mode])
            if cn_ip_adapter:
                cn_inputs.extend([cn_ipa_or_ref, cn_ipa_model_id])
            cn_inpaint_btn.click(
                run_cn_inpaint,
                inputs=cn_inputs,
                outputs=[cn_out_image, cn_iteration_count])
            get_alpha_image_btn.click(
                run_get_alpha_image,
                inputs=[input_image, sel_mask],
                outputs=[cn_alpha_out_image])
            get_mask_btn.click(
                run_get_mask,
                inputs=[sel_mask],
                outputs=[cn_mask_out_image])
            cn_mask_send_to_inpaint_btn.click(
                fn=None,
                _js="inpaintAnything_sendToInpaint",
                inputs=None,
                outputs=None)

    return segment_anything_interface

def create_ui():
    utils.reload_javascript()

    sd_model_checkpoint_interface = create_sd_model_checkpoint_interface()
    txt2img_interface = create_txt2img_interface()
    img2img_interface = create_img2img_interface()
    img_upscale_interface = create_img_upscale_interface()
    segment_anything_interface = create_segment_anything_inferface()

    interfaces = [
        (txt2img_interface, "Text to image", "txt2img"),
        (img2img_interface, "Image to image", "img2img"),
        (img_upscale_interface, "Image upscale", "upscale"),
        (segment_anything_interface, "Segment anything", "segment"),
    ]

    with gr.Blocks(theme=utils.gradio_theme, analytics_enabled=False, title="Genertive Fill") as demo:
        with gr.Row():
            gr.Markdown("## Demo for Genertive Fill on Amazon SageMaker")

        sd_model_checkpoint_interface.render()
        
        with gr.Row():
            with gr.Tabs(elem_id="tabs") as tabs:
                for interface, label, ifid in interfaces:
                    with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                        interface.render()

    return demo

class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    is_img2img: bool = Field(default=False, title="Task Flag", description="flag for txt2img or img2img")

class ProgressResponse(BaseModel):
    interrupted: bool = Field(title="Whether the task is interrupted")
    completed: bool = Field(title="Whether the task has already finished")

def progress_api(req: ProgressRequest):
    global txt2img_interrupted, img2img_interrupted
    if not req.is_img2img and txt2img_interrupted:
        txt2img_interrupted = False
        return ProgressResponse(interrupted=True, completed=False)
    if req.is_img2img and img2img_interrupted:
        img2img_interrupted = False
        return ProgressResponse(interrupted=True, completed=False)
    
    return ProgressResponse(interrupted=False, completed=progress[req.id_task]==1 if req.id_task in progress else False )

demo = create_ui()

app, _, _ = demo.launch(share=True, prevent_thread_lock=True)
app.middleware_stack = None  # reset current middleware to allow modifying user provided list
app.add_middleware(GZipMiddleware, minimum_size=1000)


app.add_api_route('/internal/progress', progress_api, methods=["POST"], response_model=ProgressResponse)

try:
    while 1:
        pass
except KeyboardInterrupt:
    print('Caught KeyboardInterrupt, stopping...')
    demo.close()
