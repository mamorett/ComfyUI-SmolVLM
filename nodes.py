import os, sys
from PIL import Image
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import torchvision.transforms.functional as F
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForVision2Seq


import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

class Tagger:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['SmolVLM_v1.0'], {
                    "default": "SmolVLM_v1.0"
                }),
                "folder_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Path to your image folder"
                }),
                "caption_method": (['tags', 'flux', 'detailed', 'analyze', 'custom'], {
                    "default": "flux"
                }),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096})
        },
            "optional": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"forceInput": True}),
                "captions": ("STRING", {"forceInput": True}),
                "custom_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),                 
                "prefix_caption": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "suffix_caption": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "replace_tags": ("STRING", {
                    "multiline": True,
                    "default": "replace_tags eg:search1:replace1;search2:replace2"
                }),               
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT", )
    RETURN_NAMES = ("images", "filenames", "captions", "folder_path", "batch_size", )
    OUTPUT_IS_LIST = (True, True, True, False, False, )
    FUNCTION = "start_tag"
    #OUTPUT_NODE = True
    CATEGORY = "SmolVLM Tagger"
    def tag_image(self, image, caption_method, model, processor, device, max_new_tokens, custom_prompt):

        if caption_method == 'tags':
            intermediate_prompt = "Describe this image using tags."
        elif caption_method == 'flux':
            intermediate_prompt = "Describe this image in a way that could help flux generate it."
        elif caption_method == 'detailed':
            intermediate_prompt = "Do a detailed description of this image."
        elif caption_method == 'custom' and custom_prompt:
            intermediate_prompt = custom_prompt  # Use the custom user-defined prompt          
        else:
            intermediate_prompt = "Analyze this image."


        # Create input messages for single image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": intermediate_prompt}
                ]
            },
        ]
        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)           

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

        full_decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Use the delimiter to extract only the model's generated response
        delimiter = "Assistant: "
        if delimiter in full_decoded_output:
            response_text = full_decoded_output.split(delimiter, 1)[1].strip()
        else:
            # Fallback in case the delimiter is missing (unexpected edge case)
            response_text = "Error: Could not extract response."

        print (response_text)
        return response_text

    def start_tag(self, model, folder_path, caption_method, max_new_tokens, images=None, filenames=None, captions=None, prefix_caption="", suffix_caption="", replace_tags="", custom_prompt=""):
        file_names = []
        tag_contents = []
        pil_images = []
        tensor_images = []
        # attention = 'sdpa'
        # precision = 'fp16'

        device = mm.get_torch_device()
        # offload_device = mm.unet_offload_device()
        # dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Download model if it does not exist

        hg_model = 'HuggingFaceTB/SmolVLM-Instruct'
        model_name = hg_model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)
        if not os.path.exists(model_path):
            print(f"Downloading SmolVLM model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=hg_model,
                              local_dir=model_path,
                              local_dir_use_symlinks=False)


        resolution_scale = 4  # default value
        processor = AutoProcessor.from_pretrained(model_path, size={"longest_edge": resolution_scale*384})            
        model = AutoModelForVision2Seq.from_pretrained(model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).to(device)

        # Load the processor
        # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if images is None:
            for filename in os.listdir(folder_path):
                image_types = ['png', 'jpg', 'jpeg']
                if filename.split(".")[-1] in image_types:
                    img_path = os.path.join(folder_path, filename)
                    cap_filename = '.'.join(filename.split('.')[:-1]) + '.txt'
                    image = Image.open(img_path).convert("RGB")
                    pil_images.append(image)

                    tensor_image = F.to_tensor(image)
                    tensor_image = tensor_image[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    tensor_images.append(tensor_image)

                    file_names.append(cap_filename)
        else:
            if len(images) == 1:
                images = [images]

            to_pil = transforms.ToPILImage()
            max_digits = max(3, len(str(len(images))))

            for i, img in enumerate(images, start=1):
                if img.ndim == 4:
                    # Convert (N, H, W, C) to (N, C, H, W)
                    img = img.permute(0, 3, 1, 2).squeeze(0)

                if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                    pil_img = to_pil(img.cpu())
                    pil_images.append(pil_img)

                    tensor_img = F.to_tensor(pil_img)
                    tensor_img = tensor_img[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    tensor_images.append(tensor_img)

                if filenames is None:
                    cap_filename = f"{i:0{max_digits}d}.txt"
                    file_names.append(cap_filename)
                else:
                    file_names.append(filenames)

        pbar = ProgressBar(len(pil_images))


        for i, image in enumerate(pil_images):
            tags = self.tag_image(image, caption_method, model, processor, device, max_new_tokens, custom_prompt)
            if "eg:" not in replace_tags and ":" in replace_tags:
                if ";" not in replace_tags:
                    replace_pairs = [replace_tags]
                else:
                    replace_pairs = replace_tags.split(";")
                for pair in replace_pairs:
                    search, replace = pair.split(":")
                    tags = tags.replace(search, replace)
            tags = prefix_caption + tags + suffix_caption
            # when two tagger nodes and their captions are connected
            if captions is not None:
                tags = captions + tags

            tag_contents.append(tags)

            pbar.update(1)

        batch_size = len(tensor_images)

        return (tensor_images, file_names, tag_contents, folder_path, batch_size,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    @classmethod
    def IS_CHANGED(s, model, folder_path, caption_method, max_new_tokens, images=None, filenames=None, captions=None, prefix_caption="", suffix_caption="", replace_tags="", custom_prompt=""):
        return ''

class SaveTags:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filenames": ("STRING", {"forceInput": True}),
                "captions": ("STRING", {"forceInput": True}),
                "save_folder": ("STRING", {"default": "Your save directory"}),
                "filename_prefix": ("STRING", {"default": ""}),
                "mode": (['overwrite', 'append'], {
                    "default": "overwrite"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('captions',)
    OUTPUT_IS_LIST = (False, )
    FUNCTION = "save_tag"
    OUTPUT_NODE = True
    CATEGORY = "SmolVLM Tagger"

    def save_tag(self, filenames, captions, save_folder, filename_prefix, mode):

        wmode = 'w' if mode == 'overwrite' else 'a'
        with open(os.path.join(save_folder, filename_prefix + filenames), wmode) as f:
            f.write(captions)

        print("Captions Saved")

        return (captions,)

class FluxCLIPTextEncode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "caption": ("STRING", {"forceInput": True, "dynamicPrompts": True}),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING")
    FUNCTION = "encode"

    CATEGORY = "SmolVLM Tagger"
    RETURN_NAMES = ("CONDITIONING", "EMPTY CONDITIONING", "t5xxl", "clip_l", "analyze")

    def encode(self, clip, caption, guidance):
        caption_segs = []

        for caption_seg in caption.split('\n'):
            if len(caption_seg) > 10:
                caption_segs.append(caption_seg.strip())

        t5xxl = caption_segs[0]
        if len(caption_segs) > 1:
            clip_l = caption_segs[1].replace('\\','').replace('(','').replace(')','').strip()
        else:
            clip_l = ""

        if len(caption_segs) > 2:
            analyze = caption_segs[2].replace('\\','').replace('(','').replace(')','').strip()
        else:
            analyze = ""

        tokens = clip.tokenize(f"{clip_l}\n\n{analyze}")
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        empty_tokens = clip.tokenize("")
        empty_tokens["t5xxl"] = clip.tokenize("")["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        empty_output = clip.encode_from_tokens(empty_tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        empty_cond = empty_output.pop("cond")
        output["guidance"] = guidance

        return ([[cond, output]], [[empty_cond, empty_output]], t5xxl, clip_l, analyze,)


class CaptionAnalyzer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        key_list = ["camera_angle", "art_style", "location", "text", "distance_to_camera", "background",
                    "position_in_image", "gender", "age",
                    "hair_style", "hair_color", "facial_expression", "eye_direction", "facing_direction", "race", "ear",
                    "expression", "body", "accessory", "pants", "clothing", "shoes", "action"]
        selection = {
                        "analyze": ("STRING", {"forceInput": True, "dynamicPrompts": True}),
                        "subject_index": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                     }
        for key in key_list:
            selection[key] = ("BOOLEAN", {"default": False})
        return {
            "required": selection
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "analyze"

    CATEGORY = "SmolVLM Tagger"
    RETURN_NAMES = ("selected analyze", )

    def analyze(self, analyze, subject_index, **kwargs):
        selected_analyze = []
        analyze_dict = {}
        previous_key = analyze.split(",")[0]

        for item in analyze.split(","):
            try:
                print(item)
                if ":" not in item:
                    analyze_dict[previous_key.strip()] = f'{analyze_dict[previous_key.strip()]}, {item.strip()}'
                else:
                    key, value = item.split(":")
                    analyze_dict[key.strip()] = value.strip()
                    previous_key = key
            except Exception as e:
                print(e)
                continue

        # Iterate through kwargs and check if the flag is set to True1
        print(analyze_dict)
        print(kwargs.items())
        for key, flag in kwargs.items():
            if flag and key in analyze_dict:
                if subject_index == 0 or len(analyze_dict[key].split(";")) < subject_index:
                    analyze_result  = analyze_dict[key]
                else:
                    analyze_result = analyze_dict[key].split(";")[subject_index-1].strip()

                selected_analyze.append(f"{key.replace('_', ' ')} is {analyze_result.replace('NA','unknown')}")
            elif flag and not key in analyze_dict:
                selected_analyze.append(f"{key.replace('_', ' ')} is unknown")

        return (','.join(selected_analyze),)

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Smolvlm_Tagger": Tagger,
    "Smolvlm_SaveTags": SaveTags,
    "Smolvlm_Flux_CLIPTextEncode": FluxCLIPTextEncode,
    "Smolvlm_Caption_Analyzer": CaptionAnalyzer
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Smolvlm_Tagger": "SmolVLM Tagger",
    "Smolvlm_SaveTags": "SmolVLM Save Tags",
    "Smolvlm_Flux_CLIPTextEncode": "SmolVLM Flux Clip Text Encode",
    "Smolvlm_Caption_Analyzer": "SmolVLM Caption Analyzer (Beta)"
}
