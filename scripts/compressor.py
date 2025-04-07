import os.path

import gradio as gr
import torch
from modules.script_callbacks import on_ui_tabs
from safetensors.torch import load_file, save_file
from tqdm import tqdm

MODELS: list[str] = None


def convert_to_fp8(idx: int):
    path: str = MODELS[idx]
    if not path.endswith(".safetensors"):
        raise gr.Error(f'"{path}" is not safetensors')

    output = path.replace(".safetensors", "-fp8.safetensors")
    if os.path.isfile(output):
        raise gr.Error(f'"{output}" already exists...')

    print("loading...")
    sd = load_file(path, device="cuda")
    sd_fp8 = {}
    keys_fp16 = []

    print("parsing...")
    for key, weight in tqdm(sd.items()):
        if key.startswith("model.diffusion_model"):
            sd_fp8[key] = weight.to(torch.float8_e4m3fn)
        elif weight.dtype != torch.float16:
            sd_fp8[key] = weight.to(torch.float16)
            keys_fp16.append(key)
        else:
            sd_fp8[key] = weight

    print("\tconverted model.diffusion_model to", torch.float8_e4m3fn)
    if keys_fp16:
        print(f"\tconverted [{', '.join(keys_fp16)}] to", torch.float16)

    print("saving...")
    save_file(sd_fp8, output)

    print("Done!")
    gr.Info("Done!")


def editor_ui():
    from modules import sd_models

    global MODELS
    MODELS = [mdl.filename for mdl in sd_models.checkpoints_list.values()]
    models = [os.path.basename(mdl) for mdl in MODELS]

    with gr.Blocks() as FP8_EDITOR:
        with gr.Row():
            target = gr.Dropdown(
                label="Checkpoint",
                value=models[0],
                choices=models,
                type="index",
                scale=4,
            )
            target.do_not_save_to_config = True

            button = gr.Button(
                value="Convert",
                variant="primary",
                scale=1,
            )
            button.do_not_save_to_config = True
            button.click(fn=convert_to_fp8, inputs=[target])

    return [(FP8_EDITOR, "Compressor", "sd-webui-fp8")]


on_ui_tabs(editor_ui)
