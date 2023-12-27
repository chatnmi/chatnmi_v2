from awq import AutoAWQForCausalLM
from huggingface_hub import snapshot_download, hf_hub_download
from langchain.llms import HuggingFacePipeline
from langchain.llms import LlamaCpp
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, pipeline, GenerationConfig

from Common import *

# [model_id, model_file, model_type, model_desc]
models = [
    ["TheBloke/Wizard-Vicuna-13B-Uncensored-HF", "", "HF",
     "This is an acceptable mid-sized model suitable for an average home computer"],
    ["TheBloke/Mistral-7B-Instruct-v0.1-GGUF", "mistral-7b-instruct-v0.1.Q8_0.gguf", "GGUF",
     "It is GGUF format model files for Mistral AI's Mistral 7B Instruct v0.1."],
    ["TheBloke/vicuna-13B-v1.5-GPTQ", "model.safetensors", "GPTQ",
     "It is GPTQ model files for lmsys's Vicuna 13B v1.5."],
    ["TheBloke/Llama-2-7b-Chat-GPTQ", "model.safetensors", "GPTQ",
     "GPTQ model files for Meta Llama 2's"],
    ["TheBloke/Orca-2-7B-GPTQ", "model.safetensors", "GPTQ",
     "GPTQ model files for Microsoft's Orca 2 7B"],
    ["TheBloke/Mixtral-8x7B-v0.1-GGUF", "mixtral-8x7b-v0.1.Q2_K.gguf", "GGUF",
     "The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts."],
    ["TheBloke/llama2_7b_chat_uncensored-AWQ", "", "AWQ",
     "Fine-tuned Llama-2 7B with an uncensored/unfiltered Wizard-Vicuna conversation dataset"]
]


def load_hf_model(device_type, model_id):
    device_map = "" if device_type == "cpu" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_DIR, device_map=device_map)
    model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype="auto", cache_dir=MODEL_DIR, device_map=device_map)
    return model, tokenizer


def load_gptq_model(device_type, model_id, model_file):
    device_map = "" if device_type == "cpu" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=MODEL_DIR, device_map=device_map)

    # model = AutoGPTQForCausalLM.from_quantized(
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype="auto",
        cache_dir=MODEL_DIR
        # model_basename=model_file,
        # use_safetensors=True,
        # trust_remote_code=True,
        # device_map="auto",
        # use_triton=False,
        # quantize_config=None,
    )
    return model, tokenizer


def load_gguf_model(device_type, model_path):
    try:
        kwargs = {
            "model_path": model_path,
            "n_ctx": 4096,
            "max_tokens": 4096,
            "n_batch": 512
        }
        if device_type == "cuda":
            kwargs["n_gpu_layers"] = 100

        return LlamaCpp(**kwargs)
    except:
        print(GREEN(f"Unable to load the model"))
        return None, None


def load_awq_model(device_type, model_path):
    device_map = "" if device_type == "cpu" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=MODEL_DIR, low_cpu_mem_usage=True,
                                              device_map=device_type)

    model = AutoAWQForCausalLM.from_quantized(model_path, cache_dir=MODEL_DIR, fuse_layers=True,
                                              device_map=device_map)
    if device_type == "cuda":
        model = model.to("cuda")
    return model, tokenizer


def download_model(model_id, model_file=""):
    if model_file == "":
        return snapshot_download(repo_id=model_id, cache_dir=MODEL_DIR, resume_download=True)
    else:
        return hf_hub_download(repo_id=model_id, filename=model_file, cache_dir=MODEL_DIR, resume_download=True)


def load_model(model_type, device_type, model_id, model_file=""):
    model_path = download_model(model_id, model_file)

    print(GREEN(f"Running on: {device_type}"))
    match model_type:
        case "HF":
            model, tokenizer = load_hf_model(device_type, model_id)
        case "GPTQ":
            model, tokenizer = load_gptq_model(device_type, model_id, model_file)
        case "GGUF":
            return load_gguf_model(device_type, model_path)
        case "AWQ":
            model, tokenizer = load_awq_model(device_type, model_id)
        case _:
            print(GREEN(f"Model loader not found for type: {model_type}"))
            return None

    generation_config = GenerationConfig.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        generation_config=generation_config
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
