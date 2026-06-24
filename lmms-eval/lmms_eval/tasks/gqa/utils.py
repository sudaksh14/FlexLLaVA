from datasets import load_dataset

GQA_RAW_IMAGE_DATASET = None
GQA_ID2INDEX = None


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2INDEX
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=False)
        # Build index map once (string id → integer row index) without loading images into RAM.
        GQA_ID2INDEX = {row["id"]: i for i, row in enumerate(GQA_RAW_IMAGE_DATASET.select_columns(["id"]))}
    idx = GQA_ID2INDEX[doc["imageId"]]
    image = GQA_RAW_IMAGE_DATASET[idx]["image"].convert("RGB")
    return [image]


def gqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"
