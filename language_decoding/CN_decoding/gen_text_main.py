import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

from decode_lib import decode_text_from_embeddings

warnings.filterwarnings("ignore")


files = './Decoded_story.pth'

class Config:
    # model_id = "/home/guoyi/llm_model/bloom-1b1" # bloom-1b1_path
    model_id = "bigscience/bloom-1b1"
    device = 'cuda:4'
    feature_layer = 20

    output_language = 'CN'  # 'CN', 'EN', 'FR', 'NL'
    dictionary_path = None
    pth_path = files
    # dictionary_path = "/home/guoyi/code/BrainDecoding_min_lfs/dictionary/OpenTaal-210G-basis-gekeurd.txt"


def main():
    config = Config()

    if not os.path.exists(config.pth_path):
        raise FileNotFoundError(f"❌ File not found: {config.pth_path}")

    if not config.pth_path.endswith('.pth'):
        raise ValueError(f"❌ Input must be a .pth file, got: {config.pth_path}")

    output_path = config.pth_path.replace('.pth', '_text.npy')

    print(f"\n{'=' * 60}")
    print(f"📊 Decoding: {os.path.basename(config.pth_path)}")
    print(f"{'=' * 60}")

    print(f"🚀 Loading model from {config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map=config.device,
    )
    print(f"✅ Model loaded on {config.device}")

    print(f"📁 Loading data...")
    gt_eb_all = torch.load(config.pth_path, map_location=config.device)[None, :]
    print(f"✅ Data loaded. Shape: {gt_eb_all.shape}")

    print(f"🔄 Starting decoding...")
    result = decode_text_from_embeddings(
        referen_eb=gt_eb_all,
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        output_lang=config.output_language,
        feature_layer=config.feature_layer,
        dictionary_path=config.dictionary_path,
        verbose=True
    )

    res = gt_eb_all.shape[1] - len(result)
    if res > 0:
        print(f"⚠️  {res} samples remaining, continuing...")
        while True:
            result2 = decode_text_from_embeddings(
                referen_eb=gt_eb_all[:, -res:],
                model=model,
                tokenizer=tokenizer,
                device=config.device,
                output_lang=config.output_language,
                feature_layer=config.feature_layer,
                dictionary_path=config.dictionary_path,
                verbose=True
            )
            result = np.concatenate([result, result2])
            res = gt_eb_all.shape[1] - len(result)
            if res == 0:
                break

    np.save(output_path, result)
    print(f"💾 Results saved to: {output_path}")
    print(f"✅ Completed. Total words: {len(result)}")

    del gt_eb_all
    torch.cuda.empty_cache()

    print(f"\n🎉 All done!")


if __name__ == '__main__':
    main()