from transformers import AutoTokenizer
import torch

def llama3_fenci(tokenizer, text):
    encoding = tokenizer.encode_plus(
        text,
        return_offsets_mapping=True,  # 返回 token 的位置映射
        add_special_tokens=False,  # 不添加特殊 token
    )

    # 获取 token ID 和 offset_mapping
    token_ids = encoding['input_ids']
    token_ids = torch.tensor(token_ids)
    offset_mapping = encoding['offset_mapping']

    # 通过 offset_mapping 获取每个 token 对应的原始文本部分
    tokens = [text[start:end] for start, end in offset_mapping]

    # 打印结果
    print(f"原始文本: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"分词结果: {tokens}")

    return token_ids, tokens, offset_mapping


if __name__ == '__main__':

    # 加载你的 tokenizer
    device = 'cuda:0'
    model_id = "/home/guoyi/llm_model/suzume-llama-3-8B-multilingual/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 示例中文文本
    # text = "今天天气不错"
    text = "我们经常会说教育关系千家万户，有关教育的讨论总能引发社会关注。最近，不少媒体对云南丽江华坪女子高级中学校长张桂梅的事迹进行了报道，“女校长让1600多名女孩走出大山”感动无数网友。"
    llama3_fenci(tokenizer, text)
