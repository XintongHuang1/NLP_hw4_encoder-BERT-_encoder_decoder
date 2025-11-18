
import argparse

import os

import torch

from transformers import GenerationConfig



from load_data import (

    T5Dataset,

    normal_collate_fn,

    test_collate_fn,

)

from t5_utils import (

    initialize_model,

    load_model_from_checkpoint,

)

from utils import compute_records, set_random_seeds

from transformers import T5TokenizerFast



TOKENIZER = T5TokenizerFast.from_pretrained('google-t5/t5-small')

BOS_ID = 0

MAX_TGT_LEN = 300





def print_section(title: str, text: str):

    """漂亮一点的分割打印。"""

    print(f"{'-' * 40} {title} {'-' * 40}\n")

    if text is None or text == "":

        print("[EMPTY]")

    else:

        print(text)

    print()  # 空行





def get_single_dataloader(split: str, idx: int):

    """

    构造只包含单个样本的 DataLoader，复用主线的 T5Dataset + collate_fn。

    """

    assert split in {"train", "dev", "test"}



    from torch.utils.data import DataLoader, Subset



    dataset = T5Dataset(data_folder="data", split=split)



    if idx < 0 or idx >= len(dataset):

        raise IndexError(f"Index {idx} out of range for split '{split}' (size={len(dataset)})")



    subset = Subset(dataset, [idx])

    collate_fn = test_collate_fn if split == "test" else normal_collate_fn



    loader = DataLoader(

        subset,

        batch_size=1,

        shuffle=False,

        collate_fn=collate_fn,

    )

    return loader, dataset





def build_model(args):

    """

    根据 init_mode 选择：

    - checkpoint: 加载本地 fine-tuned 模型（主线行为）

    - pretrained: 加载原始 HuggingFace t5-small（未在本任务上训练）

    """

    if args.init_mode == "checkpoint":

        # 和 train_t5.py 主线一致：从 checkpoint_dir 载入 best_model.pt

        model = load_model_from_checkpoint(args, best=True)

    else:

        # 使用 initialize_model: pretrained

        args.finetune = True  # Load pretrained weights

        model = initialize_model(args)



    # 和主线保持一致：设置 decoder_start_token_id

    model.config.decoder_start_token_id = BOS_ID

    return model





def debug_one_case(args):

    set_random_seeds(args.seed)



    # 1) 单样本 DataLoader

    loader, _ = get_single_dataloader(args.split, args.index)



    # 2) 初始化或加载模型

    model = build_model(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    model.eval()



    # 3) 取出这个 batch（只有一个样本）

    batch = next(iter(loader))



    if args.split == "test":

        encoder_input_ids, encoder_attention_mask, initial_decoder_inputs = batch

        decoder_targets = None

    else:

        encoder_input_ids, encoder_attention_mask, decoder_inputs, decoder_targets, initial_decoder_inputs = batch



    encoder_input_ids = encoder_input_ids.to(device)

    encoder_attention_mask = encoder_attention_mask.to(device)



    # 4) 按主线的方式做 generate（贴着 eval_epoch / test_inference）

    with torch.no_grad():

        gen_outputs = model.generate(

            input_ids=encoder_input_ids,

            attention_mask=encoder_attention_mask,

            max_new_tokens=MAX_TGT_LEN,

            num_beams=8,

            early_stopping=True,

            length_penalty=0.95,

            decoder_start_token_id=0

        )



    # 5) 可视化「给模型的 prompt」和「模型原始输出」

    # 5.1 实际喂给 encoder 的文本（从 encoder_input_ids 反解码）

    encoder_prompt_text = TOKENIZER.decode(

        encoder_input_ids[0],

        skip_special_tokens=True

    ).strip()



    # 5.2 模型原始输出（主线用于 SQL 的 decode 结果）

    raw_decoded_output = TOKENIZER.decode(

        gen_outputs[0],

        skip_special_tokens=True

    ).strip()



    # 主线实际用于执行 SQL 的内容

    pred_sql_for_exec = raw_decoded_output



    # 6) 从 data/ 读取原始 NL 和 gold SQL（如果有）

    nl_path = os.path.join("data", f"{args.split}.nl")

    with open(nl_path, "r", encoding="utf-8") as f:

        nl_lines = [line.strip() for line in f.readlines() if line.strip()]

    nl_query = nl_lines[args.index]



    gold_sql = None

    if args.split != "test":

        sql_path = os.path.join("data", f"{args.split}.sql")

        with open(sql_path, "r", encoding="utf-8") as f:

            sql_lines = [line.strip() for line in f.readlines() if line.strip()]

        gold_sql = sql_lines[args.index]



    # 7) 用主线的 compute_records 执行 SQL（注意：只传一个参数）

    records, error_msgs = compute_records([pred_sql_for_exec])



    # 8) 漂亮打印：把整个「从输入到执行」流程真实展开



    print(f"\n{'=' * 40} DEBUG ONE CASE {'=' * 40}\n")

    print(f"Split      : {args.split}")

    print(f"Index      : {args.index}")

    print(f"Init mode  : {args.init_mode}  "

          f"({'fine-tuned checkpoint' if args.init_mode == 'checkpoint' else 'pretrained model'})\n")



    # 原始 NL

    print_section("Natural Language Input", nl_query)



    # encoder 的 prompt（模型真正看到的文本，含前缀 & 截断效果）

    print_section("Encoder Prompt to Model", encoder_prompt_text)



    # gold SQL（只有 train/dev 有）

    if gold_sql is not None:

        print_section("Gold SQL", gold_sql)



    # 模型原始输出（主线 decode 结果）

    print_section("Model Raw Decoded Output", raw_decoded_output)



    # 真正用于执行的 SQL（主线流程）

    print_section("SQL Used for Execution", pred_sql_for_exec)



    # 执行情况

    if error_msgs[0]:

        print_section("Execution Error", error_msgs[0])

    else:

        rec_text = "\n".join(str(row) for row in records[0]) or "[NO RECORDS RETURNED]"

        print_section("Execution Records", rec_text)



    print(f"{'=' * 40} END {'=' * 40}\n")





def parse_args():

    parser = argparse.ArgumentParser(description="Debug a single example with the T5 text-to-SQL model.")

    parser.add_argument(

        "--split",

        type=str,

        default="dev",

        choices=["train", "dev", "test"],

        help="数据集划分",

    )

    parser.add_argument(

        "--index",

        type=int,

        default=0,

        help="要查看的样本 index（0-based）",

    )

    parser.add_argument(

        "--checkpoint_dir",

        type=str,

        default="checkpoints/ft_experiments/exp4_lr3e4",

        help="包含 best.pt 的目录（和 train_t5.py 里一致）",

    )

    parser.add_argument(

        "--seed",

        type=int,

        default=42,

        help="随机种子",

    )

    parser.add_argument(

        "--init_mode",

        type=str,

        default="checkpoint",

        choices=["checkpoint", "pretrained"],

        help=(

            "模型初始化方式：\n"

            "  - checkpoint: 使用本地 fine-tuned checkpoint（主线默认）\n"

            "  - pretrained: 使用原始 HuggingFace 预训练权重（未在本任务上训练）\n"

        ),

    )

    parser.add_argument('--finetune', action='store_true', help='For compatibility')

    return parser.parse_args()





if __name__ == "__main__":

    args = parse_args()

    debug_one_case(args)


