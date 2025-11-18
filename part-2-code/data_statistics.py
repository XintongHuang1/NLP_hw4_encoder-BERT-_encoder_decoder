"""

数据统计脚本：用于Q4报告

统计预处理前后的数据特征

"""

import os

from collections import Counter

from transformers import T5TokenizerFast

import numpy as np



# 初始化T5分词器

tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')



def load_file_lines(filepath):

    """加载文件并返回行列表"""

    if not os.path.exists(filepath):

        return []

    with open(filepath, 'r', encoding='utf-8') as f:

        return [line.strip() for line in f if line.strip()]



def compute_statistics_before_preprocessing(data_folder='data'):

    """

    表1: 预处理之前的数据统计

    直接对原始文本进行分词统计

    """

    print("=" * 70)

    print("表1: 预处理之前的数据统计")

    print("=" * 70)

    

    splits = ['train', 'dev']

    results = {}

    

    for split in splits:

        nl_path = os.path.join(data_folder, f'{split}.nl')

        sql_path = os.path.join(data_folder, f'{split}.sql')

        

        nl_lines = load_file_lines(nl_path)

        sql_lines = load_file_lines(sql_path)

        

        # 统计示例数量

        num_examples = len(nl_lines)

        

        # 对每个自然语言句子分词（不添加prompt）

        nl_token_lengths = []

        nl_vocab = set()

        for nl in nl_lines:

            tokens = tokenizer.encode(nl, add_special_tokens=False)

            nl_token_lengths.append(len(tokens))

            nl_vocab.update(tokens)

        

        # 对每个SQL查询分词

        sql_token_lengths = []

        sql_vocab = set()

        for sql in sql_lines:

            tokens = tokenizer.encode(sql, add_special_tokens=False)

            sql_token_lengths.append(len(tokens))

            sql_vocab.update(tokens)

        

        # 计算平均长度

        avg_nl_length = np.mean(nl_token_lengths) if nl_token_lengths else 0

        avg_sql_length = np.mean(sql_token_lengths) if sql_token_lengths else 0

        

        # 词汇量大小

        nl_vocab_size = len(nl_vocab)

        sql_vocab_size = len(sql_vocab)

        

        results[split] = {

            'num_examples': num_examples,

            'avg_nl_length': avg_nl_length,

            'avg_sql_length': avg_sql_length,

            'nl_vocab_size': nl_vocab_size,

            'sql_vocab_size': sql_vocab_size,

            'max_nl_length': max(nl_token_lengths) if nl_token_lengths else 0,

            'max_sql_length': max(sql_token_lengths) if sql_token_lengths else 0,

        }

    

    # 打印表格

    print(f"\n{'统计名称':<30} {'训练集':<15} {'开发集':<15}")

    print("-" * 70)

    print(f"{'示例数量':<30} {results['train']['num_examples']:<15} {results['dev']['num_examples']:<15}")

    print(f"{'平均句子长度(tokens)':<30} {results['train']['avg_nl_length']:<15.2f} {results['dev']['avg_nl_length']:<15.2f}")

    print(f"{'平均SQL查询长度(tokens)':<30} {results['train']['avg_sql_length']:<15.2f} {results['dev']['avg_sql_length']:<15.2f}")

    print(f"{'词汇量大小(自然语言)':<30} {results['train']['nl_vocab_size']:<15} {results['dev']['nl_vocab_size']:<15}")

    print(f"{'词汇量大小(SQL)':<30} {results['train']['sql_vocab_size']:<15} {results['dev']['sql_vocab_size']:<15}")

    print(f"{'最大句子长度(tokens)':<30} {results['train']['max_nl_length']:<15} {results['dev']['max_nl_length']:<15}")

    print(f"{'最大SQL查询长度(tokens)':<30} {results['train']['max_sql_length']:<15} {results['dev']['max_sql_length']:<15}")

    print()

    

    return results



def compute_statistics_after_preprocessing(data_folder='data'):

    """

    表2: 预处理后的数据统计

    考虑prompt添加、截断等预处理操作

    """

    print("=" * 70)

    print("表2: 预处理后的数据统计")

    print("=" * 70)

    

    # 预处理参数（与load_data.py中的设置一致）

    ENCODER_MAX_LENGTH = 64  # 自然语言编码器最大长度

    DECODER_MAX_LENGTH = 512  # SQL解码器最大长度

    PROMPT_TEMPLATE = "Translate the question to SQL: "

    

    splits = ['train', 'dev']

    results = {}

    

    for split in splits:

        nl_path = os.path.join(data_folder, f'{split}.nl')

        sql_path = os.path.join(data_folder, f'{split}.sql')

        

        nl_lines = load_file_lines(nl_path)

        sql_lines = load_file_lines(sql_path)

        

        num_examples = len(nl_lines)

        

        # 模拟预处理：添加prompt + 截断

        processed_nl_lengths = []

        processed_nl_vocab = set()

        truncated_nl_count = 0

        

        for nl in nl_lines:

            # 添加prompt

            prompted_nl = f"{PROMPT_TEMPLATE}{nl}"

            # 编码并截断

            tokens = tokenizer.encode(prompted_nl, add_special_tokens=True, 

                                     max_length=ENCODER_MAX_LENGTH, truncation=True)

            processed_nl_lengths.append(len(tokens))

            processed_nl_vocab.update(tokens)

            

            # 检查是否发生截断

            tokens_no_truncate = tokenizer.encode(prompted_nl, add_special_tokens=True, truncation=False)

            if len(tokens_no_truncate) > ENCODER_MAX_LENGTH:

                truncated_nl_count += 1

        

        # 处理SQL（截断）

        processed_sql_lengths = []

        processed_sql_vocab = set()

        truncated_sql_count = 0

        

        for sql in sql_lines:

            tokens = tokenizer.encode(sql, add_special_tokens=True, 

                                     max_length=DECODER_MAX_LENGTH, truncation=True)

            processed_sql_lengths.append(len(tokens))

            processed_sql_vocab.update(tokens)

            

            # 检查是否发生截断

            tokens_no_truncate = tokenizer.encode(sql, add_special_tokens=True, truncation=False)

            if len(tokens_no_truncate) > DECODER_MAX_LENGTH:

                truncated_sql_count += 1

        

        # 计算统计

        avg_processed_nl = np.mean(processed_nl_lengths) if processed_nl_lengths else 0

        avg_processed_sql = np.mean(processed_sql_lengths) if processed_sql_lengths else 0

        

        # Prompt token数量

        prompt_tokens = tokenizer.encode(PROMPT_TEMPLATE, add_special_tokens=False)

        prompt_length = len(prompt_tokens)

        

        results[split] = {

            'num_examples': num_examples,

            'avg_processed_nl': avg_processed_nl,

            'avg_processed_sql': avg_processed_sql,

            'processed_nl_vocab': len(processed_nl_vocab),

            'processed_sql_vocab': len(processed_sql_vocab),

            'max_nl_after': max(processed_nl_lengths) if processed_nl_lengths else 0,

            'max_sql_after': max(processed_sql_lengths) if processed_sql_lengths else 0,

            'truncated_nl_pct': 100 * truncated_nl_count / num_examples if num_examples > 0 else 0,

            'truncated_sql_pct': 100 * truncated_sql_count / num_examples if num_examples > 0 else 0,

            'prompt_length': prompt_length,

            'encoder_max': ENCODER_MAX_LENGTH,

            'decoder_max': DECODER_MAX_LENGTH,

        }

    

    # 打印表格

    print(f"\n模型: T5-small with custom preprocessing")

    print(f"Encoder最大长度: {ENCODER_MAX_LENGTH} tokens")

    print(f"Decoder最大长度: {DECODER_MAX_LENGTH} tokens")

    print(f"Prompt模板: '{PROMPT_TEMPLATE}'")

    print(f"Prompt长度: {results['train']['prompt_length']} tokens")

    print()

    

    print(f"{'统计名称':<35} {'训练集':<15} {'开发集':<15}")

    print("-" * 70)

    print(f"{'示例数量':<35} {results['train']['num_examples']:<15} {results['dev']['num_examples']:<15}")

    print(f"{'平均编码后句子长度(tokens)':<35} {results['train']['avg_processed_nl']:<15.2f} {results['dev']['avg_processed_nl']:<15.2f}")

    print(f"{'平均编码后SQL长度(tokens)':<35} {results['train']['avg_processed_sql']:<15.2f} {results['dev']['avg_processed_sql']:<15.2f}")

    print(f"{'处理后词汇量(自然语言)':<35} {results['train']['processed_nl_vocab']:<15} {results['dev']['processed_nl_vocab']:<15}")

    print(f"{'处理后词汇量(SQL)':<35} {results['train']['processed_sql_vocab']:<15} {results['dev']['processed_sql_vocab']:<15}")

    print(f"{'最大编码后句子长度(tokens)':<35} {results['train']['max_nl_after']:<15} {results['dev']['max_nl_after']:<15}")

    print(f"{'最大编码后SQL长度(tokens)':<35} {results['train']['max_sql_after']:<15} {results['dev']['max_sql_after']:<15}")

    print(f"{'被截断的自然语言比例(%)':<35} {results['train']['truncated_nl_pct']:<15.2f} {results['dev']['truncated_nl_pct']:<15.2f}")

    print(f"{'被截断的SQL比例(%)':<35} {results['train']['truncated_sql_pct']:<15.2f} {results['dev']['truncated_sql_pct']:<15.2f}")

    print()

    

    return results



def compute_tokenizer_info():

    """打印分词器基本信息"""

    print("=" * 70)

    print("T5 Tokenizer 信息")

    print("=" * 70)

    print(f"模型: google-t5/t5-small")

    print(f"总词汇量大小: {tokenizer.vocab_size}")

    print(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    print(f"UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

    print()



def main():

    """主函数：生成完整的统计报告"""

    print("\n" + "=" * 70)

    print("数据统计报告 - Q4")

    print("=" * 70 + "\n")

    

    # 分词器信息

    compute_tokenizer_info()

    

    # 表1: 预处理前

    stats_before = compute_statistics_before_preprocessing()

    

    # 表2: 预处理后

    stats_after = compute_statistics_after_preprocessing()

    

    # 额外分析：比较预处理前后的变化

    print("=" * 70)

    print("预处理影响分析")

    print("=" * 70)

    print(f"\n训练集:")

    nl_increase = stats_after['train']['avg_processed_nl'] - stats_before['train']['avg_nl_length']

    print(f"  - 平均自然语言长度变化: +{nl_increase:.2f} tokens (添加prompt)")

    print(f"  - 自然语言截断率: {stats_after['train']['truncated_nl_pct']:.2f}%")

    print(f"  - SQL截断率: {stats_after['train']['truncated_sql_pct']:.2f}%")

    

    print(f"\n开发集:")

    nl_increase_dev = stats_after['dev']['avg_processed_nl'] - stats_before['dev']['avg_nl_length']

    print(f"  - 平均自然语言长度变化: +{nl_increase_dev:.2f} tokens (添加prompt)")

    print(f"  - 自然语言截断率: {stats_after['dev']['truncated_nl_pct']:.2f}%")

    print(f"  - SQL截断率: {stats_after['dev']['truncated_sql_pct']:.2f}%")

    

    print("\n" + "=" * 70)

    print("统计完成！可以将以上表格复制到报告中")

    print("=" * 70)



if __name__ == "__main__":

    main()


