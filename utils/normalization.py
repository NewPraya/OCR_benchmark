import re
import unicodedata
import string

def normalize_text(text: str, remove_punctuation: bool = True, strict_semantic: bool = False) -> str:
    """
    Highly robust normalization for OCR benchmark to focus on semantic content.
    :param strict_semantic: If True, removes all non-alphanumeric characters for maximum matching robustness.
    """
    if not text:
        return ""
    
    # 1. Unicode NFKC 归一化
    text = unicodedata.normalize('NFKC', text)
    
    # 2. 移除模型特有的描述性噪声 (Case-insensitive)
    noise_patterns = [
        r'handwritten note:', 
        r'handwritten:', 
        r'\[redacted\]', 
        r'note:', 
        r'caption this image',
        r'recognized text:'
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 3. 统一转为大写
    text = text.upper()

    # 4. 符号语义映射 (将打钩/打叉/圈选 统一为 Y/N)
    # 选中的标志
    text = re.sub(r'[\(\[\{][XVV\u2713\u2714][\)\]\}]', ' Y ', text)
    # 未选中的标志
    text = re.sub(r'[\(\[\{]\s*[\)\]\}]', ' N ', text)
    
    # 5. 处理 CJK 字符间的空格
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    
    # 6. 打平所有换行符和多余空白 (OCR 内容一致性比排版更重要)
    text = text.replace('\n', ' ')
    
    # 7. 去除标点符号 (保留必要的 Y/N 判断，但去除括号等干扰)
    if remove_punctuation or strict_semantic:
        # 我们要保留 Y 和 N，但去掉包裹它们的括号
        cjk_punctuation = r"""！"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”‘’〈〉《》「」『』【】〔〕〖〗〽〰〾〿–—‘’“”„‟†‡•‥…‰′″‹›※‼‽‾‿⁀⁁⁂⁃"""
        all_punct = string.punctuation + cjk_punctuation
        table = str.maketrans('', '', all_punct)
        text = text.translate(table)

    # 8. 严格语义模式：移除所有非字母数字字符 (用于 "under medication" vs "undermedication")
    if strict_semantic:
        text = re.sub(r'[^A-Z0-9]', '', text)

    # 9. 最后的清理：合并空格并去除首尾空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
