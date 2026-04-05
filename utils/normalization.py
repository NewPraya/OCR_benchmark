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
    
    # 1) Unicode NFKC normalization.
    text = unicodedata.normalize('NFKC', text)
    
    # 2) Remove model-specific descriptive noise (case-insensitive).
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

    # 3) Uppercase for case-insensitive matching.
    text = text.upper()

    # 4) Symbol semantic mapping: normalize common checkbox marks to Y/N.
    # Selected mark variants.
    text = re.sub(r'[\(\[\{][XVV\u2713\u2714][\)\]\}]', ' Y ', text)
    # Unselected blank box variants.
    text = re.sub(r'[\(\[\{]\s*[\)\]\}]', ' N ', text)
    
    # 5) Remove spaces between adjacent CJK characters.
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    
    # 6) Flatten newlines and extra whitespace.
    text = text.replace('\n', ' ')
    
    # 7) Remove punctuation noise (including bracket wrappers around Y/N marks).
    if remove_punctuation or strict_semantic:
        cjk_punctuation = r"""！"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”‘’〈〉《》「」『』【】〔〕〖〗〽〰〾〿–—‘’“”„‟†‡•‥…‰′″‹›※‼‽‾‿⁀⁁⁂⁃"""
        all_punct = string.punctuation + cjk_punctuation
        table = str.maketrans('', '', all_punct)
        text = text.translate(table)

    # 8) Strict semantic mode: keep only alphanumeric chars.
    if strict_semantic:
        text = re.sub(r'[^A-Z0-9]', '', text)

    # 9) Final cleanup.
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
