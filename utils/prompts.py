# Centralized prompts for OCR tasks

# V1: Standard text-based OCR (General purpose)
V1_TEXT_PROMPT = (
    "Please perform OCR on this image. Return the recognized text exactly as it appears. "
    "CRITICAL RULES FOR FAIR EVALUATION:\n"
    "1. Do NOT add any numbering (like '1.', '2.') unless they are printed in the image.\n"
    "2. Do NOT add any markdown formatting, code blocks (```), tables, or bold text.\n"
    "3. Do NOT fix typos, grammar, punctuation, or 'improve' the text. Output exactly what is seen.\n"
    "4. Do NOT add any extra punctuation or spaces to align text. Preserve the original layout roughly via line breaks.\n"
    "5. Do NOT try to format the output as a list if it's not a clear list in the image.\n"
    "6. For selection options (like Yes/No or checkboxes), represent the marked choice as 'Y' or 'N' (e.g., if 'N' is circled, output 'N').\n"
    "7. Do NOT include any introductory or concluding remarks. Start the output directly with the recognized text."
)

# V2: Simple handwriting + Y/N extraction (format-agnostic)
V2_SIMPLE_PROMPT = (
    "Perform OCR with two goals: (1) extract all handwritten text, "
    "and (2) detect Y/N selections for any options or checkboxes, regardless of form format.\n\n"
    "Return ONLY a JSON object with this structure:\n"
    "{\n"
    "  \"handwriting_text\": \"...\", \n"
    "  \"yn_options\": { \"Label 1\": \"Y/N\", \"Label 2\": \"Y/N\" }\n"
    "}\n\n"
    "RULES:\n"
    "1. handwriting_text: include ONLY handwritten content. Preserve line breaks when possible. If none, use an empty string.\n"
    "2. yn_options: use the visible option/question label as the key (any language). "
    "Mark 'Y' only if the Y/Yes option is explicitly ticked/circled/checked. "
    "If N/No is marked, or neither is marked, use 'N'.\n"
    "3. Do NOT add extra text outside the JSON. Do NOT use markdown/code blocks."
)

# Default mapping
DEFAULT_PROMPTS = {
    "v1": V1_TEXT_PROMPT,
    "v2": V2_SIMPLE_PROMPT
}
