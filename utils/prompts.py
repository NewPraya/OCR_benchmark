# Centralized prompts for OCR tasks

# V1: Standard text-based OCR (General purpose)
V1_TEXT_PROMPT = (
    'Please perform OCR on this image. Return only the recognized text, preserving the layout and line breaks. '
    'For selection options (like Yes/No or checkboxes), if an option is circled, checked (V), or crossed (X), '
    'represent it as "Y" or "N" to indicate it is the selected/correct choice. '
    'Do not add any introduction or conclusion.'
)

# V2: Structured Extraction (Schema-dependent)
# Note: This specific prompt is for the Medical Form dataset.
V2_MEDICAL_STRUCTURED_PROMPT = (
    "Please analyze this medical form and return a JSON object with the following fields:\n"
    "1. 'logical_values': A dictionary for questions. Use keys like 'q1' , 'q14'. For each, if 'Y' is circled/marked, use 'Y'; if 'N' is marked, use 'N'.\n"
    "2. 'disease_status': A dictionary for the bottom list (e.g., 'Heart Disease', 'T.B.', 'Hypertension', etc.). For each, if 'Y' is marked/written, use 'Y'; if 'N' is marked/written, use 'N'. If empty, use 'N'.\n"
    "3. 'medical_entities': A list of all handwritten medical terms, medications, history notes, and handwritten details you found.\n"
    "4. 'field_pairings': A dictionary where keys are the printed field names (e.g., 'MEDICAL HISTORY', 'Childhood Diseases') and values are the specific handwritten notes following them.\n"
    "Return ONLY the raw JSON object, no markdown code blocks."
)

# Default mapping
DEFAULT_PROMPTS = {
    "v1": V1_TEXT_PROMPT,
    "v2": V2_MEDICAL_STRUCTURED_PROMPT
}
