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

# V2: Structured Extraction (Schema-dependent)
# Note: This specific prompt is for the Medical Form dataset.
V2_MEDICAL_STRUCTURED_PROMPT = (
    "Analyze the medical form image and extract information into a JSON object. "
    "You must act as a precise data entry clerk. Focus on the actual visual markings (circles, ticks, crosses, handwriting) rather than just the presence of printed text.\n\n"
    "REQUIRED JSON STRUCTURE:\n"
    "{\n"
    "  \"logical_values\": { \"q1\": \"Y/N\", ..., \"q14\": \"Y/N\" },\n"
    "  \"disease_status\": { \"Heart Disease\": \"Y/N\", \"Hypertension\": \"Y/N\", \"Blood Disease\": \"Y/N\", \"V.D.\": \"Y/N\", \"Kidney Disease\": \"Y/N\", \"Diabetes\": \"Y/N\", \"Thyroid Disease\": \"Y/N\", \"Other Medical Problems\": \"Y/N\", \"T.B.\": \"Y/N\", \"Epilepsy\": \"Y/N\", \"Stroke\": \"Y/N\" },\n"
    "  \"medical_entities\": [ \"list\", \"of\", \"handwritten\", \"notes\" ],\n"
    "  \"field_pairings\": { \"Field Name\": \"Handwritten Content\" }\n"
    "}\n\n"
    "EXTRACTION RULES:\n"
    "1. 'logical_values': Only mark 'Y' if the 'Y' option is explicitly circled, ticked, or checked. If 'N' is marked or neither is marked, use 'N'.\n"
    "2. 'disease_status': Use ONLY the English keys provided in the structure above. \n"
    "   IMPORTANT: A tick '(V)', a checkmark, or a circle around the label means 'Y'. A cross '(X)', a slash, or a blank space means 'N'.\n"
    "3. 'medical_entities': Extract all handwritten text (medications, history, dates). Do not include printed text. If no handwriting is present, return an empty list.\n"
    "4. 'field_pairings': Map printed headers (e.g., 'DRUGS', 'Date', 'MEDICAL HISTORY') to their corresponding handwritten values. Use the exact printed header as the key.\n"
    "5. Return ONLY the raw JSON object. No markdown code blocks, no explanations, no 'Here is the JSON'."
)

# Default mapping
DEFAULT_PROMPTS = {
    "v1": V1_TEXT_PROMPT,
    "v2": V2_MEDICAL_STRUCTURED_PROMPT
}
