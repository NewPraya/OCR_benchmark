# OCR Benchmark

A modular framework for benchmarking OCR capabilities of various Large Language Models (LLMs) and traditional OCR engines.

## Project Structure

- `data/`: Contains test images and Ground Truth (GT) files.
- `models/`: Modular model implementations (e.g., GPT-4o, Claude 3.5, Gemini, Tesseract).
- `evaluators/`: Metrics and evaluation logic.
- `results/`: Output from benchmark runs.
- `utils/`: Common utilities.

## Formats

### Ground Truth (GT) Format (V2 - Structured)
We now use a structured JSON format to evaluate logical understanding and medical entity extraction:
```json
[
  {
    "file_name": "sample.png",
    "logical_values": { "q1": "N", "q2": "Y", ... },
    "disease_status": { "Heart Disease": "N", "Hypertension": "Y", ... },
    "medical_entities": ["previous NPC", "completed RT", ...],
    "field_pairings": { "MEDICAL HISTORY": "previous NPC completed RT last year." }
  }
]
```

### Model Output Format
Models are now prompted to output a raw JSON object matching the structure above.

## Usage

1. **Setup Environment**:
   ```bash
   conda activate OCR_benchmark
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Copy the template file to `.env` and fill in your keys:
   ```bash
   cp env.example .env
   ```
   Or create a `.env` file manually:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   DASHSCOPE_API_KEY=your_dashscope_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run Benchmark**:
   - **V1 (Text Mode)**: General OCR for any image.
     ```bash
     python3 main.py -v v1 -m gemini -id gemini-3-flash-preview
     ```
   - **V2 (Structured Mode)**: Specific to the Medical Form layout.
     ```bash
     python3 main.py -v v2 -m gemini -id gemini-3-flash-preview
     ```

## Ground Truth Creation Workflow

To benchmark models, you need Ground Truth (GT) JSON files. We use a Markdown-based workflow to make this process easier.

### Step 1: Draft Labels (Model-Assisted)
Use Gemini to generate initial drafts for your images.
- **For V1 (Plain Text)**:
  ```bash
  python3 utils/prep_labels.py -v v1
  ```
- **For V2 (Structured JSON)**:
  ```bash
  python3 utils/prep_labels.py -v v2
  ```
This creates Markdown files in `labeling_v1/` or `labeling_v2/`.

### Step 2: Human Review & Edit
1. Open the `.md` files in `labeling_v1/` or `labeling_v2/`.
2. In Cursor/VSCode, press `Cmd+Shift+V` to open the **Markdown Preview**.
3. You will see the image at the top. Edit the text (V1) or JSON (V2) in the source view below to match exactly what is in the image.

### Step 3: Sync to GT JSON
Once you have saved your edits, sync them back to the central GT files (`data/sample_gt.json` or `data/sample_gt_v2.json`).
- **Sync V1**:
  ```bash
  python3 utils/sync_to_gt.py -v v1
  ```
- **Sync V2**:
  ```bash
  python3 utils/sync_to_gt.py -v v2
  ```

## Visualization Dashboard

The dashboard now supports switching between **V1** and **V2** modes. Use the sidebar to toggle the evaluation logic and view corresponding leaderboards.

## Customizing Structured Prompts (V2)

The V2 structured extraction is schema-dependent. Currently, it is configured for the medical dental form.
To adapt it for new layouts:
1.  Update the prompt in `utils/prompts.py` under `V2_MEDICAL_STRUCTURED_PROMPT`.
2.  Update the Ground Truth in `data/sample_gt_v2.json`.
3.  Adjust the scoring logic in `evaluators/evaluator_v2.py` if the keys change.

## Metrics
### V1 (Text)
- **CER (Character Error Rate)**
- **WER (Word Error Rate)**

### V2 (Structured)
- **Logical Value Accuracy**: Correctness of Y/N logic.
- **Medical Entity Recall**: Recall of handwritten medical terms.
- **Field-Value Pairing**: Correct association of notes to headers.

