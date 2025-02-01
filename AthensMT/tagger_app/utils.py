# tagger_app/utils.py

import pandas as pd
import os

def load_config_file(config_path):
    """
    Reads the config CSV (if it exists) and returns a list of dicts:
    [
      {"OutputColumn": "Tag1", "PromptTemplate": "Is {Food} vegetarian?"},
      ...
    ]
    If file doesn't exist or is empty, returns an empty list.
    """
    if not config_path or not os.path.exists(config_path):
        return []

    try:
        df = pd.read_csv(config_path)
        # Ensure columns exist
        if 'OutputColumn' not in df.columns or 'PromptTemplate' not in df.columns:
            return []

        # Convert to list of dicts
        return df.to_dict('records')
    except:
        # If any error reading, return empty
        return []

def save_config_file(config_path, config_data):
    """
    Saves a list of dicts (with keys "OutputColumn", "PromptTemplate")
    to a CSV at config_path.
    """
    df = pd.DataFrame(config_data)
    df.to_csv(config_path, index=False)

# tagger_app/utils.py
import threading
import time
import pandas as pd
import os

# We'll store progress info keyed by session ID or some unique key
PROGRESS_STATUS = {}
# Example structure: 
# PROGRESS_STATUS[session_key] = {
#     "done": 0,
#     "total": 0,
#     "status": "running" or "finished" or "error",
#     "tagged_file": "...",
#     "logs_file": "..."
# }

def row_by_row_tagger(session_key, csv_path, config_path, input_columns, output_definitions):
    """
    Runs in a separate thread:
    1) Reads the CSV
    2) For each row, applies prompts and writes results
    3) Logs each row's output
    4) Saves final tagged CSV and sets progress in PROGRESS_STATUS
    """
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        PROGRESS_STATUS[session_key] = {
            "done": 0,
            "total": total_rows,
            "status": "running",
            "tagged_file": "",
            "logs_file": ""
        }

        # Prepare logs
        logs = []

        # For each row, replace placeholders & "call LLM"
        for i in range(total_rows):
            row = df.loc[i]

            for definition in output_definitions:
                out_col = definition['OutputColumn']
                prompt_template = definition['PromptTemplate']

                # Create the column if not present
                if out_col not in df.columns:
                    df[out_col] = ""

                # Substitute placeholders in prompt
                prompt = prompt_template
                for col in input_columns:
                    placeholder = f'{{{col}}}'
                    if placeholder in prompt:
                        prompt = prompt.replace(placeholder, str(row[col]))

                # "Call LLM" -> For now, a dummy function
                llm_result = dummy_llm_call(prompt)

                # Assign the result
                df.at[i, out_col] = llm_result

                # Log this row
                logs.append({
                    "row_index": i,
                    "prompt": prompt,
                    "result": llm_result
                })

            # Update progress
            PROGRESS_STATUS[session_key]["done"] = i + 1
            time.sleep(0.1)  # simulate some delay

        # Once done, save the tagged CSV
        base, ext = os.path.splitext(csv_path)
        tagged_path = base + "_tagged.csv"
        df.to_csv(tagged_path, index=False)

        # Save logs CSV
        logs_path = base + "_logs.csv"
        pd.DataFrame(logs).to_csv(logs_path, index=False)

        # Update status
        PROGRESS_STATUS[session_key]["status"] = "finished"
        PROGRESS_STATUS[session_key]["tagged_file"] = tagged_path
        PROGRESS_STATUS[session_key]["logs_file"] = logs_path

    except Exception as e:
        # In case of error, store in progress
        PROGRESS_STATUS[session_key]["status"] = "error"

def dummy_llm_call(prompt):
    """Fake LLM call for demonstration."""
    # In reality, you'd call your local LLM here.
    return "FakeTag"