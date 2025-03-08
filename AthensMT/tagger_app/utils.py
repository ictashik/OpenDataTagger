import openai
import pandas as pd
import os
import time
from django.core.cache import cache

LLM_MODEL_NAME = "llama3"
LLM_CACHE_KEYS = {
    "requests": "llm_request_count",
    "total_time": "llm_total_inference_time",
}

client = openai.OpenAI(
    base_url='http://10.20.110.114:11434/v1',
    api_key='ollama'  # Required, but unused
)

def call_llm_tagging(system_prompt, user_prompt):
    """
    Calls the LLM, tracks request count, and measures response time.
    """
    try:
        # Increment request count
        request_count = cache.get(LLM_CACHE_KEYS["requests"], 0) + 1
        cache.set(LLM_CACHE_KEYS["requests"], request_count, None)  # No expiration

        start_time = time.time()

        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        elapsed_time = time.time() - start_time

        # Update total inference time
        total_time = cache.get(LLM_CACHE_KEYS["total_time"], 0.0) + elapsed_time
        cache.set(LLM_CACHE_KEYS["total_time"], total_time, None)

        message = response.choices[0].message.content.strip()

        # Extract best answer & explanation
        if "Best Answer:" in message and "Explanation:" in message:
            parts = message.split("Explanation:")
            best_answer = parts[0].replace("Best Answer:", "").strip()
            explanation = parts[1].strip()
        else:
            best_answer = message.strip()
            explanation = "No explanation provided."

        return best_answer, explanation

    except Exception as e:
        print(f"LLM API error: {e}")
        return "ERROR", "LLM call failed."

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
    except Exception as e:
        print(f"Error reading config file: {e}")
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

from django.contrib.sessions.backends.db import SessionStore

def row_by_row_tagger(session_key, csv_path, config_path, input_columns, output_definitions):
    """
    Runs in a separate thread:
    1) Reads the CSV
    2) Applies prompts & writes results
    3) Calls LLM for tagging (Best Answer & Explanation)
    4) Saves the final tagged CSV and logs
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

        # Define paths for tagged CSV & logs
        base, ext = os.path.splitext(csv_path)
        logs_path = base + "_logs.csv"
        tagged_path = base + "_tagged.csv"

        # Create empty log file if not exists
        if not os.path.exists(logs_path):
            pd.DataFrame(columns=["row_index", "column", "prompt", "best_answer", "explanation"]).to_csv(logs_path, index=False)

        # Define SYSTEM PROMPT
        system_prompt = f"""
        You are an AI-powered CSV Tagger.
        The user has uploaded a CSV file containing {len(df.columns)} columns and {total_rows} rows.
        The columns are: {', '.join(df.columns)}.
        Your job is to analyze the data row by row and infer values based on user-defined prompts.
        Always return answers in the expected format.
        """

        # Process rows for tagging
        for i in range(total_rows):
            row = df.loc[i]

            for definition in output_definitions:
                out_col = definition['OutputColumn']
                prompt_template = definition['PromptTemplate']

                if out_col not in df.columns:
                    df[out_col] = ""

                user_prompt = f"""
                You are analyzing row {i+1}/{total_rows}.
                Here is the row data:
                {row.to_dict()}
                The user-defined prompt is:
                "{prompt_template}"
                
                Please strictly format your response as:
                Best Answer: <your answer>
                Explanation: <why you chose this answer>
                """

                best_answer, explanation = call_llm_tagging(system_prompt, user_prompt)

                df.at[i, out_col] = best_answer

                log_entry = {
                    "row_index": i,
                    "column": out_col,
                    "prompt": user_prompt,
                    "best_answer": best_answer,
                    "explanation": explanation
                }
                pd.DataFrame([log_entry]).to_csv(logs_path, mode='a', header=False, index=False)

            PROGRESS_STATUS[session_key]["done"] = i + 1

        # Save the tagged CSV
        df.to_csv(tagged_path, index=False)

        # ✅ Update progress status
        PROGRESS_STATUS[session_key]["status"] = "finished"
        PROGRESS_STATUS[session_key]["tagged_file"] = tagged_path
        PROGRESS_STATUS[session_key]["logs_file"] = logs_path

        # ✅ Store in Django cache instead of SessionStore
        cache.set(f"tagged_file_{session_key}", tagged_path, timeout=3600)
        cache.set(f"logs_file_{session_key}", logs_path, timeout=3600)

        print(f"DEBUG: Tagging completed. Files saved -> {tagged_path} and {logs_path}")

    except Exception as e:
        PROGRESS_STATUS[session_key]["status"] = "error"
        print(f"Tagging Error: {e}")

def dummy_llm_call(prompt):
    """Fake LLM call for demonstration."""
    # In reality, you'd call your local LLM here.
    return "FakeTag"