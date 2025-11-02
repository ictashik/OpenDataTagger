import openai
import pandas as pd
import os
import time
from django.core.cache import cache

LLM_MODEL_NAME = "gemma3:27b"
LLM_CACHE_KEYS = {
    "requests": "llm_request_count",
    "total_time": "llm_total_inference_time",
}

client = openai.OpenAI(
    base_url='http://10.60.23.102:11434/v1',
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
    4) Saves the tagged CSV and logs after every 10 rows for persistence
    5) Saves the final tagged CSV and logs
    """
    try:
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        PROGRESS_STATUS[session_key] = {
            "done": 0,
            "total": total_rows,
            "status": "running",
            "tagged_file": "",
            "logs_file": "",
            "start_time": time.time(),
            "last_update": time.time()
        }

        # Define paths for tagged CSV & logs
        base, ext = os.path.splitext(csv_path)
        logs_path = base + "_logs.csv"
        tagged_path = base + "_tagged.csv"

        # ✅ Store file paths in cache immediately for persistence (24 hours for long jobs)
        cache.set(f"tagged_file_{session_key}", tagged_path, timeout=86400)  # 24 hours
        cache.set(f"logs_file_{session_key}", logs_path, timeout=86400)
        PROGRESS_STATUS[session_key]["tagged_file"] = tagged_path
        PROGRESS_STATUS[session_key]["logs_file"] = logs_path

        # Create empty log file if not exists
        if not os.path.exists(logs_path):
            pd.DataFrame(columns=["row_index", "column", "prompt", "best_answer", "explanation"]).to_csv(logs_path, index=False)

        # ✅ Initialize tagged CSV with original data structure
        for definition in output_definitions:
            out_col = definition['OutputColumn']
            if out_col not in df.columns:
                df[out_col] = ""
        
        # Save initial structure to disk
        df.to_csv(tagged_path, index=False)
        print(f"DEBUG: Initial tagged CSV structure saved to {tagged_path}")

        # Define SYSTEM PROMPT
        system_prompt = f"""
        You are an AI-powered CSV Tagger.
        The user has uploaded a CSV file containing {len(df.columns)} columns and {total_rows} rows.
        The columns are: {', '.join(df.columns)}.
        Your job is to analyze the data row by row and infer values based on user-defined prompts.
        Always return answers in the expected format.
        """

        # ✅ Collect log entries for batch writing
        log_batch = []

        # Process rows for tagging
        for i in range(total_rows):
            row = df.loc[i]

            for definition in output_definitions:
                out_col = definition['OutputColumn']
                prompt_template = definition['PromptTemplate']

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
                log_batch.append(log_entry)

            # Update progress
            PROGRESS_STATUS[session_key]["done"] = i + 1
            PROGRESS_STATUS[session_key]["status"] = f"Processing row {i + 1}/{total_rows}"
            PROGRESS_STATUS[session_key]["last_update"] = time.time()

            # ✅ SAVE EVERY 10 ROWS FOR PERSISTENCE
            if (i + 1) % 10 == 0:
                try:
                    # Save current progress to tagged CSV
                    df.to_csv(tagged_path, index=False)
                    
                    # Append log batch to logs CSV
                    if log_batch:
                        pd.DataFrame(log_batch).to_csv(logs_path, mode='a', header=False, index=False)
                        log_batch = []  # Clear batch after saving
                    
                    print(f"DEBUG: Progress saved at row {i + 1}/{total_rows} -> {tagged_path}")
                    PROGRESS_STATUS[session_key]["status"] = f"Saved progress at row {i + 1}/{total_rows}"
                    
                except Exception as save_error:
                    print(f"ERROR: Failed to save progress at row {i + 1}: {save_error}")
                    PROGRESS_STATUS[session_key]["status"] = f"Save error at row {i + 1}: {save_error}"

        # ✅ FINAL SAVE - Save any remaining log entries
        if log_batch:
            pd.DataFrame(log_batch).to_csv(logs_path, mode='a', header=False, index=False)

        # Save the final tagged CSV
        df.to_csv(tagged_path, index=False)

        # ✅ Update progress status
        PROGRESS_STATUS[session_key]["status"] = "finished"
        PROGRESS_STATUS[session_key]["done"] = total_rows
        PROGRESS_STATUS[session_key]["last_update"] = time.time()

        print(f"DEBUG: Tagging completed. Final files saved -> {tagged_path} and {logs_path}")

    except Exception as e:
        PROGRESS_STATUS[session_key]["status"] = f"error: {str(e)}"
        PROGRESS_STATUS[session_key]["last_update"] = time.time()
        print(f"Tagging Error: {e}")
        
        # ✅ Even on error, try to save partial progress
        try:
            if 'df' in locals() and 'tagged_path' in locals():
                df.to_csv(tagged_path, index=False)
                print(f"DEBUG: Partial progress saved on error -> {tagged_path}")
        except Exception as save_error:
            print(f"ERROR: Failed to save partial progress: {save_error}")

def dummy_llm_call(prompt):
    """Fake LLM call for demonstration."""
    # In reality, you'd call your local LLM here.
    return "FakeTag"

def cleanup_abandoned_sessions(force_cleanup_hours=24):
    """
    Remove progress data for sessions that are clearly abandoned.
    Only cleans up sessions that are:
    1. Older than specified hours (default: 24 hours)
    2. Have status 'finished', 'error', or haven't been updated recently
    
    This is conservative to avoid interfering with long-running jobs.
    """
    current_time = time.time()
    abandoned_sessions = []
    cleanup_threshold = force_cleanup_hours * 3600  # Convert hours to seconds
    
    for session_key, data in PROGRESS_STATUS.items():
        session_start_time = data.get("start_time", current_time)
        session_age = current_time - session_start_time
        status = data.get("status", "")
        
        # Only cleanup if session is old AND meets one of these conditions:
        should_cleanup = (
            session_age > cleanup_threshold and (
                status == "finished" or 
                status.startswith("error") or
                status == "Completed" or
                # If status hasn't changed in the last hour and it's old
                (session_age > cleanup_threshold and "Processing row" not in status)
            )
        )
        
        if should_cleanup:
            abandoned_sessions.append(session_key)
    
    for session_key in abandoned_sessions:
        print(f"DEBUG: Cleaning up old completed/errored session: {session_key}")
        del PROGRESS_STATUS[session_key]
        
        # Note: We keep cache entries as they expire naturally
        # This allows file recovery even after progress cleanup

def get_tagging_session_status(session_key):
    """
    Get the current status of a tagging session.
    Returns None if session doesn't exist.
    """
    return PROGRESS_STATUS.get(session_key)

def is_tagging_in_progress(session_key):
    """
    Check if a tagging session is currently running.
    """
    status = PROGRESS_STATUS.get(session_key)
    if not status:
        return False
    return status.get("status") == "running" or "Processing row" in status.get("status", "")

def get_session_files(session_key, csv_path=None):
    """
    Get the tagged and logs file paths for a session, with fallback to auto-generated names.
    Returns tuple: (tagged_file_path, logs_file_path)
    """
    # Try cache first
    tagged_file = cache.get(f"tagged_file_{session_key}")
    logs_file = cache.get(f"logs_file_{session_key}")
    
    # If not in cache and csv_path provided, generate expected paths
    if (not tagged_file or not logs_file) and csv_path:
        base, ext = os.path.splitext(csv_path)
        tagged_file = base + "_tagged.csv"
        logs_file = base + "_logs.csv"
    
    return tagged_file, logs_file

def ensure_file_persistence(session_key, csv_path):
    """
    Ensure that file paths are stored persistently for a session.
    Call this at the start of tagging to guarantee file access later.
    """
    if csv_path:
        base, ext = os.path.splitext(csv_path)
        tagged_path = base + "_tagged.csv"
        logs_path = base + "_logs.csv"
        
        # Store in cache with long expiration for multi-hour jobs
        cache.set(f"tagged_file_{session_key}", tagged_path, timeout=86400)  # 24 hours
        cache.set(f"logs_file_{session_key}", logs_path, timeout=86400)
        
        return tagged_path, logs_path
    
    return None, None