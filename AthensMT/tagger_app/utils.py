# tagger_app/utils.py

import pandas as pd
import os
# tagger_app/utils.py

import openai
import pandas as pd
import os

# OpenAI client (connecting to locally hosted Ollama)
client = openai.OpenAI(
    base_url='http://10.20.110.114:11434/v1',
    api_key='ollama'  # Required, but unused
)

def call_llm_tagging(system_prompt, user_prompt):
    """
    Calls Llama and requests a single best answer + explanation.
    The LLM must return responses in the exact format:
    
    Best Answer: <your answer>
    Explanation: <why you chose this answer>
    """
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        message = response.choices[0].message.content.strip()

        # Extract best answer & explanation
        best_answer = ""
        explanation = ""

        if "Best Answer:" in message and "Explanation:" in message:
            parts = message.split("Explanation:")
            best_answer = parts[0].replace("Best Answer:", "").strip()
            explanation = parts[1].strip()
        else:
            # Fallback in case format is incorrect
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
    3) Calls LLM for tagging (gets both Best Answer & Explanation)
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

        # Prepare logs
        logs = []

        # Construct base system prompt
        system_prompt = f"""
        You are an AI Tagger app. The user has provided a CSV file with {len(df.columns)} columns and {total_rows} rows.
        These are the column names: {', '.join(df.columns)}.
        You will now evaluate each row and infer values based on the given prompt.
        
        IMPORTANT: 
        - Provide your answer in the following strict format:
          Best Answer: <one-line answer>
          Explanation: <brief explanation>
        """

        # Iterate through rows for tagging
        for i in range(total_rows):
            row = df.loc[i]

            for definition in output_definitions:
                out_col = definition['OutputColumn']
                prompt_template = definition['PromptTemplate']

                # Ensure column exists in dataframe
                if out_col not in df.columns:
                    df[out_col] = ""

                # Construct the row-specific user prompt
                user_prompt = f"""
                You are analyzing row {i+1}/{total_rows}.
                Here is the row data:
                {row.to_dict()}
                
                Your task is to predict the value for the column **{out_col}**.
                The user-defined prompt is:
                "{prompt_template}"
                
                Please strictly format your response as:
                Best Answer: <your answer>
                Explanation: <why you chose this answer>
                """

                # Call LLM API
                best_answer, explanation = call_llm_tagging(system_prompt, user_prompt)

                # Save the best answer in the DataFrame (for _tagged.csv)
                df.at[i, out_col] = best_answer

                # Save log entry (for _logs.csv)
                logs.append({
                    "row_index": i,
                    "column": out_col,
                    "prompt": user_prompt,
                    "best_answer": best_answer,
                    "explanation": explanation
                })

            # Update progress
            PROGRESS_STATUS[session_key]["done"] = i + 1

        # Save the tagged CSV
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
        PROGRESS_STATUS[session_key]["status"] = "error"
        print(f"Tagging Error: {e}")
def dummy_llm_call(prompt):
    """Fake LLM call for demonstration."""
    # In reality, you'd call your local LLM here.
    return "FakeTag"