# OpenDataTagger

OpenDataTagger is an **AI-powered CSV tagging tool** built with Django. It allows users to upload CSV files, configure custom tagging prompts, and process data row-by-row using a local LLM (Llama3). The application provides real-time progress updates, logs detailed AI responses, and makes the final tagged CSV available for download—all through a clean, responsive UI powered by Tailwind CSS.

## Features

- **CSV Upload & Configuration**: Upload your dataset along with an optional configuration CSV that pre-defines tagging prompts.
- **Custom Column Definition**: Select input columns and define output columns along with custom AI prompt templates.
- **Background Processing**: Tagging is performed in a background thread so that users can monitor progress in real time.
- **Real-Time Progress & Logs**: View live updates and logs for each processed row, including the AI’s best answer and explanation.
- **Downloadable Results**: Once tagging is complete, download both the tagged CSV file and a detailed log file.
- **Local LLM Integration**: Leverage a locally hosted LLM (configured with Llama3) via an OpenAI-like API for generating AI-powered tags.
- **Modern UI**: A responsive interface built with Tailwind CSS, complete with a sidebar for easy navigation.

## Prerequisites

- **Python 3.8+**
- **Django 5.0.1** (or later)
- **Pandas**
- **OpenAI Python Library** (or a client compatible with your local LLM)
- A local LLM server running at `http://localhost:6000/v1` (configured to work with Llama3)

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/ictashik-opendatatagger.git
    cd ictashik-opendatatagger.git
    ```

2. **Create a Virtual Environment & Activate It:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    If a `requirements.txt` is provided, run:

    ```bash
    pip install -r requirements.txt
    ```

    Otherwise, install manually:

    ```bash
    pip install Django==5.0.1 pandas openai
    ```

4. **Apply Migrations:**

    ```bash
    python AthensMT/manage.py migrate
    ```

5. **Start the Development Server:**

    ```bash
    python AthensMT/manage.py runserver
    ```

6. **Access the Application:**

    Open your browser and navigate to:  
    [http://localhost:8000/ODT/](http://localhost:8000/ODT/)  
    (Note: The base URL is defined as `/ODT/` in the settings and URLs.)

## Usage

1. **Upload CSV File:**
   - Navigate to the **Upload CSV** page.
   - Choose your main CSV file and, optionally, a configuration CSV.
   - Click **Upload & Continue**.

2. **Define Columns & Prompts:**
   - On the **Define Input & Output Columns** page, select the input columns from your CSV.
   - Define one or more output columns and specify corresponding prompt templates.
   - Click **Save & Continue**.

3. **Start Tagging:**
   - The tagging process will begin in the background.  
   - Monitor progress and view live logs on the **Tagging** page.

4. **View & Download Results:**
   - Once the process is complete, navigate to the **Results** page.
   - Preview the tagged CSV and download the tagged file along with detailed logs.

## Project Structure

```plaintext
ictashik-opendatatagger.git/
├── inint.t
└── AthensMT/
    ├── db.sqlite3
    ├── manage.py
    ├── AthensMT/
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    └── tagger_app/
        ├── __init__.py
        ├── admin.py
        ├── apps.py
        ├── forms.py
        ├── models.py
        ├── tests.py
        ├── urls.py
        ├── utils.py
        ├── views.py
        ├── migrations/
        │   └── __init__.py
        └── templates/
            ├── base.html
            ├── define_columns.html
            ├── results.html
            ├── tagging.html
            └── upload.html
            ```
    Configuration
        •	Django Settings:
    The project settings are managed in AthensMT/AthensMT/settings.py. Be sure to update ALLOWED_HOSTS, DEBUG, and other deployment settings as needed.
        •	Media Files:
    Uploaded CSV files and generated outputs (tagged CSV and logs) are stored in the media/ directory.
        •	LLM Integration:
    The integration with the local LLM is implemented in tagger_app/utils.py using an OpenAI-like client. Update the LLM_MODEL_NAME, API endpoint, or API key in this file to match your LLM server configuration.
        •	Caching:
    Django’s LocMemCache is used to track real-time LLM usage statistics and the progress of the tagging process.

    Contributing

    Contributions are welcome! If you have suggestions, bug reports, or improvements, please open an issue or submit a pull request.

    License

    This project is open source. [Include your license information here if applicable.]

    Acknowledgments
        •	Django
        •	Tailwind CSS
        •	OpenAI API
        •	Many thanks to all contributors and the open source community for their support and inspiration.