Below is the entire Markdown text enclosed in triple backticks. You can copy and paste this directly into your README.md file.

# OpenDataTagger

OpenDataTagger is an **AI-powered CSV tagging tool** built with Django. It allows users to upload CSV files, configure custom tagging prompts, and process data row-by-row using a local LLM (Llama3). The application provides real-time progress updates, detailed logs, and downloadable results—all wrapped in a modern, responsive UI powered by Tailwind CSS.

## Features

- **CSV Upload & Configuration:**  
  Upload a CSV file along with an optional configuration file that pre-defines tagging prompts.

- **Custom Column Definition:**  
  Select input columns and define output columns with custom AI prompt templates.

- **Background Tagging Process:**  
  The tagging process runs in a background thread, allowing you to monitor progress in real time.

- **Real-Time Progress & Logging:**  
  Get live updates and view detailed logs of AI responses (including the best answer and explanations) for each processed row.

- **Downloadable Results:**  
  Once tagging is complete, preview and download both the tagged CSV file and the detailed logs.

- **Local LLM Integration:**  
  Uses a locally hosted LLM (configured as Llama3) via an OpenAI-like API for generating tagging results.

- **Responsive UI:**  
  Clean and responsive interface built with Tailwind CSS and featuring a sidebar for easy navigation.

## Prerequisites

- **Python 3.8+**
- **Django 5.0.1** (or later)
- **Pandas**
- **OpenAI Python Library**
- A local LLM server running at `http://localhost:6000/v1` (configured for Llama3)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ictashik-opendatatagger.git
   cd ictashik-opendatatagger.git

	2.	Create a Virtual Environment & Activate It:

python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate


	3.	Install Dependencies:
If you have a requirements.txt file:

pip install -r requirements.txt

Otherwise, install manually:

pip install Django==5.0.1 pandas openai


	4.	Apply Migrations:

python AthensMT/manage.py migrate


	5.	Start the Development Server:

python AthensMT/manage.py runserver


	6.	Access the Application:
Open your browser and navigate to http://localhost:8000/ODT/.

Usage
	1.	Upload CSV File:
	•	Navigate to the Upload CSV page.
	•	Upload your main CSV file and, optionally, a configuration CSV file.
	•	Click Upload & Continue.
	2.	Define Columns & Prompts:
	•	On the Define Input & Output Columns page, select the input columns from your CSV.
	•	Define one or more output columns and specify corresponding prompt templates.
	•	Click Save & Continue.
	3.	Tagging Process:
	•	The tagging process starts in the background.
	•	Monitor the real-time progress and view detailed logs on the Tagging page.
	4.	View & Download Results:
	•	Once tagging is complete, navigate to the Results page.
	•	Preview the tagged CSV file and download both the tagged file and the logs.

Project Structure

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

Configuration
	•	Django Settings:
All configurations are managed in AthensMT/AthensMT/settings.py. Update ALLOWED_HOSTS, DEBUG, and other settings as necessary for your deployment environment.
	•	Media Files:
Uploaded CSV files and generated outputs (tagged CSV and logs) are stored in the media/ directory.
	•	LLM Integration:
The integration with the local LLM is implemented in tagger_app/utils.py using an OpenAI-like client. Update LLM_MODEL_NAME, the API endpoint, or the API key as needed.
	•	Caching:
Django’s LocMemCache is used to track real-time LLM usage statistics and the progress of the tagging process.

Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

License

This project is open source. Include your license information here if applicable.

Acknowledgments
	•	Django
	•	Tailwind CSS
	•	OpenAI API
	•	Special thanks to all contributors and the open source community.

If you still encounter formatting issues, try pasting the content into a plain text editor and then saving it as `README.md`.