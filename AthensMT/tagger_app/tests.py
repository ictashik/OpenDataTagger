from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.cache import cache
import json
import os
from django.conf import settings
import pandas as pd

class APITests(TestCase):
    def setUp(self):
        self.client = Client()
        # Create test CSV file
        self.csv_content = b"col1,col2\nvalue1,value2\n"
        self.csv_file = SimpleUploadedFile("test.csv", self.csv_content, content_type="text/csv")
        
        # Create test config file
        self.config_content = b"OutputColumn,PromptTemplate\noutput1,template1\n"
        self.config_file = SimpleUploadedFile("test_config.csv", self.config_content, content_type="text/csv")

    def tearDown(self):
        # Clean up test files
        test_files = ['test.csv', 'test_config.csv', 'test_tagged.csv', 'test_logs.csv']
        for file in test_files:
            file_path = os.path.join(settings.MEDIA_ROOT, file)
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_llm_status_view(self):
        response = self.client.get(reverse('llm_status'))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn('models', data)

    def test_set_selected_model(self):
        # Test valid model selection
        data = {'selected_model': 'test-model'}
        response = self.client.post(
            reverse('set_selected_model'),
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session['LLM_MODEL_NAME'], 'test-model')

        # Test invalid request
        response = self.client.post(
            reverse('set_selected_model'),
            data='invalid json',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_file_view(self):
        # Test successful upload
        response = self.client.post(reverse('upload_file'), {
            'csv_file': self.csv_file,
            'config_file': self.config_file
        })
        self.assertEqual(response.status_code, 302)  # Redirect status
        self.assertTrue(os.path.exists(os.path.join(settings.MEDIA_ROOT, 'test.csv')))
        self.assertTrue(os.path.exists(os.path.join(settings.MEDIA_ROOT, 'test_config.csv')))

        # Test upload without config file
        csv_only = SimpleUploadedFile("test2.csv", self.csv_content, content_type="text/csv")
        response = self.client.post(reverse('upload_file'), {'csv_file': csv_only})
        self.assertEqual(response.status_code, 302)

    def test_define_columns_view(self):
        # Setup: Upload file first
        self.client.post(reverse('upload_file'), {
            'csv_file': self.csv_file,
            'config_file': self.config_file
        })

        # Test GET request
        response = self.client.get(reverse('define_columns'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'define_columns.html')

        # Test POST request
        post_data = {
            'input_columns': ['col1'],
            'output_column': ['new_col'],
            'prompt_template': ['test template']
        }
        response = self.client.post(reverse('define_columns'), post_data)
        self.assertEqual(response.status_code, 302)  # Redirect to tagging

    def test_tagging_progress_view(self):
        # Setup: Create a mock session key
        session = self.client.session
        session['tagging_session_key'] = 'test-session'
        session.save()

        # Test without progress data
        response = self.client.get(reverse('tagging_progress'))
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertIn('error', data)

    def test_results_view(self):
        # Setup: Create mock tagged file
        tagged_df = pd.DataFrame({'col1': ['value1'], 'col2': ['value2']})
        tagged_file = os.path.join(settings.MEDIA_ROOT, 'test_tagged.csv')
        tagged_df.to_csv(tagged_file, index=False)

        # Set up session and cache
        session_key = 'test-session'
        session = self.client.session
        session['tagging_session_key'] = session_key
        session.save()

        # Set up cache with tagged file path
        cache.set(f"tagged_file_{session_key}", tagged_file)
        cache.set(f"logs_file_{session_key}", os.path.join(settings.MEDIA_ROOT, 'test_logs.csv'))

        # Test results view
        response = self.client.get(reverse('results'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'results.html')
