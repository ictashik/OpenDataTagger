# tagger_app/forms.py
from django import forms

class UploadForm(forms.Form):
    csv_file = forms.FileField(
        required=True,
        label="Main CSV File"
    )
    config_file = forms.FileField(
        required=False,
        label="Config File (Optional)"
    )