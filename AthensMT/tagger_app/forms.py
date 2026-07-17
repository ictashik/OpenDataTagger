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
    # Reference dataset(s) for retrieval-augmented tagging are handled
    # directly via request.FILES.getlist('reference_files') in the view —
    # Django's FileField doesn't support multiple files on its own, and any
    # number of files (0 or more) should be accepted here.