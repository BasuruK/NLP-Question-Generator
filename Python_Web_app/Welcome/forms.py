from django import forms


class PathForm(forms.Form):
    path = forms.CharField(required=True)



