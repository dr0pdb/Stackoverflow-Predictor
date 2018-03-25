from django import forms

class QuestionForm(forms.Form):
    title = forms.CharField(label='Title of question', max_length=100)
    reputation = forms.IntegerField(label='Your StackOverflow reputation', min_value = 0)
    deleted_questions = forms.IntegerField(label='Number of deleted questions', min_value = 0)
    body = forms.CharField(label='Body of the question', max_length=5000)


