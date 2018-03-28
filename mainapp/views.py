from django.shortcuts import render, redirect
from django.urls import reverse
from .forms import QuestionForm
from .utils import Question



def home_screen(request):
	# if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = QuestionForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # processing the data
            data = form.cleaned_data
            title = data['title']
            reputation = data['reputation']
            deleted_questions = data['deleted_questions']
            body = data['body']
            question = Question(title, reputation, deleted_questions, body)
            outcome = question.predict_outcome()
            # redirect to the results page with the result:
            return render(request, 'mainapp/result.html', {'outcome' : outcome})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = QuestionForm()
    return render(request, 'mainapp/home_screen.html', {'form': form})
