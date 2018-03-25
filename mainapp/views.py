from django.shortcuts import render, redirect
from .forms import QuestionForm



def home_screen(request):
	# if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = QuestionForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # processing the data
        	
            # redirect to the results page with the result:
            return redirect('result', {})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = QuestionForm()
    return render(request, 'mainapp/home_screen.html', {'form': form})


def result(request):
	pass