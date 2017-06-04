from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponseRedirect
from .forms import PathForm
from django.template import Context
from django.shortcuts import redirect
from QuestionGeneratorApp.QGenerationModel.Question_Generation_Using_NLP import QuestionGenerator as QG


# Create your views here.

class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)


def generate(request):
    form_class = PathForm

    if request.method == 'POST':
        form = form_class(data=request.POST)

        if form.is_valid():
            pathName = request.POST.get('path')

            execute_process = QG()
            questions = execute_process.generate(pathName)

            return render(request, 'questions.html', {'Questions': questions})
