#Welcome/urls.py

from django.conf.urls import url
from Welcome import views

urlpatterns = [
    url(r'^$', views.HomePageView.as_view()),

    url(r'^generate/$', views.generate, name='generate'),
]
