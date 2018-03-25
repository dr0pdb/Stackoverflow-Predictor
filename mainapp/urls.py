from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.home_screen, name='home_screen'),
    url(r'^/result$', views.result, name='result'),
]