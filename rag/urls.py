from django.urls import path

from . import views

urlpatterns = [
    path("", views.hello, name="hello"),
    path("upload/", views.upload_files, name="file_upload"),
    path("query/", views.query, name="query"),
    
]