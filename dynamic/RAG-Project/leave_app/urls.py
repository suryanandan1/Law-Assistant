from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_page, name='chat'),
    path('login/', views.login_page, name='login'),
    path('signup/', views.signup_page, name='signup'),
    path('logout/', views.logout_page, name='logout'),
]
