from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from .forms import EmailUserCreationForm, EmailAuthForm
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .models import HealthMeasurement
from datetime import datetime


# Регистрация нового пользователя
def register_view(request):
    if request.method == 'POST':
        form = EmailUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # Сохраняем пользователя
            # Автоматически входим после регистрации
            login(request, user)
            messages.success(request, f'Вы успешно зарегистрированы!')
            return redirect('measurements')  # Перенаправление после регистрации
    else:
        form = EmailUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})


# Авторизация пользователя
def login_view(request):
    if request.method == 'POST':
        form = EmailAuthForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)

                # Если отмечено "Запомнить меня", устанавливаем длительную сессию (2 недели)
                if request.POST.get('remember_me'):
                    request.session.set_expiry(1209600)  # 2 недели в секундах
                else:
                    # Сессия будет длиться до закрытия браузера
                    request.session.set_expiry(0)

                # Перенаправляем на страницу измерений
                next_url = request.POST.get('next') or 'measurements'
                return redirect(next_url)
    else:
        # Добавляем параметр next из GET-запроса, если он есть
        initial = {}
        if 'next' in request.GET:
            initial['next'] = request.GET['next']
        form = EmailAuthForm(initial=initial)

    return render(request, 'registration/login.html', {
        'form': form,
        'next': request.GET.get('next', '')  # Передаем next в шаблон
    })


# Выход из аккаунта
def logout_view(request):
    logout(request)
    return redirect('index')


# Просмотр списка измерений (только для авторизованных)
@login_required
def measurements_view(request):
    measurements = HealthMeasurement.objects.filter(user=request.user).order_by('-date')
    return render(request, 'videodelivery/measurements.html', {
        'measurements': measurements
    })
