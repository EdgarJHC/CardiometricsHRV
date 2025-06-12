from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import AuthenticationForm, PasswordResetForm
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.core.mail import send_mail
from django.utils.encoding import force_bytes
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from .models import *
from .forms import *
from django.db.models import Q
import pandas as pd 
from datetime import datetime
from django.http import JsonResponse
import math

from weasyprint import HTML
from django.http import HttpResponse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image


from django.template.loader import render_to_string

import numpy as np
from scipy.signal import firwin, lfilter, find_peaks, welch
from scipy.interpolate import interp1d
from pyhrv import time_domain
from io import BytesIO
import base64
import json
import plotly
from django.core.serializers.json import DjangoJSONEncoder
from biosppy.signals import ecg
from django.utils import timezone

from biosppy.signals.ecg import engzee_segmenter, correct_rpeaks
from django.views.decorators.csrf import csrf_exempt
from random import random
import string




def signup(request):
    if request.method == 'GET':
        departamentos = Departamento.objects.all()
        form = UserRegistrationForm()
        return render(request, 'signup.html', {"form": form, "departamentos": departamentos})
    else:
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            password1 = form.cleaned_data.get('password1')
            password2 = form.cleaned_data.get('password2')
            if password1 == password2:
                try:
                    # Intento de crear el usuario
                    user = form.save(commit=False)
                    user.set_password(password1)  # Encripta la contraseña antes de guardar el usuario
                    user.save()

                    # Recoge los datos del formulario adicionales
                    nombre_especialista = request.POST.get("nombres")
                    apellido_paterno = request.POST.get("apellido_paterno")
                    apellido_materno = request.POST.get("apellido_materno")
                    telefono = request.POST.get("telefono")
                    correo = request.POST.get("correo")
                    especialidad = request.POST.get("especialidad")
                    departamento_id = request.POST.get("departamento")
                    fecha_nacimiento = request.POST.get("fecha_nacimiento")

                    # Validación de que la fecha de nacimiento no sea nula
                    if not fecha_nacimiento:
                        return render(request, 'signup.html', {
                            "form": form,
                            "departamentos": Departamento.objects.all(),
                            "error": "Por favor, proporciona una fecha de nacimiento válida."
                        })

                    # Obtiene la instancia de Departamento usando el ID proporcionado
                    departamento = Departamento.objects.get(id_departamento=departamento_id)

                    # Crea el objeto Especialista
                    especialista = Especialista.objects.create(
                        user=user,
                        nombre_especialista=nombre_especialista,
                        apellido_paterno=apellido_paterno,
                        apellido_materno=apellido_materno,
                        telefono=telefono,
                        correo=correo,
                        especialidad=especialidad,
                        fecha_nacimiento=fecha_nacimiento,
                        departamento_id=departamento  # Ahora asigna la instancia de Departamento
                    )
                    especialista.save()

                    login(request, user)
                    return redirect('homeDoctor')
                except IntegrityError as e:
                    print(e)  # Esto imprimirá la excepción completa en la consola.
                    if 'UNIQUE constraint' in str(e):
                        error_message = "El usuario ya existe. Prueba con otro nombre de usuario."
                    else:
                        error_message = f"Ocurrió un error durante el registro: {e}."  # Muestra el error específico
                    return render(request, 'signup.html', {
                        "form": form,
                        "departamentos": Departamento.objects.all(),
                        "error": error_message
                    })

            else:
                # Si las contraseñas no coinciden
                return render(request, 'signup.html', {
                    "form": form,
                    "departamentos": Departamento.objects.all(),
                    "error": "Las contraseñas no coinciden."
                })
        
        # Si el formulario no es válido
        return render(request, 'signup.html', {
            "form": form,
            "departamentos": Departamento.objects.all(),
            "error": "Por favor corrige los errores del formulario."
        })


def forgot_password(request):
    print("Vista llamada")
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            print(f"Correo recibido: {email}")
            
            try:
                # Buscar al especialista usando el correo
                especialista = Especialista.objects.get(correo=email)
                
                # Obtener el usuario asociado al especialista
                user = especialista.user
                
                # Si el correo es válido, generar el enlace para restablecer la contraseña
                subject = 'Restablecimiento de contraseña CardioMetricsHRV'  # Asunto del correo
                message = f'Hola {user.username}, haz clic en el siguiente enlace para restablecer tu contraseña.'
                
                # Generar el enlace de restablecimiento (usando la URL absoluta)
                reset_link = request.build_absolute_uri(f'/reset-password/{user.id}/')
                message += f"\n\n{reset_link}"
                
                # Enviar el correo
                send_mail(subject, message, 'no-reply@miweb.com', [email])
                
                # Mensaje de éxito, sin importar si el correo estaba registrado o no
                messages.success(request, "Si el correo ingresado está asociado a una cuenta, te hemos enviado un correo para restablecer tu contraseña.")
                return redirect('forgot_password')  # Redirigir a la misma página después de enviar el correo

            except Especialista.DoesNotExist:
                # No hacer nada si el especialista no existe, solo mostrar el mensaje genérico
                messages.success(request, "Si el correo ingresado está asociado a una cuenta, te hemos enviado un correo para restablecer tu contraseña.")
                return redirect('forgot_password')  # Redirigir a la misma página después de intentar

        else:
            return render(request, 'forgot_password.html', {'form': form})

    else:
        form = PasswordResetForm()

    return render(request, 'forgot_password.html', {'form': form})
def forgot_password(request):
    if request.method == 'POST':
        # Lógica para manejar el restablecimiento de la contraseña, por ejemplo,
        # enviar un correo de restablecimiento.
        pass
    return render(request, 'forgot_password.html', {})


# Vista para mostrar la página de inicio
def home(request):
    if request.user.is_authenticated:
        # Primero, intenta obtener el perfil de Especialista para el usuario logeado.
        especialista = Especialista.objects.filter(user=request.user).first()

        if especialista is None:
            # Si el usuario NO tiene un perfil de Especialista:
            print(f"DEBUG: Usuario '{request.user.username}' logeado pero sin perfil de Especialista. Mostrando formulario en home.")

            if request.method == 'POST':
                form = EspecialistaForm(request.POST)
                if form.is_valid():
                    especialista_perfil = form.save(commit=False)
                    especialista_perfil.user = request.user
                    especialista_perfil.save()
                    # Si el perfil se crea correctamente, redirige a homeDoctor
                    return redirect('homeDoctor')
            else:
                form = EspecialistaForm() # Crea un formulario vacío para el GET

            # Renderiza home.html y pasa el formulario para que el usuario pueda crear su perfil.
            # 'home.html' debe mostrar este formulario.
            return render(request, 'home.html', {'form': form, 'show_create_profile_form': True})
        else:
            # Si el usuario SÍ tiene un perfil de Especialista, redirige a homeDoctor.
            print(f"DEBUG: Usuario '{request.user.username}' logeado con perfil de Especialista. Redirigiendo a homeDoctor.")
            return redirect('homeDoctor')
    else:
        # Si el usuario no está autenticado, muestra la página de inicio normal.
        print("DEBUG: Usuario no autenticado. Mostrando home normal.")
        return render(request, 'home.html', {'show_create_profile_form': False})


@login_required
def homeDoctor(request):
    # Aquí, en teoría, el especialista ya existe o el usuario no habría llegado aquí.
    # Si llega aquí y no tiene perfil, sería un caso excepcional (ej. acceso directo a URL sin pasar por home)
    # y puedes añadir un fallback. Usar .get() es seguro si confías en el flujo de 'home'.
    try:
        especialista = Especialista.objects.get(user=request.user)
    except Especialista.DoesNotExist:
        print("DEBUG: ¡ERROR INESPERADO! Usuario logeado en homeDoctor sin perfil de Especialista. Redirigiendo a home.")
        return redirect('home') # Fallback para evitar el error 500

    return render(request, 'homeDoctor.html', {'especialista': especialista})
    
# Vista para el inicio de sesión
def signin(request):
    
    if request.method == 'GET':
        return render(request, 'signin.html', {"form": AuthenticationForm()})
    else:
        user = authenticate(
            request, username=request.POST['username'], password=request.POST['password1'])
        if user is None:
            return render(request, 'signin.html', {"form": AuthenticationForm(), "error": "Nombre de usuario o contraseña incorrectos."})

        login(request, user)
        return redirect('homeDoctor')

# Vista para cerrar la sesión de un usuario
@login_required
def signout(request):
    logout(request)
    return redirect('home')  # Redirige a la URL 'home', que apunta a la vista 'home' definida en urls.py


# Vista para mostrar los pacientes pendientes (no completados)
@login_required
def pacientes(request):
    query = request.GET.get('query', '')  # Captura el parámetro de búsqueda desde la URL
    especialista = request.user.especialista  # Obtiene el especialista asociado al usuario actual

    if query:
        # Filtra pacientes del especialista actual que coincidan con el criterio de búsqueda
        pacientes = Paciente.objects.filter(
            Q(especialista=especialista) &
            (
                Q(nombre_paciente__icontains=query) |
                Q(apellido_paterno__icontains=query) |
                Q(apellido_materno__icontains=query) |
                Q(id_paciente__icontains=query) |
                Q(sexo__icontains=query) |
                Q(correo__icontains=query)
            )
        )
    else:
        # Si no hay búsqueda, muestra todos los pacientes del especialista actual
        pacientes = Paciente.objects.filter(especialista=especialista)

    return render(request, 'paciente.html', {"pacientes": pacientes})



def buscar_registro(request):
    registro_busqueda = request.GET.get('registro_busqueda', '')  # Captura el parámetro de búsqueda desde la URL
    paciente = request.user.Paciente  # Obtiene el especialista asociado al usuario actual

    if registro_busqueda:
        # Filtra pacientes del especialista actual que coincidan con el criterio de búsqueda
        registro = ECG.objects.filter(
                Q(registro=registro) &
            (
                Q(homoclave__icontains=registro_busqueda) |
                Q(fecha_informe__icontains=registro_busqueda) |
                Q(apellido_materno__icontains=registro_busqueda) 
            )
        )
    else:
        # Si no hay búsqueda, muestra todos los pacientes del especialista actual
        pacientes = Paciente.objects.filter(paciente=paciente)

    return render(request, 'paciente.html', {"pacientes": pacientes})

    


@login_required
def editar_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)  # Cambiado a id_paciente
    if request.method == 'POST':
        form = PacienteForm(request.POST, instance=paciente)
        if form.is_valid():
            form.save()
            return redirect('pacientes')
    else:
        form = PacienteForm(instance=paciente)
    return render(request, 'editar_paciente.html', {'form': form})

'''En la plantilla editar.html, puedes usar el formulario de la siguiente manera:
#{'Formulario': Formulario} significa que dentro de la plantillaeditar.html, puedes acceder a la instancia del formulario con la variable Formulario.'''


@login_required
def eliminar_paciente(request, paciente_id):
    pacientes = get_object_or_404(Paciente, id_paciente=paciente_id) # referencia al campo de la clase 
    pacientes.delete() # Elimina el paciente
    return redirect('pacientes') # Redirige a la lista de pacientes

@login_required
def historial(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    registros_ecg = ECG.objects.filter(paciente=paciente)  # Filtrar por el paciente específico
    
    # Verificación de si hay registros
    no_registros = registros_ecg.count() == 0
    
    # Depuración: mostrar en la consola si no hay registros
    print(f"No hay registros para el paciente {paciente_id}: {no_registros}")
    
    return render(request, 'historial.html', {'paciente': paciente, 'registros_ecg': registros_ecg, 'no_registros': no_registros})


@login_required
def buscar(request):
    query = request.GET.get('query', '')
    pacientes = Paciente.objects.filter(
        Q(nombre_paciente__icontains=query) |
        Q(apellido_paterno__icontains=query)
    ) if query else []
    
    return render(request, 'paciente.html', {'pacientes': pacientes, 'query': query})

# Vista para crear un nuevo paciente
@login_required
def create_paciente(request):
    if request.method == "GET":
        return render(request, 'create_paciente.html', {"form": PacienteForm()})
    else:
        form = PacienteForm(request.POST)
        if form.is_valid():
            try:
                # Buscar el especialista vinculado al usuario actual
                especialista = Especialista.objects.get(user=request.user)
                
                # Obtener los datos del formulario (sin usuario y contraseña)
                nombre_paciente = form.cleaned_data['nombre_paciente']
                apellido_paterno = form.cleaned_data['apellido_paterno']
                apellido_materno = form.cleaned_data['apellido_materno']
                telefono = form.cleaned_data['telefono']
                correo = form.cleaned_data['correo']
                sexo = form.cleaned_data['sexo']
                fecha_nacimiento = form.cleaned_data['fecha_nacimiento']

                # Crear el usuario para el paciente (sin usuario_paciente y contrasenia_paciente)
                user = User.objects.create_user(username=nombre_paciente, password="defaultpassword")  # Aquí puedes elegir una lógica para el password
                user.save()

                # Crear el paciente, asignando el usuario creado y el especialista actual
                new_paciente = form.save(commit=False)
                new_paciente.user = user  # Vincula el paciente con el usuario creado
                new_paciente.especialista = especialista  # Vincula el paciente con el especialista actual
                new_paciente.save()

                # Redirigir a la página de pacientes
                return redirect('pacientes')

            except Especialista.DoesNotExist:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": "El especialista no está registrado. Por favor, verifica tu cuenta."
                })
            except ValueError:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": "Surgió un error al crear al paciente."
                })
            except Exception as e:
                return render(request, 'create_paciente.html', {
                    "form": form,
                    "error": f"Hubo un error: {str(e)}"
                })
        
        # Si el formulario no es válido, se muestra el error
        print(form.errors)
        return render(request, 'create_paciente.html', {
            "form": form,
            "error": "Por favor corrige los errores del formulario."
        })

@login_required
def crear_informe(request, paciente_id):
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    Formulario = ExpedienteForm(request.POST or None, request.FILES or None)  # Cargar los datos del formulario

    if request.method == 'POST':  # Solo manejamos el archivo cuando sea POST
        if Formulario.is_valid():
            # Validación adicional para el archivo
            archivo = request.FILES.get('archivo_ecg')
            if archivo:
                if not (archivo.name.endswith('.txt') or archivo.name.endswith('.csv')):
                    Formulario.add_error('archivo_ecg', 'El archivo debe ser de tipo .txt o .csv.')
                else:
                    expediente = Formulario.save(commit=False)  # No guarda todavía
                    expediente.paciente = paciente  # Relaciona el paciente
                    expediente.save()  # Ahora guarda el informe
                    return redirect('historial', paciente_id=paciente.id_paciente)
            else:
                Formulario.add_error('archivo_ecg', 'Debes seleccionar un archivo.')
        else:
            print(Formulario.errors)  # Mostrar los errores si los hay

    # Si es una solicitud GET o si hay errores en el formulario, renderizamos de nuevo el formulario
    return render(request, 'crear_informe.html', {'Formulario': Formulario, 'id_paciente': paciente.id_paciente})

@login_required
def eliminar_informe(request, paciente_id):
    ecg = get_object_or_404(ECG, id_ecg=paciente_id) # referencia al campo de la clase 
    ecg.delete() # Elimina el paciente
    return redirect('historial', paciente_id=ecg.paciente.id_paciente ) # Redirige a la lista de pacientes

import pandas as pd

def formato_ecg(file):
    delimitadores = [r"\s+", "\t", ",", " ", ";"]  # Lista de posibles delimitadores
    #datos_ecg = None  # Variable para almacenar los datos leídos
    
    # Intentamos leer el archivo con cada delimitador
    for delim in delimitadores:
        try:
            # Leer el archivo con el delimitador actual, sin encabezado
            datos_ecg_temp = pd.read_csv(file, sep=delim, header=None)
            
            # Verificar si el DataFrame no está vacío
            if not datos_ecg_temp.empty:
                datos_ecg = datos_ecg_temp
                print(f"Datos leídos con el delimitador '{delim}':")
                print(datos_ecg.head())  # Mostrar las primeras filas para verificar
                break  # Si se logra leer correctamente, salimos del bucle
        
        except pd.errors.EmptyDataError:
            print(f'El archivo está vacío (con delimitador "{delim}").')
            return None
        except pd.errors.ParserError:
            print(f'Error de análisis al leer el archivo con el delimitador "{delim}".')
            continue  # Continuamos con el siguiente delimitador si ocurre un error
        except Exception as e:
            print(f'Error desconocido con delimitador "{delim}": {str(e)}')
            continue  # Continuamos con el siguiente delimitador si ocurre un error
    
    # Si no se pudo leer el archivo correctamente con ningún delimitador, notificamos al usuario
    if datos_ecg is None:
        print(f'Formato de archivo no soportado o error de lectura: {file.name}')
        return None

    # Verificar el número de columnas del DataFrame después de intentar con todos los delimitadores
    if datos_ecg.shape[1] >= 2:
        print(f'Archivo con {datos_ecg.shape[1]} columnas. Procesado exitosamente.')
        return datos_ecg
    elif datos_ecg.shape[1] == 1:
        print('Archivo con 1 columna. Procesado exitosamente.')
        return datos_ecg
    else:
        print(f'El archivo tiene un número inesperado de columnas: {datos_ecg.shape[1]}')
        return None


def convertir_tipos(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convertir arrays a listas
    elif isinstance(obj, np.float64) or isinstance(obj, np.int64):
        return obj.item()  # Convertir números de numpy a tipos estándar de Python
    elif isinstance(obj, dict):
        return {k: convertir_tipos(v) for k, v in obj.items()}  # Convertir diccionarios recursivamente
    elif isinstance(obj, (tuple, list)):
        return [convertir_tipos(i) for i in obj]  # Convertir listas o tuplas recursivamente
    else:
        return obj  # Mantener los demás tipos igual

def filtro_pasa_bajas(v, fs):
    señal_corregida = v - np.mean(v)
    low_cutoff = 10
    high_cutoff = 30
    coeficientes = firwin(101, [low_cutoff, high_cutoff], fs=fs, window='hamming', pass_zero=False)
    V_FIL =  lfilter(coeficientes, 1.0, señal_corregida)

    return (V_FIL)

def algoritmo_segunda_derivada(v):#tiene entrada la señal filtrada
    coeficientes = np.array([1,1,1,1,-10,1,1,1,1])
    ECG_A = 4 * (np.convolve(v, coeficientes, mode= 'same' ))
    UM = 0
    return (ECG_A*ECG_A), UM

def detección_RR(V, T, FC):
    output = engzee_segmenter(signal= V, sampling_rate= FC, threshold= 0.6)
    rpeaks =  output['rpeaks']
    rpeaks_corr = correct_rpeaks(signal= V, rpeaks=rpeaks, sampling_rate= FC, tol = 0.05)
    rpeaks_corr = rpeaks_corr['rpeaks']
    intervalosRR = np.diff(T[rpeaks_corr])
    umbral_segundos = 300/1000
    picos_validos = [rpeaks_corr[0]]
    for i in range(1, len(rpeaks_corr)):
        if intervalosRR[i-1] >= umbral_segundos:
            picos_validos.append(rpeaks_corr[i])

    return rpeaks_corr, intervalosRR*1000#ya salen en ms
#Lo necesitamos en ms para poder analizarlo ene el tiempo

def Parametro_dominio_tiempo(intervalos_RR):
    resultadosDT = time_domain.time_domain(intervalos_RR)
    print(f'Los resultados del analisis en el dominio del tiempo: {resultadosDT}')
    resultadosDT_obj = {
        "nni_mean": resultadosDT[1],
        "nni_min": resultadosDT[2],
        "nni_max": resultadosDT[3],
        "hr_mean": resultadosDT[4],
        "hr_min": resultadosDT[5],
        "hr_max": resultadosDT[6],
        "hr_std": resultadosDT[7],
        "nni_diff_mean": resultadosDT[8],
        "nni_diff_min": resultadosDT[9],
        "nni_diff_max": resultadosDT[10],
        "sdnn": resultadosDT[11],
        "sdann": resultadosDT[13],
        "rmssd": resultadosDT[14],
        "sdsd": resultadosDT[15],
        "nn50": resultadosDT[16],
        "pnn50": resultadosDT[17],
        "nn20": resultadosDT[18],
        "pnn20": resultadosDT[19],
    }

    resultadosDT = tuple(convertir_tipos(i) for i in resultadosDT)
    print(type(resultadosDT))  # Debería decir <class 'tuple'>
    print(len(resultadosDT))   # Ver cuántos elementos tiene la tupla

    for i, item in enumerate(resultadosDT):
        print(f"Elemento {i}: Tipo {type(item)}")
    SDNN = 0


   # print(resultadosDT)
    return resultadosDT_obj, SDNN


def Parametro_dominio_frecuencia(RR, intervalos_RR, T, TipoArchivo):#Picos RR son el parametro RR
    #El segmento T nunca fue delimitado. 
    print('Se estan realizando los calculos del analisis en el dominio de la frecuencia')
    print('----------------------------------------------------------------------------')

    RR = np.array(RR, dtype = int)#convertirlo en un array
    intervalos_RR = np.array(intervalos_RR, dtype = int)
    T = np.array(T, dtype =  float)
    #print(f'El numero de puntos puntos en el tiempo son {T}')
    #print(f'El numero de intervalos RR son {intervalos_RR}')
    #llegan chidos
    fs = 4
    if TipoArchivo == 0:
    #Interpolar el Tacograma para Obtener un Muestreo Regular
        #nosotros ya tenemos en un tacograma T_RR
        
        T_RR = T[np.array(RR, dtype = int)]

        print(f'El numero de puntos puntos en el tiempo son {len(T_RR)}')
        print(f'El numero de intervalos RR son {len(intervalos_RR)}')
        tiempo_regular = np.arange(T_RR[0],T[RR][-1], 1 /fs )

        interp_func = interp1d(T_RR, intervalos_RR, kind='cubic')
        rr_uniforme = interp_func(tiempo_regular)
        print(f'Se realizo la interpolación correctamente ')

        return fs, tiempo_regular, rr_uniforme
    elif TipoArchivo == 1:
        RR = (RR / 1000)
        #RR = RR + np.random.normal(0, 0.001, size=RR.shape)
       # print('Se realizara un analisis con un procesamiento para el tacograma')
       # print(f'El numero de puntos puntos en el tiempo son {len(T)}')
       #¿ print(f'El numero de intervalos RR son {len(intervalos_RR)}')      
       ###Si hay valores RR duplicados, significa que dos o más latidos tuvieron el mismo intervalo RR.
        #El RR que es el tiempo ya esta e segundos y el los intervalos estan en ms. 
        tiempo_regular = np.arange(RR[0], RR[-1], 0.25)
        interp_func = interp1d(RR, intervalos_RR, kind='cubic')
        rr_uniforme = interp_func(tiempo_regular)
        return fs, tiempo_regular, rr_uniforme

 

def calcular_potencia_banda(frecuencias, psd, banda):
    indices = np.where((frecuencias >= banda[0]) & (frecuencias <= banda[1]))[0]
    potencia = np.trapz(psd[indices], frecuencias[indices])
    frecuencia_pico = frecuencias[indices][np.argmax(psd[indices])] if indices.size > 0 else 0

    return potencia, frecuencia_pico

# Cálculo de parámetros de dominio de frecuencia usando Welch
def calculo_welch(RR, FS):

    # Estimación PSD con el método de Welch
    frecuencias_welch, psd = welch(RR, fs=FS, window='hamming', nperseg=1200, noverlap=600, nfft=1200, detrend = 'linear')


    # Definición de bandas de frecuencia
    banda_vlf = (0.0033, 0.04)
    banda_lf = (0.04, 0.15)
    banda_hf = (0.15, 0.4)
    
    # Calcular potencia y frecuencia pico en cada banda, d
    potencia_vlf, pico_vlf = calcular_potencia_banda(frecuencias_welch, psd, banda_vlf)
    potencia_lf, pico_lf = calcular_potencia_banda(frecuencias_welch, psd, banda_lf)
    potencia_hf, pico_hf = calcular_potencia_banda(frecuencias_welch, psd, banda_hf)


    # Potencia total
    potencia_total = potencia_vlf + potencia_lf + potencia_hf
    
    # Calcular el porcentaje de cada banda
    porcentaje_vlf = (potencia_vlf / potencia_total) * 100
    porcentaje_lf = (potencia_lf / potencia_total) * 100
    porcentaje_hf = (potencia_hf / potencia_total) * 100
    
    # Potencia en unidades normalizadas
    potencia_lf_nu = (potencia_lf / (potencia_lf + potencia_hf)) * 100
    potencia_hf_nu = (potencia_hf / (potencia_lf + potencia_hf)) * 100
    
    # Cociente LF/HF
    lf_hf_ratio = potencia_lf / potencia_hf if potencia_hf != 0 else 0
    
    # Potencia en escala logarítmica
    potencia_vlf_log = np.log(potencia_vlf) if potencia_vlf > 0 else 0
    potencia_lf_log = np.log(potencia_lf) if potencia_lf > 0 else 0
    potencia_hf_log = np.log(potencia_hf) if potencia_hf > 0 else 0



    # Imprimir resultados
    print(f'VLF - Potencia: {potencia_vlf:.4f} ms², Frecuencia Pico: {pico_vlf:.4f} Hz, Potencia (log): {potencia_vlf_log:.4f}, Porcentaje: {porcentaje_vlf:.2f}%')
    print(f'LF - Potencia: {potencia_lf:.4f} ms², Frecuencia Pico: {pico_lf:.4f} Hz, Potencia (log): {potencia_lf_log:.4f}, Porcentaje: {porcentaje_lf:.2f}%')
    print(f'HF - Potencia: {potencia_hf:.4f} ms², Frecuencia Pico: {pico_hf:.4f} Hz, Potencia (log): {potencia_hf_log:.4f}, Porcentaje: {porcentaje_hf:.2f}%')
    print(f'Potencia total: {potencia_total:.4f} ms²')
    print(f'Potencia LF (nu): {potencia_lf_nu:.2f}')
    print(f'Potencia HF (nu): {potencia_hf_nu:.2f}')
    print(f'Ratio LF/HF: {lf_hf_ratio:.4f}')

    # Retornar los valores calculados si deseas usarlos en otros lugares
    return {
        "potencia_vlf": float(potencia_vlf),
        "pico_vlf": float(pico_vlf),
        "potencia_vlf_log": float(potencia_vlf_log),
        "porcentaje_vlf": float(porcentaje_vlf),
        "potencia_lf": float(potencia_lf),
        "pico_lf": float(pico_lf),
        "potencia_lf_log": float(potencia_lf_log),
        "porcentaje_lf": float(porcentaje_lf),
        "potencia_hf": float(potencia_hf),
        "pico_hf": float(pico_hf),
        "potencia_hf_log": float(potencia_hf_log),
        "porcentaje_hf": float(porcentaje_hf),
        "potencia_total": float(potencia_total),
        "potencia_lf_nu": float(potencia_lf_nu),
        "potencia_hf_nu": float(potencia_hf_nu),
        "lf_hf_ratio": float(lf_hf_ratio)
    }, frecuencias_welch, psd



def reemplazar_nan_y_convertir(data):
    if isinstance(data, dict):
        return {k: reemplazar_nan_y_convertir(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [reemplazar_nan_y_convertir(item) for item in data]
    # Para enteros de numpy
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    # Para flotantes de numpy o float nativos
    elif isinstance(data, (np.float64, np.float32, float)):
        return None if math.isnan(data) else float(data)
    else:
        return data
def tiempoHMS(segundos):
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segundos_restantes  = int(segundos % 60) 
    return f"{horas:02}:{minutos:02}:{segundos_restantes:02}"


# Vista para marcar un paciente como completado
@login_required
def complete_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, pk=paciente_id, user=request.user)
    if request.method == 'POST':
        paciente.fecha_nacimiento = timezone.now()
        paciente.save()
        return redirect('pacientes')


# Vista del perfil del especialista
@login_required
def perfil_doc(request):
    user = request.user
    try:
        doctor = Especialista.objects.get(user=user)
        context = {
            'nombre_especialista': doctor.nombre_especialista,
            'apellido_paterno': doctor.apellido_paterno,
            'apellido_materno': doctor.apellido_materno,
            'fecha_nacimiento': doctor.fecha_nacimiento,
            'departamento': doctor.departamento_id,  # Cambié a la forma correcta
            'username': user.username,
        }
        return render(request, 'perfil_especialista.html', context)
    except Especialista.DoesNotExist:
        return redirect('error_page')

def error_page(request):
    return render(request, 'error_page.html')


@login_required
@csrf_exempt
def ver_grafico(request, ecg_id):
    banda = float(request.GET.get('banda', 300))  # Valor predeterminado de 10000 muestras
    banda_tacograma_segundos = float(request.GET.get('banda_tacograma', 300))  # Valor predeterminado para tacograma
    tiempo_requerido_ms  = banda_tacograma_segundos * 1000
    banda_analisis = int(request.GET.get('banda_analisis', 300)) #Análisis por default de 300 segundos
    try:
        registro_ecg = get_object_or_404(ECG, id_ecg = ecg_id)
        ecg_id = registro_ecg.id_ecg
        paciente = registro_ecg.paciente
        tipo_registro = registro_ecg.tipo_archivo
        especialista = paciente.especialista
    except Exception as e:
        print(f"Error al obtener el registro ECG: {e}")
        return render(request, 'ver_grafico.html', {'mensaje': 'No se pudo encontrar el registro ECG.'})
    ecg_path = registro_ecg.archivo_ecg.path
    datos_ecg = formato_ecg(ecg_path)
    print(datos_ecg)

    if datos_ecg is None:
        return render(request, 'ver_grafico.html', {'mensaje': 'El archivo ECG no pudo ser leído.'})
       # return render(request, 'ver_grafico.html', {'mensaje': 'El archivo ECG no tiene la estructura correcta.'})
    if tipo_registro  == 'Electrocardiograma Digital' or tipo_registro == 'ECG':
        TipoArchivo = 0
        print('Este archivo ha sido seleccionado como un Electrocardiograma')
        datos_ecg.iloc[:, 0] = pd.to_numeric(datos_ecg.iloc[:, 0], errors='coerce')
        datos_ecg.iloc[:, 1] = pd.to_numeric(datos_ecg.iloc[:, 1], errors='coerce')
        # 0.000	2.331
        tiempo_ECG = datos_ecg.iloc[:, 0]
        voltaje_ECG = datos_ecg.iloc[:, 1]
        print(f'La linea de tiempo ECG = {tiempo_ECG}')
        print(f'La linea de voltaje ECG = {voltaje_ECG}') 
        request.session['tiempo_ECG'] = tiempo_ECG.tolist()
        request.session['voltaje_ECG'] = voltaje_ECG.tolist()
        fm = 1 /(tiempo_ECG.iloc[1]-tiempo_ECG.iloc[0])
        Periodo = 1/fm
        #######################
        voltaje_F = filtro_pasa_bajas(voltaje_ECG, fm)#
        voltaje, umbral = algoritmo_segunda_derivada(voltaje_F)#
        picosRR, intervalosRR = detección_RR(voltaje_F, tiempo_ECG, fm)
        request.session['picosRR'] = picosRR.tolist()
        request.session['intervalosRR'] = intervalosRR.tolist()
        print(f'Picos entre 1000 {picosRR}')
        print(f'Los intervalos totales son {intervalosRR}')
        print(f'Los picos totales son {picosRR}')
      #  print(f'Tiempo_Total {tiempo_ECG}')
        tiempoInicio =tiempo_ECG.iloc[0]
        tiempoFin = tiempo_ECG.iloc[-1]
        duracionECG = tiempoFin - tiempoInicio
        print(f'El electrocardiograma duro {duracionECG} ms')
        print('**********************************************')
       # fs, TR, RRuniform = Parametro_dominio_frecuencia(picosRR, intervalosRR, tiempo_ECG, TipoArchivo)
       # resultadosDF, frecuencias_welch, psd = calculo_welch(RRuniform, fs)
       # request.session['Frecuencias_welch'] = frecuencias_welch.tolist()
       # request.session['PSD'] = psd.tolist()

    else:
        TipoArchivo = 1
        print('Archivo con una columna detectado')
        intervalosRR = datos_ecg.iloc[:, 0].astype(float).to_numpy()#Cada uno de los intervalos RR
        #Si lo divido entre mil Expect x to not have duplicates
        picosRR = np.cumsum(intervalosRR)#720, 1344, ...#los picos RR son nuestro nuevo tiempo 
        print(f'Picos entre 1000 {picosRR}')
        duracionECG = picosRR[-1]/1000#481321 ms
        print('***********************')
        print(duracionECG)
        tiempo_ECG = pd.Series(picosRR) #720, 1344,.... enumerado 0, 1...
        fm = 0
        voltaje_ECG = [0]
        voltaje_ECG = pd.Series(voltaje_ECG)#Solo es para no graficar  
        picosRR = pd.Series(picosRR) 
        request.session['picosRR'] = picosRR.tolist()
        request.session['intervalosRR'] = intervalosRR.tolist()
        print('**********************************************')
        fs, TR, RRuniform = Parametro_dominio_frecuencia(picosRR, intervalosRR, tiempo_ECG, TipoArchivo)
        resultadosDF, frecuencias_welch, psd = calculo_welch(RRuniform, fs)
        request.session['Frecuencias_welch'] = frecuencias_welch.tolist()
        request.session['PSD'] = psd.tolist()
    

        # PERO YO QUIERO QUE CREE: ALGO
        #print(f'Tiempo_ECG: {tiempo_ECG}')#trabajando con ms...
        #print(f'Los picos leidos directamente del archivo son: {picosRR}')
    print(f'El tipo de archivo es {tipo_registro}')
    formatoDuracion = tiempoHMS(duracionECG)
    print(f'Formato de Duración {formatoDuracion}')
    # Si estás esperando una cadena con formato HH:MM:SS, asegúrate de tratarlo como cadena
    formatoDuracion = request.GET.get('formatoDuracion', formatoDuracion)



    #lOS UNICOS RESULTADOS QUE NECESITAN UN RECALCULO...
    if request.method == 'POST':
        try:
            data = json.loads(request.body)#solicitud json
            inicio = int(data.get('inicio', 0))*1000
            fin = int(data.get('fin', 0))*1000

            #llegan correctamente
            indices_tramo = []
            tramo = 0
            print(f'Inicio: {inicio}, Fin: {fin}')
            for i, intervalo in enumerate(intervalosRR):
                tramo += intervalo
              #  print(f'Iteración {i}: Tramo acumulado: {tramo}')
                if inicio <= tramo <= fin:
                    indices_tramo.append(i)

            Tramo_intervalosRR = [float(intervalosRR[i]) for i in indices_tramo]
            Tramo_picosRR = [int(picosRR[i]) for i in indices_tramo]

            Tramo_picosRR = np.array(Tramo_picosRR)
            Tramo_intervalosRR = np.array(Tramo_intervalosRR)

            #Por el momento ya realiza la actualización del segmento enviado
            #Ahora solo es que se actualicen los nuevos datos ..

           #print(Tramo_picosRR)
            #El tramo numero 79 corresponde el intervalo que se limita en un inicio con 100 ms..
            #Y el tramo de tiempo ECG tiene otra linea de tiempo porque el tacograma y el ECG son tramos diferentes 
                #print(Tramo_intervalosRR)

            resultadosDT, SDNN = Parametro_dominio_tiempo(Tramo_intervalosRR)
          #  print(f'Los resultados en el dominio del tiempo : {resultadosDT}')
            fs, TR, RRuniform = Parametro_dominio_frecuencia(Tramo_picosRR, Tramo_intervalosRR, tiempo_ECG, TipoArchivo) #vamos a darle todo el segmento de tiepo y el filtrara los corresppondoetes a los picos 
            resultadosDF, frecuencias_welch, psd = calculo_welch(RRuniform, 4)
            HR = (60/Tramo_intervalosRR)*1000 # Frecuencia cardiaca
            print(f'psd: {psd}')
            psd = psd/100000
            rr_mean = np.mean(Tramo_intervalosRR)
            total_intervalos = len(Tramo_intervalosRR)
            print(resultadosDT)
            
            request.session['picosRR'] = picosRR.tolist()
            request.session['intervalosRR'] = intervalosRR.tolist()
            request.session['Frecuencias_welch'] = frecuencias_welch.tolist()
            request.session['PSD'] = psd.tolist()

            fig_histogramaRR = {
            'data': [
                {'x': Tramo_intervalosRR.tolist(), 'type': 'histogram'}
            ],
            'layout': {
                'title': 'Histograma - Intervalos RR',
                'xaxis': {'title': 'Intervalo RR (ms)'},
                'yaxis': {'title': 'Frecuencia'}
            }
        }
            
            fig_histogramaHR = {
            'data' : [
                {'x': HR.tolist(), 'type': 'histogram'}
            ],
            'layout': {
                'title': 'Histograma Frecuencia Cardiaca',
                'xaxis':{'title': 'FC (lpm)'},
                'yaxis': {'title': 'Frecuencia'}
            },
        }
            fig_welch = {
                'data': [
                    {'x': frecuencias_welch.tolist(), 'y': psd.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Welch ECG'}
                ],
                'layout': {
                    'title': 'Welch ',
                    'xaxis': {'title': 'Frecuencia (Hz)'},
                    'yaxis': {'title': 'Amplitud (ms2/Hz)'}
                }
            }
            respuesta = {
                'graph_json_Welch' : json.dumps(fig_welch),
                'graph_json_RR': json.dumps(fig_histogramaRR),
                'graph_json_HR': json.dumps(fig_histogramaHR),
                'resultadosDT': resultadosDT,
                'resultadosDF': resultadosDF,
                'frecuencias_welch': frecuencias_welch.tolist(),
                'psd': psd.tolist(),
                'HR': HR.tolist(),
                'rr_mean': rr_mean,
                'total_intervalos': total_intervalos,
                'mensaje': 'Cálculos realizados correctamente.'
            }

            respuesta = reemplazar_nan_y_convertir(respuesta)
            respuesta_json = json.dumps(respuesta)
         #   print("Respuesta JSON:", respuesta_json)
            return JsonResponse(respuesta)

        except Exception as e:
            print(f"Error al calcular parámetros: {e}")
            return render(request,'ver_grafico.html', {'mensaje': 'Error al calcular parámetros.'})
    else:
        Banda_Analisis = banda_analisis * 1000 #conversión a milisegundos (300,000 ms)
        print(f'Banda de análisis: {Banda_Analisis}')
        picos_acumulados = 0 #en el tacograma seria cantidad de segmentos de tiempo acumulado 
        duracion_intervalos = 0# tiempo acumulado de los intervalos RR. 
        for intervalo in intervalosRR:
            duracion_intervalos += intervalo #funciona como cumsum
            picos_acumulados += 1 #cada intervalo equivale a un punto R             #para el tacograma, segemnto de la linea del tiempo== 1 pico R
            if duracion_intervalos >= Banda_Analisis:#si se superan los 300, 000 ms
                break
        print(f'pICOS ACUMULADOS: {picos_acumulados}')
        picos_RR = picosRR[:picos_acumulados]
        intervalos_RR = intervalosRR[:picos_acumulados]
        print(f'Ya se delimitaron los intervalos {intervalos_RR}')
        print(f'Los picos RR son: {picos_RR}')
        #print(f'Los intervalos RR son: {intervalos_RR}')
                # Calcular parámetros de dominio de tiempo y frecuencia
        #SE FILTRAN CORRECTAMENTEEEEEEEE
        resultadosDT, SDNN = Parametro_dominio_tiempo(intervalos_RR)
        fs, TR, RRuniform = Parametro_dominio_frecuencia(picos_RR, intervalos_RR, tiempo_ECG, TipoArchivo)
        resultadosDF, frecuencias_welch, psd = calculo_welch(RRuniform, fs)
        HR = (60/intervalosRR)*1000 # Frecuencia cardiaca
        psd = psd/100000
        rr_mean = np.mean(intervalos_RR)
        total_intervalos = len(intervalos_RR)
        print(f'SDNN: {SDNN}')

        duracion_acumulada = 0
        puntos_seleccionados = 0

        for intervalo in intervalosRR:
            duracion_acumulada += intervalo
            puntos_seleccionados += 1
            if duracion_acumulada >= tiempo_requerido_ms:
                break
        Tramo_picosRR = picosRR[:puntos_seleccionados]
        Tramo_intervalosRR = intervalosRR[:puntos_seleccionados]
       
        Tramo_picosRR = np.array(Tramo_picosRR)
        Tramo_intervalosRR = np.array(Tramo_intervalosRR)

#Tenemos que definir cual seria nuestra f

    muestras = int(banda * fm)


    datos_completos = {
        "ecg": {
            "tiempo": tiempo_ECG.astype(float).tolist(),
            "voltaje": voltaje_ECG.astype(float).tolist(),
        },
        "tacograma": {
            "picos": picosRR.astype(float).tolist(),
            "intervalos": intervalosRR.astype(float).tolist(),
        },
        'fm': float(fm),  # Asegúrate de que 'fm' también sea un valor flotante
    }

    # Datos iniciales para gráficos
    tiempo_mostrar = tiempo_ECG[:muestras]
    voltaje_mostrar = voltaje_ECG[:muestras]

    fig_ecg = {
        'data': [
            {'x': tiempo_mostrar.tolist(), 'y': voltaje_mostrar.tolist(), 'type': 'scatter', 'mode': 'lines', 'name': 'ECG'}
        ],
        'layout': {
            'title': 'Gráfico ECG',
            'xaxis': {'title': 'Tiempo'},
            'yaxis': {'title': 'Voltaje'}
        }
    }

    fig_tacograma = {
        'data': [
            {'x': picosRR.tolist(), 'y': intervalosRR.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Tacograma'}
        ],
        'layout': {
            'title': 'Tacograma - Intervalos RR',
            'xaxis': {'title': 'Tiempo (s)'},
            'yaxis': {'title': 'Intervalo RR (ms)'}
        }
    }
    fig_histogramaRR = {
        'data': [
            {'x': Tramo_intervalosRR.tolist(), 'type': 'histogram'}
        ],
        'layout': {
            'title': 'Histograma - Intervalos RR',
            'xaxis': {'title': 'Intervalo RR (ms)'},
            'yaxis': {'title': 'Frecuencia'}
        }
    }
    
    fig_histogramaHR = {
        'data' : [
            {'x': HR.tolist(), 'type': 'histogram'}
        ],
        'layout': {
            'title': 'Histograma Frecuencia Cardiaca',
            'xaxis':{'title': 'FC (lpm)'},
            'yaxis': {'title': 'Frecuencia'}
        },
    }
    fig_welch = {
        'data': [
            {'x': frecuencias_welch.tolist(), 'y': psd.tolist(), 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Welch ECG'}
        ],
        'layout': {
            'title': 'Welch ',
            'xaxis': {'title': 'Frecuencia (Hz)'},
            'yaxis': {'title': 'Amplitud (ms2/Hz)'}
        }
    }
    
    # Convertir el ndarray a lista
    #fig_tacograma_list = fig_tacograma.tolist()
    return render(request, 'ver_grafico.html', {
        'TipoArchivo': TipoArchivo,
        'voltaje_ECG': voltaje_ECG,
        'formatoDuracion': formatoDuracion,
        'banda_analisis': banda_analisis,
        'fm': fm,
        'rr_mean' : rr_mean,
        'total_intervalos': total_intervalos,
        'paciente': paciente,
        'registro_ecg': registro_ecg,
        'datos_completos_json': json.dumps(datos_completos),
        'banda': banda,
        'graph_json': json.dumps(fig_ecg),
        'graph_json_tacograma': json.dumps(fig_tacograma),
        'banda_tacograma': banda_tacograma_segundos,
        'resultadosDT': (resultadosDT),
        'resultadosDF': (resultadosDF),
        'graph_json_RR': json.dumps(fig_histogramaRR),
        'graph_json_HR': json.dumps(fig_histogramaHR),
        'graph_json_Welch': json.dumps(fig_welch),
        'especialista': especialista,  # Datos del especialista
        'paciente': paciente,  # Datos del paciente
        'ecg_id' : ecg_id,
    })


def visualizacion_informe(request, paciente_id):
    # Obtener el paciente y el especialista relacionado
    paciente = get_object_or_404(Paciente, id_paciente=paciente_id)
    especialista = paciente.especialista  # Obtener el especialista relacionado con el paciente
    # Solo mostrar los datos del especialista y paciente
    return render(request, 'visualizacion_informe.html', {
        'especialista': especialista,  # Datos del especialista
        'paciente': paciente,  # Datos del paciente
    })




from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Line
import io
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
from reportlab.graphics.charts.lineplots import LinePlot

from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis
import matplotlib.pyplot as plt
import numpy as np

banda_vlf = (0, 0.04)
banda_lf = (0.04, 0.15)
banda_hf = (0.15, 0.4)

def conversionTiempoSegundos(hora_minuto_segundo):
    if hora_minuto_segundo:
        h, m, s = map(int, hora_minuto_segundo.split(':'))
        return int(h * 3600 + m * 60 + s)
    return None  # Manejo de valores nulos o vacíos


def generar_graficos(datosX, datosY, nombre_grafico = 'ecg.png', xlabel = 'xlabel', ylabel = 'ylabel', title = 'Electrocardiograma'):
    # Crear la figura con un fondo transparente

    plt.figure(figsize=(18, 6))  # tamaño del cuadro
    plt.plot(datosX, datosY, color='#6e0000', lw=1)
    plt.title(title, fontsize=12, color='black', fontweight='normal', family='Times New Roman')
    plt.xlabel(xlabel, fontsize=10, color='black')
    plt.ylabel(ylabel, fontsize=10, color='black')
    plt.axhline(0, color='black',lw =1 )  # Línea del eje X
    plt.axvline(0, color='black', lw= 1)  # Línea del eje Y
    
    # Guardar la imagen con fondo transparente
    plt.savefig(nombre_grafico, format="png", dpi=300, bbox_inches="tight", transparent=True, facecolor='white')

    pass
def colorear_banda(frecuencias, psd, banda, color):
    frecuencias = np.array(frecuencias)  # Asegurar que es un array
    psd = np.array(psd)
    idx = np.logical_and(frecuencias >= banda[0], frecuencias <= banda[1])
    plt.fill_between(frecuencias[idx], psd[idx] / 100000, color=color)

def generar_histograma(datos, nombre_histograma = 'histograma.png', xlabel = 'xlabel', ylabel = 'ylabel', title = 'Histograma'):
    # Crear el histograma con un fondo transparente
    plt.figure(figsize=(6, 4))  # Ajustar el tamaño de la figura
    plt.hist(datos, bins=25, color='#990a22', edgecolor='black', alpha=0.7)  # Azul suave
    plt.title(title, fontsize=12, color='black', fontweight='normal', family='Times New Roman')
    plt.xlabel(xlabel, fontsize=10, color='black')
    plt.ylabel(ylabel, fontsize=10, color='black')


    #plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    
    # Guardar la imagen con fondo transparente
    plt.savefig(nombre_histograma, format="png", dpi=300, bbox_inches="tight", transparent=True, facecolor='white')
    plt.close()  # Cerrar la figura para liberar memoria
def generar_psd(frecuencias_welch, psd, nombre_psd = 'psd.png', xlabel = 'xlabel', ylabel = 'ylabel', title = 'psd'):
        frecuencias_welch = np.array(frecuencias_welch)  # Convertir a NumPy
        psd = np.array(psd)

        plt.figure(figsize=(6, 4))
        plt.plot(frecuencias_welch, psd/100000)
        plt.xlim(0, 0.4)
        plt.title('Densidad Espectral de Potencia (PSD)')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Densidad de potencia espectral [ms²/Hz]')
        

        colorear_banda(frecuencias_welch, psd, banda_vlf, (0/255, 15/255, 181/255))
        colorear_banda(frecuencias_welch, psd, banda_lf, (17/255, 120/255, 100/255))
        colorear_banda(frecuencias_welch, psd, banda_hf, (243/255, 41/255, 41/255))

        # Guardar la imagen con fondo transparente
        plt.savefig(nombre_psd, format="png", dpi=300, bbox_inches="tight", transparent=True, facecolor='white')
        plt.close()  # Cerrar la figura para liberar memoria


def generar_pdf(request, ecg_id=None):
    if not ecg_id:
        ecg_id = request.GET.get('ecg_id')  # Recibe ecg_id de parámetros de la URL
    
    if not ecg_id:
        return HttpResponse("Falta el ID del ECG", status=400)
    registro_ecg = get_object_or_404(ECG, id_ecg = ecg_id)
    ecg_id = registro_ecg.id_ecg
    paciente = registro_ecg.paciente
    especialista = paciente.especialista
    
    # Configurar la respuesta HTTP para el PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="reporte.pdf"'

    picosRR = request.session.get('picosRR', '[]')
    picosRR = np.array(picosRR)/1000
    intervalosRR = request.session.get('intervalosRR', '[]')#nos da una duración de 736
    intervalosRR = np.array(intervalosRR)
    print(f'INTERVALOS RR TOTALES = {intervalosRR}')
    print(f'PICOS TOTALES = {picosRR}')

    tiempo_ECG = request.session.get('tiempo_ECG', '[]')
    voltaje_ECG = request.session.get('voltaje_ECG', '[]')
    HR = list((60/rr)*1000 for rr in intervalosRR)
    
    frecuencias_welch = request.session.get('Frecuencias_welch', '[]')
    psd = request.session.get('PSD', '[]')

    Inicio =request.GET.get('rr-range-start')
    Fin = request.GET.get('rr-range-end')
    print(f'El tiempo inicio original: {Inicio}')
    print(f'El tiempo fin original: {Fin}')
    
    Inicio_segundos = conversionTiempoSegundos(Inicio)
    Fin_segundos = conversionTiempoSegundos(Fin)
    Inicio_segundosS = int(conversionTiempoSegundos(Inicio))*1000
    Fin_segundosS = int(conversionTiempoSegundos(Fin))*1000

    print(f'El tiempo inicio en segundos: {Inicio_segundosS}')
    print(f'El tiempo fin en segundos : {Fin_segundosS}')
    tiempo_ECG = np.array(tiempo_ECG)
    voltaje_ECG = np.array(voltaje_ECG)
    intervalosRR = np.array(intervalosRR)
    #vamos a filtrar el tiempo_ECG y voltaje_ECG con respectoal inicio y fin 
    tiempo_filtrado = tiempo_ECG[(tiempo_ECG >= (Inicio_segundos)) & (tiempo_ECG <= (Fin_segundos))]
    voltaje_filtrado = voltaje_ECG[(tiempo_ECG >= (Inicio_segundos)) & (tiempo_ECG <= (Fin_segundos))]



    Inicio_segundosS = int(conversionTiempoSegundos(Inicio))
    Fin_segundosS = int(conversionTiempoSegundos(Fin))

    indices_filtrados = np.where((picosRR >= Inicio_segundosS) & (picosRR <= Fin_segundosS))[0]

    # Ajustar el índice inicial para incluir el intervalo que termina en el primer pico filtrado
    primer_indice_para_intervalos = indices_filtrados[0]
    if primer_indice_para_intervalos > 0:
        primer_indice_para_intervalos -= 1 # Retrocede un índice para incluir el intervalo completo

    # Seleccionar los picos temporales para calcular los intervalos
    picosRR_filtrados_temp = picosRR[primer_indice_para_intervalos : indices_filtrados[-1] + 1]

    # Calcular los intervalos RR a partir de los picos filtrados
    intervalosRR_filtrados = np.diff(picosRR_filtrados_temp) * 1000 # Multiplicar por 1000 para volver a milisegundos
    picosRR_filtrados = picosRR_filtrados_temp[1:] # Los picos asociados a estos intervalos (los que terminan el intervalo)

    HR_Filtrada = list((60/rr)*1000 for rr in intervalosRR_filtrados)
   # picosRR_filtrado = picosRR[(picosRR >= Inicio_segundos) & (picosRR <= Fin_segundos)]
    #intervalosRR_filtrado = np.diff(picosRR_filtrado)
    print("Indices:", indices_filtrados)  # Debería contener valores como [1, 3, 5]
    print(f'Los intervalos RR filtrados son {intervalosRR_filtrados} ')
    print(f'Los picos filtradoss {picosRR_filtrados}')



    print('Se ha filtrado el ECG')

   # print(f'Frecuencias welch: {frecuencias_welch}')
    #print(f'psd: {psd}')

    # Crear el objeto canvas de ReportLab
    p = canvas.Canvas(response, pagesize=letter)#objeto u hoja
    width, height = letter  # Tamaño de la hoja 

    # Crear un área de dibujo


    #Recuadro de la información clinica
    color_recuadro1 = colors.Color(235/255, 235/255, 235/255)
    p.setFillColor(color_recuadro1)
    p.rect(30, height - 130, 551, 70, fill=True, stroke=False)
    #Recuadro de la información en el dominio del tiempo
    color_recuadro2 = colors.Color(255/255, 234/255, 234/255)
    p.setFillColor(color_recuadro2)
    p.rect(30, height - 469, 551, 335, fill=True, stroke=False)
    #Recuadro de la información en el dominio de la frecuecia
    color_recuadro3 = colors.Color(235/255, 235/255, 235/255)
    p.setFillColor(color_recuadro3)
    p.rect(30, height - 751, 551, 278, fill=True, stroke=False)


    color_titulo = colors.Color(141/255, 0/255, 0/255)
    # Título del documento
    p.setFont("Times-Roman", 20)
    p.setFillColor(color_titulo)
    p.drawString(60, height - 50, "Reporte general de la Variabilidad de la Frecuencia Cardiaca")
    
    # Línea horizontal para separar el título
    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 60, width - 30, height - 60)
    edad = paciente.calcular_edad()

    color_datos_clinicos = colors.Color(19/255, 10/255, 48/255)
    # Sección de paciente
    p.setFont("Times-Roman", 14)
    p.setFillColor(color_datos_clinicos)
    p.drawString(50, height - 78, f"Especialista a cargo: {especialista.nombre_especialista}")
    p.drawString(50, height - 93, f"Paciente: {paciente.nombre_paciente} {paciente.apellido_paterno} {paciente.apellido_materno}")
    p.drawString(50, height - 108, f"Sexo: {paciente.sexo} ")
    p.drawString(50, height - 123, f"Edad: {edad} ")
    p.drawString(350, height - 78, f"Actividad Física: {paciente.actividad_fisica}")
    p.drawString(350, height - 93, f"IMC: {paciente.imc} ")
    p.drawString(350, height - 108, f"Uso de medicamentos: {paciente.uso_de_medicamentos} ")
    #p.drawString(350, height - 120, f"pICOSRR: {intervalosRR} ")
         
 # Abrir la imagen con PIL
    datos_analizados = [
        ['Segmento analizado', 'rata'],
        [request.GET.get('rr-range-start'), request.GET.get('rr-range-end')]
    ]
    # Obtener los datos del formulario con valores predeterminados
    datos_dominio_tiempo = [
        ["Parámetro", "Min", "Mean", "Max"],
        ["NNI", request.GET.get('nni_mean', 'N/A'), request.GET.get('nni_min', 'N/A'), request.GET.get('nni_max', 'N/A')],
        ["HR", request.GET.get('hr_mean', 'N/A'), request.GET.get('hr_min', 'N/A'), request.GET.get('hr_max', 'N/A')],
        ["STD HR", request.GET.get('hr_std', 'N/A'), '', ''],  # No tiene min, medio, ni max
        ["NNI Diff", request.GET.get('nni_diff_mean', 'N/A'), request.GET.get('nni_diff_min', 'N/A'), request.GET.get('nni_diff_max', 'N/A')],  # No tiene min, medio, ni max
        ["SDNN", request.GET.get('sdnn', 'N/A'), '', ''],
        ["SDANN", request.GET.get('sdann', 'N/A'), '', ''],
        ["RMSSD", request.GET.get('rmssd', 'N/A'), '', ''],
        ["SDSD", request.GET.get('sdsd', 'N/A'), '', ''],
        ["NN50", request.GET.get('nn50', 'N/A'), '', ''],
        ["pNN50", request.GET.get('pnn50', 'N/A'), '', ''],
        ["NN20", request.GET.get('nn20', 'N/A'), '', ''],
        ["R-R Mean", request.GET.get('rr_mean', 'N/A'), '', ''],
        ["Total Intervalos", request.GET.get('total_intervalos', 'N/A'), '', '']]
    datos_dominio_frecuencia = [
        ['Parámetro', 'VLF (0 - 0.04 Hz)', 'LF (0.04 - 0.15 Hz)', 'HF (0.15 - 0.4 Hz)'],
        ['Potencia (ms2)', request.GET.get('potencia_vlf', 'N/A'), request.GET.get('potencia_lf'), request.GET.get('potencia_hf')],
        ['Peak (Hz)', request.GET.get('pico_vlf'), request.GET.get('pico_lf'), request.GET.get('pico_hf')],
        ['Potencia (log)', request.GET.get('potencia_vlf_log'), request.GET.get('potencia_lf_log'), request.GET.get('potencia_hf_log')],
        ['Potencia (n.u.)', '', request.GET.get('potencia_lf_nu'), request.GET.get('potencia_hf_nu')],
         ]
    datos_generales = [
        ['Parametro', 'Valor'],
        ['Potencia Total (ms2)', request.GET.get('potencia_total')],
        ['LF/HF', request.GET.get('lf_hf_ratio')]
        ]


    tabla = Table(datos_dominio_tiempo, colWidths=[75, 50, 50, 50])
    tabla2 = Table(datos_dominio_frecuencia, colWidths = [75, 82, 82, 82])
    tabla3 = Table(datos_generales, colWidths=(90, 90))
    table4 = Table(datos_analizados, colWidths=(90, 90))
    color_encabezado = colors.Color(204/255, 0/255, 34/255)
    color_filas = colors.Color(255/255, 141/255, 141/255)
    color_encabezado2 = colors.Color(146/255, 16/255, 16/255)
    color_filas2 = colors.Color(221/255, 135/255, 135/255)


    # Estilo de la tabla
    estilo = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), color_encabezado),  # Fondo encabezado
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Texto blanco en el encabezado
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Centrado de texto
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),  # Fuente negrita en el encabezado
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),  # Espaciado inferior en el encabezado
        ('BACKGROUND', (0, 1), (-1, -1), color_filas),  # Fondo beige en las filas
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Bordes de la tabla
    ])
    tabla.setStyle(estilo)

    estilo2 = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), color_encabezado2),  # Fondo encabezado
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Texto blanco en el encabezado
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Centrado de texto
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),  # Fuente negrita en el encabezado
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),  # Espaciado inferior en el encabezado
        ('BACKGROUND', (0, 1), (-1, -1), color_filas2),  # Fondo beige en las filas
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Bordes de la tabla
    ])
    
    tabla2.setStyle(estilo2)
    tabla3.setStyle(estilo2)
    table4.setStyle(estilo2)

    # Dibujar la tabla en el PDF
    tabla.wrapOn(p, width, height)
    tabla.drawOn(p, 50, height - 460)  # Ajustar la posición Y según el contenido
    
    tabla2.wrapOn(p, width, height)
    tabla2.drawOn(p, 50, height - 578)

    tabla3.wrapOn(p, width, height)
    tabla3.drawOn(p, 390, height - 542)

    table4.wrapOn(p, width, height)
    table4.drawOn(p, 50, height - 190)
    
   # print("Picos RR:", picosRR)
   # print("Intervalos RR:", intervalosRR)


    generar_histograma(intervalosRR_filtrados, "histograma_rr.png", xlabel="Intervalos RR (ms)", ylabel="Frecuencia", title="Histograma de Intervalos RR")
    generar_histograma(HR_Filtrada, "histograma_hr.png", xlabel="Frecuencia Cardíaca (bpm)", ylabel="Frecuencia", title="Histograma de Frecuencia Cardíaca")
    generar_psd(frecuencias_welch, psd, 'psd.png', xlabel='Frecuencia [Hz]', ylabel='Densidad de potencia espectral [ms²/Hz]', title='Densidad Espectral de Potencia')
    #p.drawImage("histograma_rr.png", 100, height - 400, width=400, height=300)


    # Insertar la imagen del histograma en el PDF (ajustando el tamaño)
    p.drawImage("histograma_rr.png", x=350, y=height-300, width=210, height=160)
    p.drawImage("histograma_hr.png", x=350, y=height-465, width=210, height=160)
    p.drawImage("psd.png", x=50, y=height-743, width=320, height=160 )



# Ruta del archivo de imagen (cambia esto a la ubicación real de tu imagen)

    ruta_imagen = "C:\\Users\\vlzdi\\Documents\\Cardio Metrics HRV\\ProyectoHRVUltimaVersion\\escudo_INC-Rojo.jpg"


    # Agregar imagen al PDF (posición x, y y tamaño opcional)
    p.drawImage(ruta_imagen, x=410, y=height-743, width=127, height=157)

    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 752, width - 30, height - 752)

    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 130, width - 30, height - 130)
    
    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 134, width - 30, height - 134)

    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 469, width - 30, height - 469)

    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 473, width - 30, height - 473)


    p.showPage()

    #Segunda página

    #Recuadro de la información clinica
    color_recuadro1 = colors.Color(235/255, 235/255, 235/255)
    p.setFillColor(color_recuadro1)
    p.rect(30, height - 130, 551, 70, fill=True, stroke=False)
    #Recuadro Electrocardiogra
    color_recuadro2 = colors.Color(255/255, 234/255, 234/255)
    p.setFillColor(color_recuadro2)
    p.rect(30, height - 276, 551, 130, fill=True, stroke=False)
    #Recuadro del segmento del electrocardiograma
    color_recuadro3 = colors.Color(235/255, 235/255, 235/255)
    p.setFillColor(color_recuadro3)
    p.rect(30, height - 420, 551, 130, fill=True, stroke=False)

    p.setFillColor(color_recuadro2)
    p.rect(30, height - 564, 551, 130, fill=True, stroke=False)

    p.setFillColor(color_recuadro3)
    p.rect(30, height - 708, 551, 130, fill=True, stroke=False)

    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 758, width - 30, height - 752)


    color_titulo = colors.Color(141/255, 0/255, 0/255)
    # Título del documento
    p.setFont("Times-Roman", 20)
    p.setFillColor(color_titulo)
    p.drawString(60, height - 50, "Reporte general de la Variabilidad de la Frecuencia Cardiaca")
    
    # Línea horizontal para separar el título
    p.setStrokeColor(colors.black)
    p.setLineWidth(1)
    p.line(30, height - 60, width - 30, height - 60)
    edad = paciente.calcular_edad()

    color_datos_clinicos = colors.Color(19/255, 10/255, 48/255)
    # Sección de paciente
    p.setFont("Times-Roman", 14)
    p.setFillColor(color_datos_clinicos)
    p.drawString(50, height - 78, f"Especialista a cargo: {especialista.nombre_especialista}")
    p.drawString(50, height - 93, f"Paciente: {paciente.nombre_paciente} {paciente.apellido_paterno} {paciente.apellido_materno}")
    p.drawString(50, height - 108, f"Sexo: {paciente.sexo} ")
    p.drawString(50, height - 123, f"Edad: {edad} ")
    p.drawString(350, height - 78, f"Actividad Física: {paciente.actividad_fisica}")
    p.drawString(350, height - 93, f"IMC: {paciente.imc} ")
    p.drawString(350, height - 108, f"Uso de medicamentos: {paciente.uso_de_medicamentos} ")
    p.drawString(190, height - 143, f"Registro Electrocardiográfico Completo ")

    #def generar_graficos(datosX, datosY, nombre_grafico = 'ecg.png', xlabel = 'xlabel', ylabel = 'ylabel', title = 'Electrocardiograma'):
    generar_graficos(tiempo_ECG, voltaje_ECG, 'electrocardiograma.png', 'Tiempo (seg)', 'Voltaje (mV)', 'Electrocardiograma')
    generar_graficos(tiempo_filtrado, voltaje_filtrado, 'electrocardiogramaSegmentado.png', 'Tiempo (seg)', 'Voltaje (mV)', 'Segmento del Electrocardiograma analizado')
    generar_graficos(picosRR[1:], intervalosRR, 'Tacograma.png', 'Tiempo (seg)', 'Voltaje (ms)', 'Tacograma')
    generar_graficos(picosRR_filtrados, intervalosRR_filtrados, 'TacogramaSegmentado.png', 'Tiempo (seg)', 'Voltaje (ms)', 'Segmento de Tacograma analizado')

    
    p.drawImage("electrocardiograma.png", x=88, y=height-269, width=430, height=117)
    p.drawImage("electrocardiogramaSegmentado.png", x=88, y=height-414, width=430, height=117)
    p.drawImage("Tacograma.png", x=88, y=height-557, width=430, height=117)
    p.drawImage("TacogramaSegmentado.png", x=88, y=height-700, width=430, height=117)
    

    # Guardar el PDF
    p.showPage()
    p.save()

    return response


