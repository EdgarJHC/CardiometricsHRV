"""
Django settings for djangocrud project.

Este archivo contiene la configuración de tu proyecto Django. Se generan automáticamente mediante el comando 'django-admin startproject' y se utilizan para definir el comportamiento del proyecto.

Para más información sobre cómo funcionan las configuraciones de Django, consulta la documentación oficial:
https://docs.djangoproject.com/en/4.1/topics/settings/

Para obtener la lista completa de configuraciones y sus valores, visita:
https://docs.djangoproject.com/en/4.1/ref/settings/
"""
import os
from pathlib import Path

# Construye las rutas dentro del proyecto como esta: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Configuración rápida de desarrollo - inadecuada para producción
# Consulta la guía de despliegue de Django para la configuración correcta en producción.
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# ADVERTENCIA DE SEGURIDAD: mantén la clave secreta utilizada en producción en secreto.
SECRET_KEY = 'django-insecure-(oa(omhdw75#3qzk_p-6zfdfmvj#%tn=oci!ww+ssog(ib%-o='

# ADVERTENCIA DE SEGURIDAD: ¡no ejecutes con debug activado en producción!
DEBUG = True

# Lista de hosts permitidos para la aplicación.
ALLOWED_HOSTS = []


# Definición de la aplicación

# Aplicaciones instaladas que se utilizarán en el proyecto.
INSTALLED_APPS = [
    'django.contrib.admin',          # Administración de Django
    'django.contrib.auth',           # Autenticación de usuarios
    'django.contrib.contenttypes',   # Tipos de contenido
    'django.contrib.sessions',        # Manejo de sesiones
    'django.contrib.messages',        # Mensajes de notificación
    'django.contrib.staticfiles',     # Archivos estáticos
    'paciente',                    # Tu aplicación 'pacientes'
]

# Middleware que se aplicará a las solicitudes y respuestas.
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',  # Middleware de seguridad
    'django.contrib.sessions.middleware.SessionMiddleware',  # Manejo de sesiones
    'django.middleware.common.CommonMiddleware',  # Middleware común
    'django.middleware.csrf.CsrfViewMiddleware',  # Protección contra ataques CSRF
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Manejo de autenticación
    'django.contrib.messages.middleware.MessageMiddleware',  # Manejo de mensajes
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # Protección contra clickjacking

]

# Ruta de configuración de URL raíz para el proyecto.
ROOT_URLCONF = 'djangocrud.urls'

# Configuración de plantillas
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',  # Backend de Django para plantillas
        'DIRS': [],  # Directorios de plantillas (se puede especificar una lista)
        'APP_DIRS': True,  # Habilitar búsqueda de plantillas en aplicaciones instaladas
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',  # Procesador de contexto para depuración
                'django.template.context_processors.request',  # Procesador de contexto para solicitudes
                'django.contrib.auth.context_processors.auth',  # Procesador de contexto para autenticación
                'django.contrib.messages.context_processors.messages',  # Procesador de contexto para mensajes
            ],
        },
    },
]

# Configuración de la aplicación WSGI
WSGI_APPLICATION = 'djangocrud.wsgi.application'


# Configuración de la base de datos
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',  # Motor de base de datos MySQL
        'NAME': 'CardioMetricsHRV$proyectovfc_hrv_vf',  # Nombre de la base de datos creada
        'USER': 'CardioMetricsHRV',  # Usuario de MySQL
        'PASSWORD': 'gansito14*',  # Contraseña de MySQL (en blanco por defecto)
        'HOST': 'CardioMetricsHRV.mysql.pythonanywhere-services.com',  # Dirección del servidor MySQL (localhost)
        'PORT': '3306',  # Puerto de MySQL (predeterminado 3306)
        'OPTIONS': {
            'sql_mode': 'STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,ONLY_FULL_GROUP_BY', # <-- ¡NO_AUTO_CREATE_USER ha sido eliminado!
        },
    }
}


# Validación de contraseñas
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',  # Valida la similitud de atributos
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',  # Valida la longitud mínima
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',  # Valida contraseñas comunes
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',  # Valida contraseñas numéricas
    },
]


# Internacionalización
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'en-us'  # Código de idioma

TIME_ZONE = 'UTC'  # Zona horaria

USE_I18N = True  # Habilita la internacionalización

USE_TZ = True  # Habilita el manejo de zonas horarias


# Archivos estáticos (CSS, JavaScript, imágenes)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS = [  # Cambiado de STATICFILES a STATICFILES_DIRS
    BASE_DIR / "staticfiles/static",  # Cambia esto a la ruta correcta
]
STATIC_ROOT = BASE_DIR / "staticfiles"  # Asegúrate de que esta carpeta exista también.



LOGIN_URL = '/signin'  # URL de inicio de sesión


# Tipo de campo de clave primaria por defecto
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'  # Establece el tipo de campo de clave primaria por defecto

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

#EMAIL_PORT = 587
#EMAIL_USE_TLS = True
#EMAIL_HOST_USER = 'vlzdiana954@gmail.com'  # Tu correo Gmail
#EMAIL_HOST_PASSWORD = 'TatEdg45*'  # Genera una contraseña de aplicación en Gmail


INTERNAL_IPS = ['127.0.0.1']

