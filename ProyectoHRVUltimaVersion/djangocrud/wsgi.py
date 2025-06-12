"""
WSGI config for djangocrud project.

Este archivo expone la llamada WSGI como una variable a nivel de módulo llamada ``application``.
WSGI (Web Server Gateway Interface) es un estándar que permite la comunicación entre servidores web y aplicaciones Python.

Para más información sobre este archivo, consulta la documentación oficial de Django:
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

# Importa el módulo os, que permite interactuar con el sistema operativo.
import os

# Importa la función get_wsgi_application del módulo django.core.wsgi.
# Esta función se utiliza para obtener la aplicación WSGI de Django.
from django.core.wsgi import get_wsgi_application

# Establece la variable de entorno DJANGO_SETTINGS_MODULE.
# Esta variable le dice a Django qué configuración utilizar.
# En este caso, se establece como 'djangocrud.settings', donde 'djangocrud' es el nombre del proyecto.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangocrud.settings')

# Llama a la función get_wsgi_application() para obtener la aplicación WSGI.
# Esta aplicación se utiliza para servir tu proyecto Django a través de un servidor WSGI.
application = get_wsgi_application()
