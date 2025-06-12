"""
ASGI config for djangocrud project.

Este archivo contiene la configuración ASGI (Asynchronous Server Gateway Interface) para el proyecto Django.
Exhibe el callable ASGI como una variable de nivel de módulo llamada ``application``.

Para más información sobre cómo configurar ASGI, consulta la documentación oficial:
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os

# Configura la variable de entorno para el módulo de configuración de Django.
# Esto le dice a Django qué configuración debe utilizar para este entorno.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangocrud.settings')

# Obtiene la aplicación ASGI para el proyecto Django.
# Este callable se utiliza para gestionar las conexiones asíncronas.
application = get_asgi_application()
