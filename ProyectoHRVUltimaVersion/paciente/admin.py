from django.contrib import admin  # Importa el módulo admin de Django para gestionar la interfaz de administración
from .models import *


# Registra el modelo Paciente y su clase administrativa asociada en la interfaz de administración de Django
admin.site.register(Paciente)
admin.site.register(Departamento)
admin.site.register(Especialista)
admin.site.register(AnalisisDominioFrecuencia)
admin.site.register(AnalisisDominioTiempo)
admin.site.register(ECG)