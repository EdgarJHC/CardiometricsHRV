from django.apps import AppConfig


from django.apps import AppConfig

"""
PacientesConfig: Esta clase hereda de AppConfig, que es una clase base para la configuración de 
aplicaciones en Django. Se utiliza para definir la configuración de la aplicación "pacientes".

default_auto_field: Este atributo especifica qué tipo de campo se usará por defecto para las claves 
primarias de los modelos en esta aplicación. En este caso, se establece como BigAutoField, lo que 
permite que las claves primarias sean números enteros grandes, útil si se espera un gran número de 
registros.

name: Este es el nombre de la aplicación, que debe coincidir con el nombre del directorio de la 
aplicación en tu proyecto Django. Este nombre se utiliza internamente por Django para identificar la 
aplicación.
"""

# Configuración de la aplicación Pacientes
class PacienteConfig(AppConfig):
    # Especifica el tipo de campo que se utilizará como ID automático por defecto para los modelos de esta aplicación.
    default_auto_field = 'django.db.models.BigAutoField'
    
    # Nombre de la aplicación. Este nombre debe coincidir con el nombre del directorio de la aplicación.
    name = 'paciente'
