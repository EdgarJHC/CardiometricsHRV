"""djangocrud URL Configuration

Este archivo configura las URL de tu proyecto Django. Las URLs se asignan a vistas específicas que manejan las solicitudes HTTP.

Para más información sobre cómo funcionan las configuraciones de URL en Django, consulta la documentación oficial:
https://docs.djangoproject.com/en/4.1/topics/http/urls/

Ejemplos:
- Vistas de función:
    1. Agrega una importación: from my_app import views
    2. Agrega una URL a urlpatterns: path('', views.home, name='home')
    
- Vistas basadas en clase:
    1. Agrega una importación: from other_app.views import Home
    2. Agrega una URL a urlpatterns: path('', Home.as_view(), name='home')
    
- Incluyendo otro archivo de configuración de URL:
    1. Importa la función include: from django.urls import include, path
    2. Agrega una URL a urlpatterns: path('blog/', include('blog.urls'))
"""

# Importa el módulo admin de django.contrib, que permite administrar el sitio.
from django.contrib import admin

# Importa las funciones path de django.urls, que se utilizan para definir las rutas de URL.
from django.urls import path, include

# Importa las vistas desde el módulo pacientes.
from paciente import views

from django.conf import settings
from django.contrib.staticfiles.urls import static
from django.conf.urls.static import static


# La lista urlpatterns contiene las rutas de URL y las vistas asociadas.
urlpatterns = [
    # Ruta para la página de inicio del sitio, asignada a la vista 'home'.
    path('', views.home, name='home'),
    
    path('homeDoctor/', views.homeDoctor, name='homeDoctor'),    
    
    # Ruta para el panel de administración de Django.
    path('admin/', admin.site.urls),
    
    # Ruta para iniciar sesión, asignada a la vista 'signin'.
    path('signin/', views.signin, name='signin'),
    
    # Ruta para el registro de nuevos usuarios, asignada a la vista 'signup'.
    path('signup/', views.signup, name='signup'),
    
    path('signout/', views.signout, name='signout'),   

   # path('forgot_password/', views.forgot_password, name='forgot_password'),
    path('forgot_password/', views.forgot_password, name='forgot_password'),

    # Ruta para visualizar todos los pacientes, asignada a la vista 'pacientes'.
    path('pacientes/', views.pacientes, name='pacientes'),
    
    # Ruta para crear un nuevo registro de paciente, asignada a la vista 'create_paciente'.
    path('create_paciente/', views.create_paciente, name='create_paciente'),
    
    path('visualizacion_informe/<int:paciente_id>/', views.visualizacion_informe, name='visualizacion_informe'),
    
    # Ruta para ver los detalles de un paciente específico, utilizando el ID del paciente.
    path('pacientes/<int:paciente_id>', views.historial, name='historial'),
    
    # Ruta para marcar un paciente como completado, utilizando el ID del paciente.
    path('paciente/<int:paciente_id>/complete', views.complete_paciente, name='complete_paciente'),
    
    # Ruta para eliminar un paciente específico, utilizando el ID del paciente.
    path('pacientes/<int:paciente_id>/eliminar', views.eliminar_paciente, name='eliminar_paciente'),
    
    path('buscar/', views.buscar, name='buscar'),
    
    path('pacientes/crear_informe/<int:paciente_id>/', views.crear_informe, name='crear_informe'),  # Cambia a la vista que necesites después de guardar
    
    # Ruta para editar un paciente específico, utilizando el ID del paciente.
    path('pacientes/<int:paciente_id>/editar', views.editar_paciente, name='editar_paciente'),

    path('perfil_especialista/', views.perfil_doc, name='perfilEspecialista'), # Nueva vista para datos personales 
     
    path('error/', views.error_page, name='error_page'),  # Agrega esta línea
    
    path('pacientes/ver_grafico/<int:ecg_id>/', views.ver_grafico, name = 'ver_grafico'),  # Nueva vista para ver el gráfico de HRV
    
    path('pacientes/eliminar_informe/<int:paciente_id>/eliminar',views.eliminar_informe, name='eliminar_informe'),

    path('pacientes/generar_pdf/<int:ecg_id>/', views.generar_pdf, name='generar_pdf'),

    ]   

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
