from django.db import models
from django.contrib.auth.models import User
import hashlib
from django.core.validators import FileExtensionValidator
import os
import datetime
from django.db.models import Max
import pandas as pd 
from django.utils import timezone
# Clase Paciente que representa los datos de un paciente en la base de datos

class Paciente(models.Model):
    id_paciente = models.AutoField(primary_key=True)
    #user = models.ForeignKey(User, on_delete=models.CASCADE)  # Relación con el modelo User
    nombre_paciente = models.CharField(max_length=45, verbose_name= 'Nombre del paciente', null=False, blank=True)
    apellido_paterno = models.CharField(max_length=45, verbose_name= 'Apellido paterno', null=False, blank=True)
    apellido_materno = models.CharField(max_length=45, verbose_name= 'Apellido materno', null=False, blank=True)
    telefono = models.CharField(max_length=15, verbose_name= 'Telefono', null=True, blank=True)
    correo = models.CharField(max_length=45, verbose_name= 'Correo electrónico', null=False, blank=True)
    sexo = models.CharField(max_length=15, verbose_name= 'Sexo', null=False, blank=True)
    fecha_nacimiento = models.DateField(verbose_name= 'Fecha Nacimiento', null=True, blank=True)
    imc = models.FloatField(null = True, blank=True, verbose_name= 'Indice de masa corporal')
    uso_de_medicamentos = models.CharField(max_length=100, null=True, blank=True, verbose_name= 'Medicamentos')  # Nuevo campo
    actividad_fisica = models.CharField(max_length=100, null=True, blank=True, verbose_name= 'Actividad física')      # Nuevo campo

    
    # Relación con Especialista: cada paciente está vinculado a un especialista específico
    especialista = models.ForeignKey('Especialista', on_delete=models.CASCADE, related_name="pacientes")

    def calcular_edad(self):
        if self.fecha_nacimiento:
            today = timezone.now().date()
            edad = today.year - self.fecha_nacimiento.year
            if today.month < self.fecha_nacimiento.month or (today.month == self.fecha_nacimiento.month and today.day < self.fecha_nacimiento.day):
                edad -= 1
            return edad
        return None
    
    def generar_id_paciente(self):
        # Obtener el año y mes actual
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month
        
        # Obtener el último número de registro para el año y mes actuales
        last_paciente = Paciente.objects.filter(id_paciente__startswith=f"PCT{current_year}{str(current_month).zfill(2)}") \
                                        .aggregate(last_id=Max('id_paciente'))

        # Si ya hay registros, obtenemos el número más alto, si no, iniciamos en 1
        if last_paciente['last_id']:
            last_number = int(last_paciente['last_id'][-3:])  # Extraemos los últimos 3 dígitos
            new_number = last_number + 1
        else:
            new_number = 1  # Si no hay registros, comenzamos desde 1

        # Generamos el ID con el prefijo, el año, el mes y el número secuencial
        return f"PCT{current_year}{str(current_month).zfill(2)}{str(new_number).zfill(3)}"

    def __str__(self):
        return f"{self.nombre_paciente} {self.apellido_paterno} - {self.especialista.user.username}"



class Departamento(models.Model):
    id_departamento = models.AutoField(primary_key=True)
    departamento = models.CharField(max_length=45, null=False, blank=True)

    def __str__(self):
        return self.departamento



class Especialista(models.Model):
    id_especialista = models.AutoField(primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    nombre_especialista = models.CharField(max_length=100)
    apellido_paterno = models.CharField(max_length=100)
    apellido_materno = models.CharField(max_length=100)
    telefono = models.CharField(max_length=15)
    correo = models.EmailField(max_length=100, unique=True)
    especialidad = models.CharField(max_length=100)
    fecha_nacimiento = models.DateField(verbose_name= 'Fecha Nacimiento', null=False, blank=True)
    departamento_id = models.ForeignKey(Departamento, on_delete=models.CASCADE)  # Cambiado a ForeignKey

    def __str__(self):
        return (f"{self.id_especialista} - {self.nombre_especialista} {self.apellido_paterno} {self.apellido_materno}, "
                f"Fecha_Naciemiento: {self.fecha_nacimiento}, Teléfono: {self.telefono}, Correo: {self.correo}, "
                f"Especialidad: {self.especialidad}, Departamento: {self.departamento_id}")

 
class ECG(models.Model):
    id_ecg = models.AutoField(primary_key=True)
    archivo_ecg = models.FileField(upload_to= 'archivo_ecg/', validators=[FileExtensionValidator(allowed_extensions=['txt', 'csv'])])
    fecha_informe = models.DateTimeField(auto_now_add=True)
    comentarios = models.TextField()
    paciente = models.ForeignKey(Paciente, on_delete=models.CASCADE, db_column='id_paciente')
    homoclave = models.CharField(max_length=64, unique=True, blank=True, null=True)  # Nuevo campo para homoclave
    tipo_archivo = models.CharField(default='ECG', 
                                    max_length=40,
                                    choices=[('ECG','Electrocardiograma Digital'), 
                                             ('TAC','Tacograma')])


    def save(self, *args, **kwargs):
        if not self.homoclave:  # Generar homoclave solo si no existe
            salt = os.urandom(16)
            fullhash = hashlib.sha256(salt).hexdigest()
            # Verificar el número de columnas del archivo
            file_path = self.archivo_ecg.path
            print(f'El ARCHIVO ESTA SIENDO GUARDADO CORRECTAMENTE {file_path}')
            try:
                delimitadores = ["\t", ",", " ", ";"]
                for delim in delimitadores:
                    datos_ecg = pd.read_csv(file_path, sep=delim, header=None)  # Leer sin encabezados
                    if datos_ecg.shape[1] == 1:
                        self.homoclave = 'TAC' + fullhash[:7].upper()
                    elif datos_ecg.shape[1] >= 2:
                        self.homoclave = 'ECG' + fullhash[:7].upper()
            except Exception as e:
                print(f"Error al leer el archivo: {e}")
                self.homoclave = 'ERR' + fullhash[:7].upper()  # Prefijo para errores

        super().save(*args, **kwargs)  # Llamar al método original
        
    def detectar_delimitador(self, file_path):
            """Detecta el delimitador correcto probando varios posibles."""
            delimitadores = ["\t", ",", " ", ";"]
            for delim in delimitadores:
                try:
                    datos_ecg = pd.read_csv(file_path, sep=delim, header=None)
                    if datos_ecg.shape[1] >= 1:  # Si hay al menos una columna, asumimos que el delimitador es correcto
                        return delim
                except Exception:
                    continue
            return None  # Si no se encuentra un delimitador válido

    def save(self, *args, **kwargs):
        if not self.homoclave:  # Generar homoclave solo si no existe
            salt = os.urandom(16)
            fullhash = hashlib.sha256(salt).hexdigest()
            file_path = self.archivo_ecg.path

            try:
                delimitador = self.detectar_delimitador(file_path)
                if delimitador:
                    datos_ecg = pd.read_csv(file_path, sep=delimitador, header=None)
                    if datos_ecg.shape[1] == 1:
                        self.homoclave = 'TAC' + fullhash[:7].upper()
                    elif datos_ecg.shape[1] >= 2:
                        self.homoclave = 'ECG' + fullhash[:7].upper()
                else:
                    raise ValueError("No se pudo detectar un delimitador válido.")
            except Exception as e:
                print(f"Error al leer el archivo: {e}")
                self.homoclave = 'ERR' + fullhash[:7].upper()  # Prefijo para errores

        super().save(*args, **kwargs)  # Llamar al método original
   
    def __str__(self):
        return f'ECG paciente: {self.paciente.nombre_paciente} {self.paciente.apellido_paterno} {self.paciente.apellido_materno}'
 
       
class AnalisisDominioFrecuencia(models.Model):
    id_analisis_frecuencia = models.AutoField(primary_key=True)
    potencia_vlf = models.FloatField(default=0.0)
    pico_vlf = models.FloatField(default=0.0)
    potencia_vlf_log = models.FloatField(default=0.0)
    porcentaje_vlf = models.FloatField(default=0.0)
    potencia_lf = models.FloatField(default=0.0)
    pico_lf = models.FloatField(default=0.0)
    potencia_lf_log = models.FloatField(default=0.0)
    porcentaje_lf = models.FloatField(default=0.0)
    potencia_hf = models.FloatField(default=0.0)
    pico_hf = models.FloatField(default=0.0)
    potencia_hf_log = models.FloatField()
    porcentaje_hf = models.FloatField(default=0.0)
    potencia_total = models.FloatField(default=0.0)
    potencia_lf_nu = models.FloatField(default=0.0)
    potencia_hf_nu = models.FloatField(default=0.0)
    lf_hf_ratio = models.FloatField(default=0.0)
    ecg = models.ForeignKey(ECG, on_delete=models.CASCADE, db_column='ID_ECG')

    class Meta:
        db_table = 'analisis_dominio_frecuencia'
    def __str__(self):
        return f'AF_{self.paciente.apellido_paterno[0]}{self.paciente.apellido_materno[0]}{self.paciente.nombre_paciente[0]}_{self.ecg.fecha_informe.strftime("%Y%m%d_%H:%M:%S")}'


class AnalisisDominioTiempo(models.Model):
    #'nni_min'
    id_analisis_tiempo = models.AutoField(primary_key=True)
    nni_mean = models.FloatField()
    nni_min = models.FloatField()
    nni_max = models.FloatField()
    hr_mean = models.FloatField()
    hr_min = models.FloatField()
    hr_max = models.FloatField()
    std_hr = models.FloatField()
    nni_diff_mean = models.FloatField()
    nni_diff_min = models.FloatField()
    nni_diff_max = models.FloatField()
    sdnn = models.FloatField()
    sdnn_index = models.FloatField()
    sdann = models.FloatField()
    rmssd = models.FloatField()
    sdsd = models.FloatField()
    nn50 = models.FloatField()
    pnn50 = models.FloatField()
    nn20 = models.FloatField()
    rr_mean = models.FloatField()
    tinn_n = models.FloatField(default=0.0)
    tinn_m = models.FloatField(default=0.0)
    tinn= models.FloatField(default=0.0)
    tri_index = models.FloatField(default=0.0)
    total_intervalos_rr = models.IntegerField()
    ecg = models.OneToOneField(ECG, on_delete=models.CASCADE, db_column='ID_ECG')

    class Meta:
        db_table = 'analisis_dominio_tiempo'
    def __str__(self):
        return f'AT_{self.paciente.apellido_paterno[0]}{self.paciente.apellido_materno[0]}{self.paciente.nombre_paciente[0]}_{self.ecg.fecha_informe.strftime("%Y%m%d_%H:%M:%S")}'
