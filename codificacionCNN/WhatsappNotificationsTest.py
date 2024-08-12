import pywhatkit
import time

phone_number = "+525539522812"  # El número de teléfono debe estar en formato internacional con '+' al inicio y sin espacios.
message = "Hola, esto es un mensaje enviado desde Python."  # El mensaje que deseas enviar.

# Enviamos un mensaje ahorita mismo
time = time.localtime()

pywhatkit.sendwhatmsg(phone_number, message, time.tm_hour, time.tm_min + 1)