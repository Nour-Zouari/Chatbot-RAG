import socket

host = "api.grok.ai"
port = 443

try:
    sock = socket.create_connection((host, port), timeout=5)
    print("Connexion OK")
    sock.close()
except Exception as e:
    print("Erreur de connexion :", e)
