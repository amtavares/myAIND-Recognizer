texto = "Nos que aqui estamos por vos esperamos"
n = 5
linhas = [texto[i:i+n] for i in range(0, len(texto), n)]

print(linhas)