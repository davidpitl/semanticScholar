import os
import gzip
import json
import fasttext
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

input_dir = '/Users/dpfernandez/DATA/semantic-scholar/samples/abstracts'

# leer listado de ficheros .gz del directorio de abstracts
archivos_gz = [archivo for archivo in os.listdir(input_dir) if archivo.endswith('.gz')]

# descomprimir fichero .gz en memoria
for archivo_gz in archivos_gz:
    print(f'Processing: {archivo_gz}')
    with gzip.open(input_dir + '/' + archivo_gz, 'rt') as archivo_gz_file:
        contenido_descomprimido = archivo_gz_file.read()
        objetos_json = [json.loads(objeto) for objeto in contenido_descomprimido.strip().split('\n')]
        for i, objeto in enumerate(objetos_json, start=1):
            # detect language
            lang_prediction = model.predict(objeto["abstract"].replace('\n', ' '))
            if '__eng_' in str(lang_prediction[0]) and lang_prediction[1][0] > 0.6:
                print(f"Información del objeto {i}:")
                print("corpusid:", objeto["corpusid"])
                print("abstract:", objeto["abstract"])
                print("updated:", objeto["updated"])
                print()
            # remove "Abstract:" del ppio, "PROBLEM TO BE SOLVED: "
            # trim



# parsear fichero json descomprimido


# recorrer cada abstract uno a uno
# leer texto y metadatos

# llamar a API de embedding en local o remoto

# guardar vector de embedding en el mapa

# guardar mapa a disco



