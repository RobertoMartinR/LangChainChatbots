 # Documentación oficial en inglés: https://python.langchain.com/docs/tutorials/classification/
"""
En español:
El etiquetado es asignar etiquetas a un documento con clases como:
    - Sentimiento: Positivo, Negativo, Neutral
    - Idioma: Inglés, Español, Francés...
    - Estilo: Formal, Informal, Técnico...
    - Tema: Deportes, Política, Tecnología...
    - Tendencia política: Izquierda, Derecha, Centro

El etiquetado tiene componentes como:
    - función: función para especificar al modelo cómo etiquetar el documento
    - esquema: cómo queremos etiquetar el documento 
"""

 # Importar librerías necesarias
import getpass
import os
from dotenv import load_dotenv


# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Solicitar la clave de API de Google Gemini si no está definida
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Introduce la clave API para Google Gemini: ")

from langchain.chat_models import init_chat_model

# Inicializar el modelo de chat de Gemini
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


# Definimos una función para etiquetar el documento y un prompt para el LLM usando un modelo Pydantic.
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm_prompt = ChatPromptTemplate.from_template(
    """
Extrae la información solicitada del siguiente texto.

Extrae únicamente las propiedades mencionadas en la función 'Classification'.

Texto:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="El sentimiento del texto")
    aggressiveness: int = Field(
        description="Qué tan agresivo es el texto en una escala del 1 al 10"
    )
    language: str = Field(description="El idioma en el que está escrito el texto")



# LLM estructurado
structured_llm = llm.with_structured_output(Classification)

# Texto de entrada a clasificar
inp = "Estoy muy feliz de estar aprendiendo Langchain! Creo que me puede ser muy útil en mi carrera profesional."
prompt = llm_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

response
print(response)
# Salida esperada: sentiment='Positive' aggressiveness=1 language='Spanish'


# Si queremos la salida como diccionario podemos usar el método .model_dump()
inp = "Estoy muy enfadado porque me han despedido del trabajo!" # La salida puede variar dependiendo del modelo usado.
prompt = llm_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

print(response.model_dump())
# Salida esperada: {'sentiment': 'negative', 'aggressiveness': 8, 'language': 'Spanish'}


# Otra cosa que se puede hacer es controlar la salida del modelo definiendo un esquema de salida más detallado.
"""
Cosas que se pueden definir son:
    - Los posibles valores para cada propiedad
    - Descripción de cada propiedad para que el modelo la entienda mejor
    - Si una propiedad es obligatoria o no
"""

# Clase con esquema de salida detallado utilizando Enum
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

# Prompt para el LLM
llm_prompt = ChatPromptTemplate.from_template(
    """
Extrae la información solicitada del siguiente texto.

Extrae únicamente las propiedades mencionadas en la función 'Classification'.

Texto:
{input}
"""
)

# LLM estructurado con el esquema de salida detallado que en este caso se debe hacer con OpenAI porque solo está implementado para estos modelos.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
    Classification
)

# Texto de entrada a clasificar
inp = "Estoy muy feliz de estar aprendiendo Langchain! Creo que me puede ser muy útil en mi carrera profesional."
prompt = llm_prompt.invoke({"input": inp})  
response = llm.invoke(prompt)
print(response)