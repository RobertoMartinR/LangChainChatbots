 # Official documentation in English: https://python.langchain.com/docs/tutorials/classification/

"""
Tagging is assigning labels to a document with classes such as:
    - Sentiment: Positive, Negative, Neutral
    - Language: English, Spanish, French...
    - Style: Formal, Informal, Technical...
    - Topic: Sports, Politics, Technology...
    - Political tendency: Left, Right, Center

Tagging has components such as:
    - function: function to specify to the model how to tag the document
    - schema: how we want to tag the document 
"""

 # Import necessary libraries
import getpass
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Request Google Gemini API key if not defined
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

# Initialize Gemini chat model
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


# Define a function to tag the document and a prompt for the LLM using a Pydantic model.
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm_prompt = ChatPromptTemplate.from_template(
    """
Extract the requested information from the following text.

Extract only the properties mentioned in the 'Classification' function.

Text:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")



# Structured LLM
structured_llm = llm.with_structured_output(Classification)

# Input text to classify
inp = "Estoy muy feliz de estar aprendiendo Langchain! Creo que me puede ser muy útil en mi carrera profesional."
prompt = llm_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

response
print(response)
# Expected output: sentiment='Positive' aggressiveness=1 language='Spanish'


# If we want dictionary output we can use the .model_dump() method
inp = "Estoy muy enfadado porque me han despedido del trabajo!" # The output may vary depending on the model used.
prompt = llm_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

print(response.model_dump())
# Expected output: {'sentiment': 'negative', 'aggressiveness': 8, 'language': 'Spanish'}


# Another thing that can be done is to control the model output by defining a more detailed output schema.
"""
Things that can be defined are:
    - The possible values for each property
    - Description of each property so the model understands it better
    - Whether a property is mandatory or not
"""

# Class with detailed output schema using Enum
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

# Prompt for the LLM
llm_prompt = ChatPromptTemplate.from_template(
    """
Extract the requested information from the following text.

Extract only the properties mentioned in the 'Classification' function.

Text:
{input}
"""
)

# Structured LLM with detailed output schema that in this case must be done with OpenAI because it's only implemented for these models.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
    Classification
)

# Input text to classify
inp = "Estoy muy feliz de estar aprendiendo Langchain! Creo que me puede ser muy útil en mi carrera profesional."
prompt = llm_prompt.invoke({"input": inp})  
response = llm.invoke(prompt)
print(response)