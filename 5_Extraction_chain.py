# Official documentation in English: https://python.langchain.com/docs/tutorials/extraction/

# Necessary libraries
from typing import Optional
from pydantic import BaseModel, Field

"""
The first thing we are going to do is define an output schema for the model describing the information we want to extract.
For this we are going to use Pydantic and define a schema
"""

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the Person entity.
    # This doc-string is sent to the LLM as the description of the Person schema,
    # and it can help to improve extraction results.

    # Note:
    # 1. Each field is `optional` -- this allows the model to decide not to extract it if not present.
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

