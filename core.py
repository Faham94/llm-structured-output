from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser
import json


root_dir = Path(__file__).resolve().parents[1]

load_dotenv(str(root_dir / ".env"))

# Read API key from environment (set MISTRAL_API_KEY in your .env)
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not mistral_api_key:
    raise RuntimeError("MISTRAL_API_KEY not found in environment or .env")

# Initialize model
model = ChatMistralAI(
    model="mistral-small-latest",
    api_key=mistral_api_key,
)

class Movie(BaseModel):  #schema 

    title:str

    release_year:Optional[int]

    genre:List[str]

    director:Optional[str]

    cast:List[str]

    rating:Optional[float]

    summary:str 

parser=PydanticOutputParser(pydantic_object=Movie) #check krega sari information sahi k nh

# Correct Prompt Template ✅

prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract movie information from the paragraph

{format_instructions}
"""),
    ("human", "{paragraph}"),
])

# Take input from user

para = input("Give your paragraph: ")

# Create final prompt


# Build a simple final prompt string (safer than relying on prompt.invoke API)

final_prompt_text = (
    "Extract movie information from the paragraph\n\n"
    + "Return ONLY valid JSON that matches the schema. Do not add extra text.\n\n"
    + parser.get_format_instructions()
    + "\n\nParagraph:\n"
    + para
)

# Get response
response = model.invoke(final_prompt_text)

# Get raw text from response (support different response types)
raw = getattr(response, "content", None) or str(response)

print("\n🔹 Extracted Info:\n")

# Try to parse as JSON then validate with Pydantic `Movie` model
try:
    data = json.loads(raw)
    movie = Movie.parse_obj(data)
    print(json.dumps(movie.model_dump(), indent=2))
except Exception:
    try:
        # Try langchain parser as fallback
        parsed = parser.parse(raw)
        # `parser.parse` may return a dict or a pydantic object
        if hasattr(parsed, "model_dump"):
            print(json.dumps(parsed.model_dump(), indent=2))
        elif isinstance(parsed, dict):
            print(json.dumps(parsed, indent=2))
        else:
            # last resort
            print(str(parsed))
    except Exception as e:
        print("Failed to parse model output as JSON. Raw response:\n")
        print(raw)
        print("\nParse error:\n", e)