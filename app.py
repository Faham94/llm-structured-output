import streamlit as st
from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
import json

# -----------------------------
# LOAD ENV (FIXED)
# -----------------------------
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

# -----------------------------
# UI START
# -----------------------------
st.set_page_config(page_title="🎬 Movie Extractor")

st.title("🎬 Movie Info Extractor")

# Debug (remove later)
# st.write("API KEY:", mistral_api_key)

if not mistral_api_key:
    st.error("❌ MISTRAL_API_KEY not found. Check your .env file")
    st.stop()

# -----------------------------
# MODEL
# -----------------------------
model = ChatMistralAI(
    model="mistral-small-latest",
    api_key=mistral_api_key,
)

# -----------------------------
# SCHEMA
# -----------------------------
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

# -----------------------------
# INPUT
# -----------------------------
paragraph = st.text_area("Enter movie paragraph:")

if st.button("Extract"):

    if not paragraph.strip():
        st.warning("⚠️ Enter something first")
    else:
        with st.spinner("Processing..."):

            prompt = (
                "Extract movie info and return ONLY JSON.\n\n"
                + parser.get_format_instructions()
                + "\n\nParagraph:\n"
                + paragraph
            )

            try:
                response = model.invoke(prompt)
                raw = getattr(response, "content", str(response))

                # Try to load as JSON first
                success = False
                try:
                    data = json.loads(raw)
                except Exception:
                    data = None

                if data is not None:
                    try:
                        # Pydantic v2 uses model_validate, v1 uses parse_obj
                        if hasattr(Movie, "model_validate"):
                            movie = Movie.model_validate(data)
                        else:
                            movie = Movie.parse_obj(data)

                        # Normalize output dict across pydantic versions
                        if hasattr(movie, "model_dump"):
                            out = movie.model_dump()
                        elif hasattr(movie, "dict"):
                            out = movie.dict()
                        else:
                            out = movie

                        st.success("✅ Done")
                        st.json(out)
                        success = True
                    except Exception:
                        success = False

                if not success:
                    try:
                        parsed = parser.parse(raw)
                        if hasattr(parsed, "model_dump"):
                            st.json(parsed.model_dump())
                        elif isinstance(parsed, dict):
                            st.json(parsed)
                        else:
                            st.write(parsed)
                    except Exception as e:
                        st.error("❌ Failed to parse model output")
                        st.text(str(e))
                        st.text("Raw response:")
                        st.write(raw)

            except Exception as e:
                st.error("❌ Error")
                st.text(str(e))