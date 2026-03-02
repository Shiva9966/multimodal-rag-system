import os
from enum import Enum
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Modality(str, Enum):
    CSV     = "csv"
    IMAGE   = "image"
    PDF     = "pdf"
    WEBPAGE = "webpage"
    GENERAL = "general"


def route_query(query: str) -> Modality:
    """
    Use Groq LLaMA to semantically classify the query into a modality.
    Falls back to GENERAL if uncertain or API fails.
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a query classifier. Classify the user query into exactly one category:

csv     - questions about structured data, spreadsheets, statistics, records, rows, survival rates, 
          passengers, fares, who died, who survived, how many, averages, trends in data
image   - questions about diagrams, pictures, photos, what is shown, architecture diagrams, 
          OCR text, visual content, what is drawn, components in a diagram
pdf     - questions about documents, resumes, CVs, research papers, reports, projects, 
          skills, education, experience, professional background, certifications
webpage - questions about web articles, concepts, techniques, definitions, explanations,
          how something works, what is a concept, types of something
general - cross-document questions, comparisons across sources, or completely unclear

Reply with ONLY one word: csv, image, pdf, webpage, or general"""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0,
            max_tokens=10,
        )

        result = response.choices[0].message.content.strip().lower()

        mapping = {
            "csv":     Modality.CSV,
            "image":   Modality.IMAGE,
            "pdf":     Modality.PDF,
            "webpage": Modality.WEBPAGE,
            "general": Modality.GENERAL,
        }
        return mapping.get(result, Modality.GENERAL)

    except Exception as e:
        print(f"  Router LLM failed ({e}), falling back to GENERAL")
        return Modality.GENERAL


def describe_route(modality: Modality) -> str:
    return {
        Modality.CSV:     "📊 Searching CSV data",
        Modality.IMAGE:   "🖼️ Searching image content",
        Modality.PDF:     "📄 Searching PDF documents",
        Modality.WEBPAGE: "🌐 Searching webpage content",
        Modality.GENERAL: "🔍 General search across all documents",
    }.get(modality, "🔍 General search")