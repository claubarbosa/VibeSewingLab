from __future__ import annotations

import json
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cv2
import faiss
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from sentence_transformers import SentenceTransformer


# ----------------------------
# Domain
# ----------------------------


@dataclass
class MaterialItem:
    name: str
    category: str
    quantity: float
    unit: str
    estimated_unit_cost: float
    notes: str = ""

    @property
    def subtotal(self) -> float:
        return self.quantity * self.estimated_unit_cost


@dataclass
class CostBreakdown:
    materials_cost: float
    labor_hours: float
    labor_hour_rate: float
    overhead_percentage: float
    margin_percentage: float

    @property
    def labor_cost(self) -> float:
        return self.labor_hours * self.labor_hour_rate

    @property
    def overhead_cost(self) -> float:
        return (self.materials_cost + self.labor_cost) * (self.overhead_percentage / 100)

    @property
    def base_cost(self) -> float:
        return self.materials_cost + self.labor_cost + self.overhead_cost

    @property
    def suggested_price(self) -> float:
        return self.base_cost * (1 + self.margin_percentage / 100)


@dataclass
class SewingProject:
    title: str
    description: str
    source_type: str
    id: str = field(default_factory=lambda: str(uuid4()))
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    image_analysis: Dict[str, Any] = field(default_factory=dict)
    materials: List[MaterialItem] = field(default_factory=list)
    costing: Dict[str, Any] = field(default_factory=dict)
    creative_assets: Dict[str, Any] = field(default_factory=dict)
    production_plan: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source_type": self.source_type,
            "concepts": self.concepts,
            "image_analysis": self.image_analysis,
            "materials": [asdict(item) for item in self.materials],
            "costing": self.costing,
            "creative_assets": self.creative_assets,
            "production_plan": self.production_plan,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ----------------------------
# LLM Providers (real only)
# ----------------------------


class LLMProviderError(RuntimeError):
    pass


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiLLMProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._model = None

    def _lazy_init(self):
        if not self.api_key:
            raise LLMProviderError("GEMINI_API_KEY is not set.")
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str) -> str:
        self._lazy_init()
        response = self._model.generate_content(prompt)
        text = getattr(response, "text", "")
        if not text:
            raise LLMProviderError("Gemini returned an empty response.")
        return text


class OpenAILLMProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise LLMProviderError("OPENAI_API_KEY is not set.")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(model=self.model_name, input=prompt)
        text = getattr(response, "output_text", "")
        if not text:
            raise LLMProviderError("OpenAI returned an empty response.")
        return text


class OllamaLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "qwen2.5:3b-instruct", host: Optional[str] = None):
        self.model_name = model_name
        self.host = host or os.getenv("OLLAMA_HOST")
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self.host) if self.host else Client()
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._lazy_client()
        response = client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in handmade bag sewing workflows."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7},
        )
        text = response.get("message", {}).get("content", "")
        if not text:
            raise LLMProviderError("Ollama returned an empty response.")
        return text


class LocalTransformersLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_name = model_name
        self._pipeline = None

    def _lazy_init(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )

    def generate(self, prompt: str) -> str:
        self._lazy_init()
        result = self._pipeline(
            prompt,
            max_new_tokens=420,
            do_sample=True,
            temperature=0.7,
            top_p=0.92,
            repetition_penalty=1.05,
        )
        if isinstance(result, list) and result:
            text = str(result[0].get("generated_text", ""))
            if text:
                return text
        raise LLMProviderError("Transformers model returned an empty response.")


def _is_colab_runtime() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def get_default_llm_provider(provider_name: str = "auto", model_name: Optional[str] = None) -> BaseLLMProvider:
    provider = (provider_name or "auto").lower()

    if provider == "gemini":
        return GeminiLLMProvider(model_name=model_name or "gemini-1.5-flash")
    if provider == "openai":
        return OpenAILLMProvider(model_name=model_name or "gpt-4o-mini")
    if provider == "ollama":
        return OllamaLLMProvider(model_name=model_name or "qwen2.5:3b-instruct")
    if provider in {"local", "transformers", "qwen"}:
        return LocalTransformersLLMProvider(model_name=model_name or "Qwen/Qwen2.5-3B-Instruct")

    if _is_colab_runtime():
        return LocalTransformersLLMProvider(model_name=model_name or "Qwen/Qwen2.5-3B-Instruct")
    return OllamaLLMProvider(model_name=model_name or "qwen2.5:3b-instruct")


# ----------------------------
# Services
# ----------------------------


class IdeaGeneratorService:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    def _extract_json_array(self, text: str):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
        return None

    def generate_concepts(self, user_prompt: str, count: int = 8) -> List[Dict[str, str]]:
        prompt = f"""
You are a creative director specialized in handmade bag products.
Generate {count} concepts in JSON array format.

Customer idea:
{user_prompt}

Return only JSON with this schema:
[
  {{
    "name": "...",
    "target_audience": "...",
    "aesthetic": "...",
    "suggested_materials": "...",
    "differential": "...",
    "difficulty": "easy|medium|hard"
  }}
]
"""
        raw = self.llm_provider.generate(prompt)
        parsed = self._extract_json_array(raw)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("LLM did not return valid JSON for concepts.")

        concepts: List[Dict[str, str]] = []
        for item in parsed[:count]:
            if not isinstance(item, dict):
                continue
            concepts.append(
                {
                    "name": str(item.get("name", "")).strip(),
                    "target_audience": str(item.get("target_audience", "")).strip(),
                    "aesthetic": str(item.get("aesthetic", "")).strip(),
                    "suggested_materials": str(item.get("suggested_materials", "")).strip(),
                    "differential": str(item.get("differential", "")).strip(),
                    "difficulty": str(item.get("difficulty", "medium")).strip().lower(),
                }
            )

        concepts = [c for c in concepts if c["name"] and c["target_audience"]]
        if not concepts:
            raise ValueError("Concept list is empty after JSON parse.")
        return concepts


class ImageAnalysisService:
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        image_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 180)

        mean_color = image_np.mean(axis=(0, 1)).tolist()
        edge_density = float((edges > 0).mean())

        difficulty = "easy"
        if edge_density > 0.08:
            difficulty = "medium"
        if edge_density > 0.15:
            difficulty = "hard"

        return {
            "style": "artisan contemporary",
            "format": "structured rectangle",
            "structure": "semi-rigid",
            "type": "utility bag",
            "difficulty": difficulty,
            "dominant_rgb": [round(c, 2) for c in mean_color],
            "edge_density": round(edge_density, 4),
        }


class SketchGeneratorService:
    def generate_sketch_views(self, title: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        views = ["Front", "Side", "Top"]
        for ax, view in zip(axes, views):
            ax.set_title(f"{title} - {view}")
            body = FancyBboxPatch(
                (0.2, 0.25),
                0.6,
                0.45,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                fill=False,
                linewidth=2,
            )
            handle = Rectangle((0.38, 0.72), 0.24, 0.08, fill=False, linewidth=2)
            ax.add_patch(body)
            ax.add_patch(handle)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        plt.tight_layout()
        return fig


class MaterialsService:
    def suggest_materials(self, project: SewingProject) -> List[MaterialItem]:
        difficulty = (project.image_analysis or {}).get("difficulty", "medium")

        items = [
            MaterialItem("Main Fabric", "fabric", 1.5, "m", 32.0, "Outer shell"),
            MaterialItem("Lining", "lining", 1.2, "m", 18.0, "Inner side"),
            MaterialItem("Interfacing", "structure", 1.0, "m", 22.0, "Support"),
            MaterialItem("Metal Zipper", "hardware", 2.0, "unit", 8.5, "Main closure"),
            MaterialItem("Handles", "component", 1.0, "pair", 24.0, "Reinforced handles"),
            MaterialItem("Premium Thread", "notion", 2.0, "unit", 6.0, "Assembly"),
        ]

        if difficulty == "hard":
            items.append(MaterialItem("Internal Divider", "structure", 1.0, "unit", 14.0, "Organization"))

        return items


class CostService:
    def calculate(
        self,
        materials: List[MaterialItem],
        labor_hours: float = 6.0,
        labor_hour_rate: float = 25.0,
        overhead_percentage: float = 12.0,
        margin_percentage: float = 80.0,
    ) -> Dict[str, float]:
        materials_cost = sum(item.subtotal for item in materials)
        cost = CostBreakdown(
            materials_cost=materials_cost,
            labor_hours=labor_hours,
            labor_hour_rate=labor_hour_rate,
            overhead_percentage=overhead_percentage,
            margin_percentage=margin_percentage,
        )

        return {
            "materials_cost": round(cost.materials_cost, 2),
            "labor_cost": round(cost.labor_cost, 2),
            "overhead_cost": round(cost.overhead_cost, 2),
            "base_cost": round(cost.base_cost, 2),
            "suggested_price": round(cost.suggested_price, 2),
        }


class CreativeStudioService:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    def _extract_json_object(self, text: str):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    def generate_brand_assets(self, project: SewingProject) -> Dict[str, str]:
        prompt = f"""
Create marketing and storytelling assets for this handmade bag project.

Title: {project.title}
Description: {project.description}
Analysis: {project.image_analysis}

Return JSON object with keys:
collection_name, story, description, slogan, instagram, linkedin, facebook, pinterest, etsy, shopee, marketplace, image_prompt, video_prompt
"""
        raw = self.llm_provider.generate(prompt)
        parsed = self._extract_json_object(raw)

        required = [
            "collection_name",
            "story",
            "description",
            "slogan",
            "instagram",
            "linkedin",
            "facebook",
            "pinterest",
            "etsy",
            "shopee",
            "marketplace",
            "image_prompt",
            "video_prompt",
        ]

        if not isinstance(parsed, dict):
            raise ValueError("LLM did not return valid JSON for creative assets.")

        missing = [k for k in required if not str(parsed.get(k, "")).strip()]
        if missing:
            raise ValueError(f"Creative assets missing keys: {missing}")

        return {k: str(parsed[k]).strip() for k in required}


class SimpleRAGService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.texts: List[str] = []
        self.index = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _add_embeddings(self, texts: List[str]):
        if not texts:
            return

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.texts.extend(texts)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))

    def ingest_long_text(self, text: str):
        if not text or not text.strip():
            raise ValueError("Knowledge base text is empty.")

        chunks = [chunk.strip() for chunk in self.splitter.split_text(text) if chunk.strip()]
        if not chunks:
            raise ValueError("Could not generate chunks from knowledge text.")

        self._add_embeddings(chunks)

    def query(self, question: str, top_k: int = 3) -> List[str]:
        if self.index is None or not self.texts:
            return []
        if not question or not question.strip():
            return []

        question_embedding = self.model.encode([question], convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(question_embedding, min(top_k, len(self.texts)))
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

    def answer_with_context(self, question: str, llm_provider: BaseLLMProvider, top_k: int = 3) -> Dict[str, Any]:
        contexts = self.query(question, top_k=top_k)
        if not contexts:
            raise ValueError("Knowledge base is empty. Add content before asking.")

        prompt = f"""
You are a technical sewing assistant.
Use only the context below.

Context:
{chr(10).join([f'- {c}' for c in contexts])}

Question: {question}

Return:
- Technical summary
- Step-by-step action
- Quality precautions
"""
        answer = llm_provider.generate(prompt)
        return {"answer": answer.strip(), "contexts": contexts}


# ----------------------------
# Persistence and orchestration
# ----------------------------


class SQLiteProjectRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                source_type TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def save(self, project: SewingProject):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO projects (id, title, description, source_type, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                project.id,
                project.title,
                project.description,
                project.source_type,
                project.to_json(),
            ),
        )
        conn.commit()
        conn.close()

    def list_projects(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description, source_type FROM projects ORDER BY rowid DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows


class VibeSewingOrchestrator:
    def __init__(self, llm_provider: BaseLLMProvider, repository: SQLiteProjectRepository):
        self.idea_service = IdeaGeneratorService(llm_provider)
        self.image_service = ImageAnalysisService()
        self.sketch_service = SketchGeneratorService()
        self.materials_service = MaterialsService()
        self.cost_service = CostService()
        self.creative_service = CreativeStudioService(llm_provider)
        self.repository = repository

    def create_project_from_text(self, title: str, description: str) -> SewingProject:
        project = SewingProject(title=title, description=description, source_type="text")
        project.concepts = self.idea_service.generate_concepts(description)
        project.materials = self.materials_service.suggest_materials(project)
        project.costing = self.cost_service.calculate(project.materials)
        project.creative_assets = self.creative_service.generate_brand_assets(project)
        project.production_plan = {
            "checklist": [
                "Define pattern",
                "Separate materials",
                "Cut main fabric",
                "Cut lining",
                "Apply structure",
                "Sew pockets",
                "Assemble body",
                "Install hardware",
                "Review finishing",
            ],
            "estimated_days": 2,
        }
        self.repository.save(project)
        return project

    def create_project_from_image(self, title: str, description: str, image: Image.Image) -> SewingProject:
        project = SewingProject(title=title, description=description, source_type="image")
        project.image_analysis = self.image_service.analyze(image)
        project.concepts = self.idea_service.generate_concepts(description or title)
        project.materials = self.materials_service.suggest_materials(project)
        project.costing = self.cost_service.calculate(project.materials, labor_hours=8.0)
        project.creative_assets = self.creative_service.generate_brand_assets(project)
        project.production_plan = {
            "checklist": [
                "Analyze visual reference",
                "Define proportions",
                "Choose materials",
                "Create base pattern",
                "Build prototype",
                "Adjust structure",
                "Finalize sewing",
                "Validate usability",
            ],
            "estimated_days": 3,
        }
        self.repository.save(project)
        return project


def build_dashboard_data(repo: SQLiteProjectRepository) -> pd.DataFrame:
    rows = repo.list_projects()
    return pd.DataFrame(rows, columns=["id", "title", "description", "source_type"])


def build_dashboard_chart(df: pd.DataFrame):
    if df.empty:
        return None
    import plotly.express as px

    return px.histogram(
        df,
        x="source_type",
        color="source_type",
        title="Projects by source type",
    )


# ----------------------------
# App
# ----------------------------


def build_app():
    project_root = Path(os.getenv("VIBE_PROJECT_ROOT", Path.cwd() / "vibe-sewing-lab"))
    project_root.mkdir(parents=True, exist_ok=True)

    db_path = project_root / "data" / "projects.db"
    repo = SQLiteProjectRepository(db_path=db_path)
    rag_service = SimpleRAGService()

    provider_name = os.getenv("VIBE_LLM_PROVIDER", "auto")
    model_name = os.getenv("VIBE_LLM_MODEL", "")

    try:
        llm = get_default_llm_provider(provider_name=provider_name, model_name=model_name or None)
        orchestrator = VibeSewingOrchestrator(llm_provider=llm, repository=repo)
        llm_status = f"Provider active: {llm.__class__.__name__}"
    except Exception as exc:
        orchestrator = None
        llm = None
        llm_status = f"Provider initialization failed: {exc}"

    def ensure_runtime_ready():
        if orchestrator is None or llm is None:
            raise gr.Error(
                "LLM is not available. Configure VIBE_LLM_PROVIDER/VIBE_LLM_MODEL and required credentials."
            )

    def run_text_pipeline(title: str, description: str):
        ensure_runtime_ready()
        if not title or not title.strip():
            raise gr.Error("Project name is required.")
        if not description or not description.strip():
            raise gr.Error("Project description is required.")

        project = orchestrator.create_project_from_text(title.strip(), description.strip())
        materials_df = pd.DataFrame(
            [
                {
                    "name": item.name,
                    "category": item.category,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "unit_cost": item.estimated_unit_cost,
                    "subtotal": item.subtotal,
                }
                for item in project.materials
            ]
        )

        checklist = "\n".join([f"- {item}" for item in project.production_plan["checklist"]])

        return (
            pd.DataFrame(project.concepts),
            project.image_analysis,
            materials_df,
            pd.DataFrame([project.costing]),
            project.creative_assets["story"],
            project.creative_assets["instagram"],
            project.creative_assets["linkedin"],
            checklist,
            project.to_json(),
        )

    def run_image_pipeline(title: str, description: str, image):
        ensure_runtime_ready()
        if image is None:
            raise gr.Error("Image is required.")

        clean_title = (title or "Image Project").strip()
        clean_desc = (description or "").strip()
        pil_image = Image.fromarray(image.astype("uint8"))

        project = orchestrator.create_project_from_image(clean_title, clean_desc, pil_image)
        materials_df = pd.DataFrame(
            [
                {
                    "name": item.name,
                    "category": item.category,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "unit_cost": item.estimated_unit_cost,
                    "subtotal": item.subtotal,
                }
                for item in project.materials
            ]
        )

        fig = orchestrator.sketch_service.generate_sketch_views(project.title)
        checklist = "\n".join([f"- {item}" for item in project.production_plan["checklist"]])

        return (
            project.image_analysis,
            materials_df,
            pd.DataFrame([project.costing]),
            project.creative_assets["description"],
            project.creative_assets["image_prompt"],
            checklist,
            fig,
            project.to_json(),
        )

    def run_rag_pipeline(knowledge_text: str, question: str):
        ensure_runtime_ready()
        if not knowledge_text or not knowledge_text.strip():
            raise gr.Error("Knowledge text is required.")
        if not question or not question.strip():
            raise gr.Error("Question is required.")

        rag_service.ingest_long_text(knowledge_text)
        result = rag_service.answer_with_context(question=question.strip(), llm_provider=llm, top_k=3)
        contexts_md = "\n\n".join([f"- {c}" for c in result["contexts"]])
        return result["answer"], contexts_md

    def refresh_dashboard():
        df = build_dashboard_data(repo)
        chart = build_dashboard_chart(df)
        return df, chart

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# Vibe Sewing Lab

**AI-powered Creative Sewing Assistant**

O Vibe Sewing Lab é um assistente criativo inteligente desenvolvido para profissionais da costura artesanal, especialmente para artesãs e criadores que confeccionam bolsas feitas à mão.

Ele não substitui a criatividade humana.

Ele a potencializa.
            """
        )
        gr.Markdown(llm_status)

        with gr.Tabs():
            with gr.Tab("Idea Generator"):
                title_input = gr.Textbox(label="Project Name")
                description_input = gr.Textbox(label="Project Description", lines=5)
                text_button = gr.Button("Generate Full Project", variant="primary")

                concepts_output = gr.Dataframe(label="Generated Concepts")
                analysis_output = gr.JSON(label="Visual Analysis")
                materials_output = gr.Dataframe(label="Material Plan")
                costing_output = gr.Dataframe(label="Costing")
                story_output = gr.Textbox(label="Story", lines=8)
                insta_output = gr.Textbox(label="Instagram Copy", lines=4)
                linkedin_output = gr.Textbox(label="LinkedIn Copy", lines=4)
                checklist_output = gr.Textbox(label="Production Checklist", lines=10)
                json_output = gr.Code(label="Project JSON", language="json")

                text_button.click(
                    fn=run_text_pipeline,
                    inputs=[title_input, description_input],
                    outputs=[
                        concepts_output,
                        analysis_output,
                        materials_output,
                        costing_output,
                        story_output,
                        insta_output,
                        linkedin_output,
                        checklist_output,
                        json_output,
                    ],
                )

            with gr.Tab("Image Analysis"):
                image_title = gr.Textbox(label="Project Name")
                image_description = gr.Textbox(label="Extra Description", lines=4)
                image_input = gr.Image(label="Upload Image", type="numpy")
                image_button = gr.Button("Analyze Image and Generate", variant="primary")

                image_analysis_output = gr.JSON(label="Analysis")
                image_materials_output = gr.Dataframe(label="Materials")
                image_costing_output = gr.Dataframe(label="Costing")
                image_desc_output = gr.Textbox(label="Product Description", lines=5)
                image_prompt_output = gr.Textbox(label="Image Prompt", lines=4)
                image_checklist_output = gr.Textbox(label="Checklist", lines=8)
                image_sketch_output = gr.Plot(label="Technical Sketch")
                image_json_output = gr.Code(label="Project JSON", language="json")

                image_button.click(
                    fn=run_image_pipeline,
                    inputs=[image_title, image_description, image_input],
                    outputs=[
                        image_analysis_output,
                        image_materials_output,
                        image_costing_output,
                        image_desc_output,
                        image_prompt_output,
                        image_checklist_output,
                        image_sketch_output,
                        image_json_output,
                    ],
                )

            with gr.Tab("Technical Assistant (RAG)"):
                rag_context_input = gr.Textbox(label="Technical Knowledge Base", lines=10)
                rag_question_input = gr.Textbox(label="Technical Question", lines=3)
                rag_button = gr.Button("Ask Technical Assistant", variant="primary")
                rag_answer_output = gr.Textbox(label="Answer", lines=10)
                rag_contexts_output = gr.Markdown(label="Retrieved Context")

                rag_button.click(
                    fn=run_rag_pipeline,
                    inputs=[rag_context_input, rag_question_input],
                    outputs=[rag_answer_output, rag_contexts_output],
                )

            with gr.Tab("Project Dashboard"):
                dash_button = gr.Button("Refresh Dashboard")
                dash_table = gr.Dataframe(label="Saved Projects")
                dash_chart = gr.Plot(label="Analytics")
                dash_button.click(fn=refresh_dashboard, inputs=[], outputs=[dash_table, dash_chart])

    return demo


if __name__ == "__main__":
    app = build_app()
    try:
        app.launch(share=True, debug=True)
    except Exception:
        app.launch(share=False, debug=True)
