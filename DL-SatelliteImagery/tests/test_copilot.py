"""
Tests for the RAG copilot and data ingestion.
"""

import pytest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.copilot.agent import CopilotAgent, CopilotResponse
from disaster_assessment.copilot.rag.ingestion import RAGIngestor
from disaster_assessment.copilot.rag.retriever import RAGRetriever
from disaster_assessment.data.ingestion.base import DataProvider, DataProduct
from disaster_assessment.data.ingestion.cache import DataCache
from disaster_assessment.data.ingestion.scheduler import IngestionScheduler


class TestCopilotAgent:
    def test_init(self):
        agent = CopilotAgent()
        assert agent.provider == "ollama"

    def test_query_without_llm(self):
        agent = CopilotAgent()
        response = agent.query("hello")
        assert isinstance(response, CopilotResponse)

    def test_bind_analysis(self):
        agent = CopilotAgent()

        class MockReport:
            def to_dict(self):
                return {
                    "severity_score": 75,
                    "severity_level": "HIGH",
                    "flood_percentage": 50,
                    "building_damage": {"damage_mode": "estimated"},
                }

        agent.bind_analysis(MockReport())
        response = agent.query("get damage statistics")
        assert response.status == "COMPLETED"

    def test_flood_extent_tool(self):
        agent = CopilotAgent()

        class MockReport:
            def to_dict(self):
                return {"flood_percentage": 42.5, "flood_area_m2": 1000.0, "flood_area_km2": 0.001}

        agent.bind_analysis(MockReport())
        response = agent.query("get flood extent")
        assert "42.5" in response.answer or "flood" in response.answer.lower()


class TestRAGIngestor:
    def test_ingest_text(self):
        ingestor = RAGIngestor(chunk_size=100, overlap=10)
        chunks = ingestor.ingest_text("This is a test document about disaster response.")
        assert len(chunks) > 0
        assert ingestor.num_documents > 0

    def test_ingest_file(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(tmp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("Disaster response guidelines for flood assessment.")
            ingestor = RAGIngestor()
            chunks = ingestor.ingest_file(test_file)
            assert len(chunks) > 0
        finally:
            shutil.rmtree(tmp_dir)


class TestRAGRetriever:
    def test_retrieve(self):
        ingestor = RAGIngestor(chunk_size=200, overlap=0)
        chunks = ingestor.ingest_text(
            "Flood assessment requires water level data. "
            "Building damage needs pre/post imagery."
        )
        assert len(chunks) > 0
        retriever = RAGRetriever()
        retriever.index(chunks)
        results = retriever.retrieve("flood water", top_k=2)
        assert len(results) > 0

    def test_retrieve_empty(self):
        retriever = RAGRetriever()
        results = retriever.retrieve("anything")
        assert len(results) == 0


class TestDataCache:
    def test_put_and_get(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            cache = DataCache(cache_dir=tmp_dir)
            test_file = os.path.join(tmp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            cache.put("key1", test_file, "test_source")
            assert cache.has("key1")
            entry = cache.get("key1")
            assert entry is not None
            assert entry.source == "test_source"
        finally:
            shutil.rmtree(tmp_dir)

    def test_make_key(self):
        cache = DataCache(cache_dir=tempfile.mkdtemp())
        key = cache.make_key(lat=28.6, lon=77.2, date="2024-01-01")
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex

    def test_count(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            cache = DataCache(cache_dir=tmp_dir)
            assert cache.count == 0
        finally:
            shutil.rmtree(tmp_dir)


class TestIngestionScheduler:
    def test_add_task(self):
        scheduler = IngestionScheduler()
        scheduler.add_task("daily_sentinel", "sentinel", interval_hours=24)
        status = scheduler.get_status()
        assert "daily_sentinel" in status
