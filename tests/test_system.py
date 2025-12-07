import pytest
from django.test import Client
from pipeline.models import ProjectTask, RegulatoryRule
from datetime import date

@pytest.mark.django_db
class TestPropLensSystem:

    def setup_method(self):
        self.client = Client()

    # --- 1. Database Model Tests ---
    def test_create_project_task(self):
        """Test that the Schedule Model works correctly"""
        task = ProjectTask.objects.create(
            task_id=999,
            task_name="Demo Verification Task",
            duration_days=5,
            start_date=date(2025, 1, 1),
            finish_date=date(2025, 1, 6)
        )
        assert task.pk is not None
        assert ProjectTask.objects.filter(task_id=999).exists()

    def test_create_regulatory_rule(self):
        """Test that the URA Rule Model works correctly"""
        rule = RegulatoryRule.objects.create(
            rule_id="TEST-Q1",
            rule_summary="This is a test summary for pytest.",
            measurement_basis="Test Basis"
        )
        assert rule.pk is not None
        assert str(rule) == "TEST-Q1"

    # --- 2. API Endpoint Tests ---
    def test_health_check_api(self):
        """Verify the API documentation endpoint is live"""
        response = self.client.get("/api/docs")
        assert response.status_code == 200

    def test_tasks_endpoint(self):
        """Verify GET /api/tasks returns a list"""
        ProjectTask.objects.create(task_id=1, task_name="T1", duration_days=1)

        response = self.client.get("/api/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) >= 1

    def test_rules_endpoint(self):
        """Verify GET /api/rules returns a list"""
        RegulatoryRule.objects.create(rule_id="R1", rule_summary="S1", measurement_basis="B1")

        response = self.client.get("/api/rules")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert response.json()[0]['rule_id'] == "R1"

    def test_search_endpoint_structure(self):
        response = self.client.get("/api/search?query=gymnasium")
        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert "results" in data
        assert data["query"] == "gymnasium"