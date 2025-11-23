"""Tests for the main module."""

import os
import json
import pytest
from unittest.mock import Mock, patch, call
from src.main import (
    extract_fnol_information_batch, 
    assess_claim_severity,
    route_claims_to_queues
)
from src.models import (
    FNOLInfo, VehicleInfo, DamageInfo, BatchFNOLInfo,
    SeverityAssessment, BatchSeverityAssessment,
    QueueRouting, BatchQueueRouting
)


def test_extract_fnol_information_batch_multiple_claims():
    """Test successful batch extraction of multiple claims."""
    mock_response = Mock()
    mock_response.text = json.dumps({
        "claims": [
            {
                "claim_id": "C001",
                "incident_date": "2024-12-15",
                "incident_location": "Highway",
                "policyholder_name": "John Smith",
                "contact_phone": None,
                "contact_email": None,
                "vehicle": {
                    "make": "Toyota",
                    "model": "Camry",
                    "year": 2018,
                    "vin": None,
                    "license_plate": None,
                    "color": None
                },
                "damage": {
                    "description": "Small chip in windshield",
                    "location": "windshield",
                    "severity": "minor",
                    "estimated_repair_cost": None
                },
                "incident_description": "Rock hit windshield",
                "other_parties_involved": False,
                "police_report_filed": False
            },
            {
                "claim_id": "C002",
                "incident_date": "2024-12-16",
                "incident_location": "Grocery store parking lot",
                "policyholder_name": "Sarah Johnson",
                "contact_phone": None,
                "contact_email": None,
                "vehicle": {
                    "make": "Honda",
                    "model": "Civic",
                    "year": 2020,
                    "vin": None,
                    "license_plate": None,
                    "color": None
                },
                "damage": {
                    "description": "Dented rear bumper and broken taillight",
                    "location": "rear bumper",
                    "severity": "moderate",
                    "estimated_repair_cost": None
                },
                "incident_description": "Hit and run in parking lot",
                "other_parties_involved": True,
                "police_report_filed": False
            }
        ]
    })
    
    with patch("src.main.genai.Client") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.return_value = mock_response
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            result = extract_fnol_information_batch("Multiple claims text")
            
        assert isinstance(result, BatchFNOLInfo)
        assert len(result.claims) == 2
        assert result.claims[0].claim_id == "C001"
        assert result.claims[1].claim_id == "C002"
        assert result.claims[0].vehicle.make == "Toyota"
        assert result.claims[1].vehicle.make == "Honda"


def test_extract_fnol_information_batch_validation():
    """Test batch validation functionality."""
    batch_info = BatchFNOLInfo(claims=[
        FNOLInfo(
            claim_id="C001",
            incident_date="2024-12-15",
            incident_location="Highway",
            policyholder_name="John Smith",
            damage=DamageInfo(description="Windshield chip")
        ),
        FNOLInfo(
            claim_id="C002",
            # Missing required fields
        )
    ])
    
    validation = batch_info.validate_all()
    
    assert validation["total_claims"] == 2
    assert validation["valid_claims"] == 1
    assert validation["invalid_claims"] == 1
    assert len(validation["invalid_details"]) == 1


def test_assess_claim_severity_success():
    """Test successful severity assessment for multiple claims."""
    batch_info = BatchFNOLInfo(claims=[
        FNOLInfo(
            claim_id="C001",
            damage=DamageInfo(
                description="Small windshield chip",
                location="windshield"
            )
        ),
        FNOLInfo(
            claim_id="C002",
            damage=DamageInfo(
                description="Severe collision damage with airbag deployment",
                location="front bumper, hood, engine compartment"
            )
        )
    ])
    
    mock_response = Mock()
    mock_response.text = json.dumps({
        "assessments": [
            {
                "claim_id": "C001",
                "severity": "Minor",
                "estimated_cost": 250.0,
                "reasoning": "Small cosmetic damage to windshield"
            },
            {
                "claim_id": "C002",
                "severity": "Major",
                "estimated_cost": 15000.0,
                "reasoning": "Severe structural damage with airbag deployment"
            }
        ]
    })
    
    with patch("src.main.genai.Client") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.return_value = mock_response
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            result = assess_claim_severity(batch_info)
            
        assert isinstance(result, BatchSeverityAssessment)
        assert len(result.assessments) == 2
        assert result.assessments[0].severity == "Minor"
        assert result.assessments[1].severity == "Major"
        assert result.assessments[0].estimated_cost == 250.0
        assert result.assessments[1].estimated_cost == 15000.0
        total_cost = sum(a.estimated_cost for a in result.assessments)
        assert total_cost == 15250.0


def test_assess_claim_severity_breakdown():
    """Test severity breakdown statistics."""
    assessment = BatchSeverityAssessment(assessments=[
        SeverityAssessment(claim_id="C001", severity="Minor", estimated_cost=300.0),
        SeverityAssessment(claim_id="C002", severity="Minor", estimated_cost=500.0),
        SeverityAssessment(claim_id="C003", severity="Moderate", estimated_cost=3000.0),
        SeverityAssessment(claim_id="C004", severity="Major", estimated_cost=20000.0)
    ])
    
    breakdown = assessment.get_severity_breakdown()
    
    assert breakdown["Minor"] == 2
    assert breakdown["Moderate"] == 1
    assert breakdown["Major"] == 1
    total_cost = sum(a.estimated_cost for a in assessment.assessments)
    assert total_cost == 23800.0


def test_assess_claim_severity_missing_api_key():
    """Test that ValueError is raised when API key is missing for Stage II."""
    batch_info = BatchFNOLInfo(claims=[
        FNOLInfo(claim_id="C001", damage=DamageInfo(description="Test damage"))
    ])
    
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable is not set"):
            assess_claim_severity(batch_info)


def test_route_claims_to_queues_success():
    """Test successful queue routing for multiple claims (Stage III)."""
    # Create Stage I data
    batch_info = BatchFNOLInfo(claims=[
        FNOLInfo(
            claim_id="C001",
            damage=DamageInfo(description="Windshield chip", location="Windshield")
        ),
        FNOLInfo(
            claim_id="C002",
            damage=DamageInfo(description="Front bumper dent", location="Front bumper")
        ),
        FNOLInfo(
            claim_id="C003",
            damage=DamageInfo(description="Total loss from collision", location="Multiple areas")
        )
    ])
    
    # Create Stage II data
    batch_severity = BatchSeverityAssessment(assessments=[
        SeverityAssessment(claim_id="C001", severity="Minor", estimated_cost=150.0),
        SeverityAssessment(claim_id="C002", severity="Minor", estimated_cost=500.0),
        SeverityAssessment(claim_id="C003", severity="Major", estimated_cost=25000.0)
    ])
    
    # Mock LLM response for Stage III
    mock_response = Mock()
    mock_response.text = json.dumps({
        "routings": [
            {
                "claim_id": "C001",
                "queue": "glass",
                "priority": 4,
                "reasoning": "Minor glass-only damage"
            },
            {
                "claim_id": "C002",
                "queue": "fast_track",
                "priority": 3,
                "reasoning": "Minor non-glass damage"
            },
            {
                "claim_id": "C003",
                "queue": "total_loss",
                "priority": 1,
                "reasoning": "Major damage, total loss"
            }
        ]
    })
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        with patch("google.genai.Client") as mock_client:
            mock_instance = Mock()
            mock_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_instance
            
            result = route_claims_to_queues(batch_info, batch_severity)
            
            assert len(result.routings) == 3
            assert result.routings[0].claim_id == "C001"
            assert result.routings[0].queue == "glass"
            assert result.routings[0].priority == 4
            assert result.routings[1].queue == "fast_track"
            assert result.routings[2].queue == "total_loss"
            assert result.routings[2].priority == 1


def test_route_claims_to_queues_breakdown():
    """Test queue and priority statistics from routing results."""
    batch_routing = BatchQueueRouting(routings=[
        QueueRouting(claim_id="C001", queue="glass", priority=4, reasoning="Glass only"),
        QueueRouting(claim_id="C002", queue="fast_track", priority=3, reasoning="Minor damage"),
        QueueRouting(claim_id="C003", queue="material_damage", priority=2, reasoning="Moderate damage"),
        QueueRouting(claim_id="C004", queue="material_damage", priority=3, reasoning="Moderate damage"),
        QueueRouting(claim_id="C005", queue="total_loss", priority=1, reasoning="Major damage")
    ])
    
    queue_breakdown = batch_routing.get_queue_breakdown()
    assert queue_breakdown["glass"] == 1
    assert queue_breakdown["fast_track"] == 1
    assert queue_breakdown["material_damage"] == 2
    assert queue_breakdown["total_loss"] == 1
    
    priority_breakdown = batch_routing.get_priority_breakdown()
    assert priority_breakdown[1] == 1
    assert priority_breakdown[2] == 1
    assert priority_breakdown[3] == 2
    assert priority_breakdown[4] == 1


def test_route_claims_to_queues_missing_api_key():
    """Test that ValueError is raised when API key is missing for Stage III."""
    batch_info = BatchFNOLInfo(claims=[
        FNOLInfo(claim_id="C001", damage=DamageInfo(description="Test damage"))
    ])
    batch_severity = BatchSeverityAssessment(assessments=[
        SeverityAssessment(claim_id="C001", severity="Minor", estimated_cost=100.0)
    ])
    
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable is not set"):
            route_claims_to_queues(batch_info, batch_severity)


def test_extract_fnol_with_feedback_loop():
    """Test that feedback loop iterates until achieving target score."""
    # Mock initial extraction response
    initial_response = Mock()
    initial_response.text = json.dumps({
        "claims": [
            {
                "claim_id": "C001",
                "incident_date": None,
                "incident_location": "highway",
                "policyholder_name": "John Smith",
                "contact_phone": None,
                "contact_email": None,
                "vehicle": {"make": "Toyota", "model": "Camry", "year": 2018},
                "damage": {"description": "Windshield chip", "location": "windshield"},
                "incident_description": "Rock hit windshield"
            }
        ]
    })
    
    # Mock first feedback response (score 7/10)
    feedback_response_1 = Mock()
    feedback_response_1.text = json.dumps({
        "feedback": "Missing incident date",
        "issues": [{"claim_id": "C001", "field": "incident_date", "issue": "Not extracted", "suggestion": "Add date"}],
        "json_valid": True,
        "overall_quality_score": 7
    })
    
    # Mock first refined extraction response
    refined_response_1 = Mock()
    refined_response_1.text = json.dumps({
        "claims": [
            {
                "claim_id": "C001",
                "incident_date": "2024-01-15",
                "incident_location": "highway",
                "policyholder_name": "John Smith",
                "contact_phone": None,
                "contact_email": None,
                "vehicle": {"make": "Toyota", "model": "Camry", "year": 2018},
                "damage": {"description": "Windshield chip", "location": "windshield"},
                "incident_description": "Rock hit windshield"
            }
        ]
    })
    
    # Mock second feedback response (score 10/10 - target achieved)
    feedback_response_2 = Mock()
    feedback_response_2.text = json.dumps({
        "feedback": "Extraction is now complete and accurate",
        "issues": [],
        "json_valid": True,
        "overall_quality_score": 10
    })
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        with patch("google.genai.Client") as mock_client:
            mock_instance = Mock()
            # Set up side_effect to return different responses for each call
            mock_instance.models.generate_content.side_effect = [
                initial_response,       # First call: initial extraction
                feedback_response_1,    # Second call: first feedback (score 7)
                refined_response_1,     # Third call: first refinement
                feedback_response_2,    # Fourth call: second feedback (score 10)
            ]
            mock_client.return_value = mock_instance
            
            result = extract_fnol_information_batch("Test FNOL text", enable_feedback_loop=True)
            
            # Verify 4 LLM calls were made (extraction + feedback + refine + feedback with 10/10)
            assert mock_instance.models.generate_content.call_count == 4
            
            # Verify the refined result has incident_date
            assert len(result.claims) == 1
            assert result.claims[0].incident_date == "2024-01-15"


def test_extract_fnol_without_feedback_loop():
    """Test that without feedback loop, only 1 LLM call is made."""
    mock_response = Mock()
    mock_response.text = json.dumps({
        "claims": [
            {
                "claim_id": "C001",
                "policyholder_name": "John Smith",
                "damage": {"description": "Test damage"}
            }
        ]
    })
    
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        with patch("google.genai.Client") as mock_client:
            mock_instance = Mock()
            mock_instance.models.generate_content.return_value = mock_response
            mock_client.return_value = mock_instance
            
            result = extract_fnol_information_batch("Test FNOL text", enable_feedback_loop=False)
            
            # Verify only 1 LLM call was made (no feedback loop)
            assert mock_instance.models.generate_content.call_count == 1
            assert len(result.claims) == 1
