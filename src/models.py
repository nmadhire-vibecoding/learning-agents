"""Data models for FNOL (First Notice of Loss) information extraction."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class VehicleInfo(BaseModel):
    """Information about the vehicle involved in the claim."""
    
    make: Optional[str] = Field(None, description="Vehicle manufacturer (e.g., Toyota, Honda)")
    model: Optional[str] = Field(None, description="Vehicle model (e.g., Camry, Civic)")
    year: Optional[int] = Field(None, description="Vehicle year", ge=1900, le=2100)
    vin: Optional[str] = Field(None, description="Vehicle Identification Number")
    license_plate: Optional[str] = Field(None, description="License plate number")
    color: Optional[str] = Field(None, description="Vehicle color")


class DamageInfo(BaseModel):
    """Information about the damage reported."""
    
    description: str = Field(..., description="Detailed description of the damage")
    location: Optional[str] = Field(None, description="Location on vehicle where damage occurred (e.g., front bumper, driver side door)")
    severity: Optional[str] = Field(None, description="Severity of damage (e.g., minor, moderate, severe)")
    estimated_repair_cost: Optional[float] = Field(None, description="Estimated cost to repair", ge=0)


class FNOLInfo(BaseModel):
    """Structured information extracted from FNOL text."""
    
    claim_id: Optional[str] = Field(None, description="Unique claim identifier")
    incident_date: Optional[str] = Field(None, description="Date and time of incident")
    incident_location: Optional[str] = Field(None, description="Location where incident occurred")
    policyholder_name: Optional[str] = Field(None, description="Name of the policyholder")
    contact_phone: Optional[str] = Field(None, description="Contact phone number")
    contact_email: Optional[str] = Field(None, description="Contact email address")
    vehicle: VehicleInfo = Field(default_factory=VehicleInfo, description="Vehicle information")
    damage: Optional[DamageInfo] = Field(None, description="Damage information")
    incident_description: Optional[str] = Field(None, description="Description of how the incident occurred")
    other_parties_involved: Optional[bool] = Field(None, description="Whether other parties were involved")
    police_report_filed: Optional[bool] = Field(None, description="Whether a police report was filed")
    
    def validate_required_fields(self) -> tuple[bool, list[str]]:
        """Validate that critical fields are present.
        
        Returns:
            Tuple of (is_valid, list of missing fields)
        """
        missing_fields = []
        
        # Check critical fields
        if not self.incident_date:
            missing_fields.append("incident_date")
        if not self.incident_location:
            missing_fields.append("incident_location")
        if not self.policyholder_name:
            missing_fields.append("policyholder_name")
        if not self.damage or not self.damage.description:
            missing_fields.append("damage.description")
            
        return len(missing_fields) == 0, missing_fields


class BatchFNOLInfo(BaseModel):
    """Container for multiple FNOL extractions."""
    
    claims: List[FNOLInfo] = Field(default_factory=list, description="List of extracted FNOL information")
    
    def validate_all(self) -> dict:
        """Validate all claims and return summary.
        
        Returns:
            Dictionary with validation summary including counts and details
        """
        total = len(self.claims)
        valid_count = 0
        invalid_claims = []
        
        for idx, claim in enumerate(self.claims):
            is_valid, missing_fields = claim.validate_required_fields()
            if is_valid:
                valid_count += 1
            else:
                invalid_claims.append({
                    "index": idx,
                    "claim_id": claim.claim_id,
                    "missing_fields": missing_fields
                })
        
        return {
            "total_claims": total,
            "valid_claims": valid_count,
            "invalid_claims": len(invalid_claims),
            "invalid_details": invalid_claims
        }


class SeverityAssessment(BaseModel):
    """Stage II: Severity assessment and cost estimation for a claim."""
    
    claim_id: Optional[str] = Field(None, description="Claim identifier for reference")
    severity: str = Field(..., description="Damage severity: Minor, Moderate, or Major")
    estimated_cost: float = Field(..., description="Estimated repair cost in dollars", ge=0)
    reasoning: Optional[str] = Field(None, description="Explanation for the severity classification")


class BatchSeverityAssessment(BaseModel):
    """Container for multiple severity assessments."""
    
    assessments: List[SeverityAssessment] = Field(default_factory=list, description="List of severity assessments")
    
    def get_severity_breakdown(self) -> dict:
        """Get count of claims by severity level.
        
        Returns:
            Dictionary with counts for each severity level
        """
        breakdown = {"Minor": 0, "Moderate": 0, "Major": 0}
        for assessment in self.assessments:
            severity = assessment.severity.capitalize()
            if severity in breakdown:
                breakdown[severity] += 1
        return breakdown


class QueueRouting(BaseModel):
    """Stage III: Queue routing and priority assignment for a claim."""
    
    claim_id: Optional[str] = Field(None, description="Claim identifier for reference")
    queue: str = Field(..., description="Assigned queue: glass, fast_track, material_damage, or total_loss")
    priority: int = Field(..., description="Priority level from 1 (highest) to 5 (lowest)", ge=1, le=5)
    reasoning: Optional[str] = Field(None, description="Explanation for the queue and priority assignment")


class BatchQueueRouting(BaseModel):
    """Container for multiple queue routing assignments."""
    
    routings: List[QueueRouting] = Field(default_factory=list, description="List of queue routing assignments")
    
    def get_queue_breakdown(self) -> dict:
        """Get count of claims by queue.
        
        Returns:
            Dictionary with counts for each queue
        """
        breakdown = {"glass": 0, "fast_track": 0, "material_damage": 0, "total_loss": 0}
        for routing in self.routings:
            queue = routing.queue.lower()
            if queue in breakdown:
                breakdown[queue] += 1
        return breakdown
    
    def get_priority_breakdown(self) -> dict:
        """Get count of claims by priority level.
        
        Returns:
            Dictionary with counts for each priority level
        """
        breakdown = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for routing in self.routings:
            if routing.priority in breakdown:
                breakdown[routing.priority] += 1
        return breakdown
