"""Main application module."""

import os
import sys
import json
from google import genai
from src.models import (
    FNOLInfo, DamageInfo, VehicleInfo, BatchFNOLInfo,
    SeverityAssessment, BatchSeverityAssessment,
    QueueRouting, BatchQueueRouting
)


def extract_fnol_information_batch(raw_text: str) -> BatchFNOLInfo:
    """Extract structured FNOL information from raw text containing multiple claims.
    
    Stage I: Information Extraction (Batch Processing)
    This function uses an LLM to extract specific, structured information 
    from raw text that may contain multiple FNOL claims.
    
    Args:
        raw_text: Raw text containing one or more FNOL claims
        
    Returns:
        BatchFNOLInfo object with list of extracted claims
        
    Raises:
        ValueError: If GOOGLE_API_KEY environment variable is not set
        Exception: If the API request fails
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Get your API key from https://aistudio.google.com/app/apikey"
        )
    
    # Initialize the client with the API key from environment
    client = genai.Client(api_key=api_key)
    
    # Craft the extraction prompt for batch processing
    prompt = f"""You are an insurance claims processing assistant. Extract structured information from the following text that contains multiple First Notice of Loss (FNOL) claims.

For EACH claim found in the text, extract the following information:

1. claim_id: Unique claim identifier (if mentioned)
2. incident_date: Date and time of incident
3. incident_location: Location where incident occurred
4. policyholder_name: Name of the policyholder/customer
5. contact_phone: Contact phone number
6. contact_email: Contact email address
7. vehicle: Object containing:
   - make: Vehicle manufacturer
   - model: Vehicle model
   - year: Vehicle year
   - vin: Vehicle Identification Number
   - license_plate: License plate number
   - color: Vehicle color
8. damage: Object containing:
   - description: Detailed description of the damage
   - location: Location on vehicle where damage occurred
   - severity: Severity (minor, moderate, severe)
   - estimated_repair_cost: Estimated cost (numeric value only)
9. incident_description: Description of how the incident occurred
10. other_parties_involved: Boolean - were other parties involved?
11. police_report_filed: Boolean - was a police report filed?

Return a JSON object with a "claims" array containing one object per claim found. Use null for any information not found in the text.

FNOL Text:
{raw_text}

Return ONLY a valid JSON object in this format:
{{
  "claims": [
    {{ ... claim 1 data ... }},
    {{ ... claim 2 data ... }},
    ...
  ]
}}

Do not include any explanation or markdown formatting, just the raw JSON."""

    # Generate content using Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Parse the response and create BatchFNOLInfo object
    try:
        # Clean the response text (remove markdown code blocks if present)
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Remove any stray text that might break JSON parsing
        # Sometimes LLM adds comments or extra text
        lines = response_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like comments or random text
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not (stripped.isalpha() and len(stripped) < 20):
                cleaned_lines.append(line)
        response_text = '\n'.join(cleaned_lines)
        
        # Parse JSON
        extracted_data = json.loads(response_text)
        
        # Create BatchFNOLInfo object from extracted data
        batch_info = BatchFNOLInfo(**extracted_data)
        return batch_info
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response.text}")
    except Exception as e:
        raise ValueError(f"Failed to create BatchFNOLInfo object: {e}")


def assess_claim_severity(batch_fnol_info: BatchFNOLInfo) -> BatchSeverityAssessment:
    """Assess severity and estimate costs for extracted FNOL claims.
    
    Stage II: Severity Assessment and Cost Estimation
    This function takes the output from Stage I and uses an LLM to classify
    damage severity and provide cost estimates for each claim.
    
    Args:
        batch_fnol_info: BatchFNOLInfo object with extracted claims from Stage I
        
    Returns:
        BatchSeverityAssessment with severity classifications and cost estimates
        
    Raises:
        ValueError: If GOOGLE_API_KEY environment variable is not set
        Exception: If the API request fails
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Get your API key from https://aistudio.google.com/app/apikey"
        )
    
    # Initialize the client
    client = genai.Client(api_key=api_key)
    
    # Prepare claim summaries for assessment
    claims_summary = []
    for idx, claim in enumerate(batch_fnol_info.claims):
        loss_desc = claim.damage.description if claim.damage else "No damage description provided"
        damage_area = claim.damage.location if claim.damage and claim.damage.location else "Unknown"
        
        claim_summary = {
            "claim_id": claim.claim_id or f"CLAIM-{idx+1}",
            "loss_desc": loss_desc,
            "damage_area": damage_area,
            "incident_description": claim.incident_description or "No incident description"
        }
        claims_summary.append(claim_summary)
    
    # Craft the Stage II prompt
    prompt = f"""You are an insurance claims severity assessment specialist. Based on the claim information provided, classify the damage severity and estimate repair costs.

For each claim, analyze the loss_desc (damage description) and damage_area to classify the damage as 'Minor', 'Moderate', or 'Major'. Also provide an estimated repair cost (estimated_cost) as a float.

Use the following guidelines:
- Minor: $100-$1,000 - Small cosmetic damage, minor scratches, small chips
- Moderate: $1,000-$5,000 - Significant body damage, broken parts, multiple areas affected
- Major: $5,000-$50,000 - Severe structural damage, airbag deployment, vehicle not drivable, extensive damage

Claims to assess:
{json.dumps(claims_summary, indent=2)}

Return a JSON object with an "assessments" array containing one assessment per claim:
{{
  "assessments": [
    {{
      "claim_id": "claim identifier",
      "severity": "Minor|Moderate|Major",
      "estimated_cost": float_value,
      "reasoning": "Brief explanation of severity classification"
    }},
    ...
  ]
}}

Do not include any explanation or markdown formatting, just the raw JSON."""

    # Generate content using Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Parse the response
    try:
        # Clean the response text
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Try to extract just the JSON object by finding the outermost braces
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            response_text = response_text[first_brace:last_brace+1]
        
        # Parse JSON
        assessment_data = json.loads(response_text)
        
        # Create BatchSeverityAssessment object
        batch_assessment = BatchSeverityAssessment(**assessment_data)
        return batch_assessment
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Stage II LLM response as JSON: {e}\nResponse: {response.text}")
    except Exception as e:
        raise ValueError(f"Failed to create BatchSeverityAssessment object: {e}")


def route_claims_to_queues(batch_fnol_info: BatchFNOLInfo, batch_severity: BatchSeverityAssessment) -> BatchQueueRouting:
    """Route claims to appropriate queues with priority assignment.
    
    Stage III: Queue Routing and Priority Assignment
    This function takes outputs from Stage I and Stage II and uses an LLM to
    assign each claim to the appropriate processing queue with priority.
    
    Args:
        batch_fnol_info: BatchFNOLInfo object with extracted claims from Stage I
        batch_severity: BatchSeverityAssessment with severity assessments from Stage II
        
    Returns:
        BatchQueueRouting with queue assignments and priorities
        
    Raises:
        ValueError: If GOOGLE_API_KEY environment variable is not set
        Exception: If the API request fails
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Get your API key from https://aistudio.google.com/app/apikey"
        )
    
    # Initialize the client
    client = genai.Client(api_key=api_key)
    
    # Prepare claim data for routing
    claims_for_routing = []
    for idx, (claim, assessment) in enumerate(zip(batch_fnol_info.claims, batch_severity.assessments)):
        damage_location = claim.damage.location if claim.damage and claim.damage.location else "Unknown"
        damage_desc = claim.damage.description if claim.damage else "No description"
        
        claim_data = {
            "claim_id": claim.claim_id or f"CLAIM-{idx+1}",
            "severity": assessment.severity,
            "estimated_cost": assessment.estimated_cost,
            "damage_location": damage_location,
            "damage_description": damage_desc,
            "incident_description": claim.incident_description or "No incident description"
        }
        claims_for_routing.append(claim_data)
    
    # Craft the Stage III prompt
    prompt = f"""You are an AI claim routing system. Based on the claim information and severity assessment, assign the claim to one of the following queues: 'glass', 'fast_track', 'material_damage', or 'total_loss'.

Use these rules:
- Minor damage involving ONLY glass goes to 'glass'
- Other Minor damage goes to 'fast_track'
- Moderate damage goes to 'material_damage'
- Major damage goes to 'total_loss'

Also assign a priority from 1 (highest) to 5 (lowest) based on the overall situation described.

Priority guidelines:
- Priority 1: Critical/urgent (major damage, vehicle not drivable, safety concerns)
- Priority 2: High priority (significant damage, multiple areas affected)
- Priority 3: Medium priority (moderate damage, standard processing)
- Priority 4: Low priority (minor damage, cosmetic issues)
- Priority 5: Lowest priority (very minor, no safety concerns)

Claims to route:
{json.dumps(claims_for_routing, indent=2)}

Return a JSON object with a "routings" array containing one routing per claim:
{{
  "routings": [
    {{
      "claim_id": "claim identifier",
      "queue": "glass|fast_track|material_damage|total_loss",
      "priority": 1-5,
      "reasoning": "Brief explanation of queue and priority assignment"
    }},
    ...
  ]
}}

Do not include any explanation or markdown formatting, just the raw JSON."""

    # Generate content using Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Parse the response
    try:
        # Clean the response text
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Try to extract just the JSON object by finding the outermost braces
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            response_text = response_text[first_brace:last_brace+1]
        
        # Parse JSON
        routing_data = json.loads(response_text)
        
        # Create BatchQueueRouting object
        batch_routing = BatchQueueRouting(**routing_data)
        return batch_routing
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Stage III LLM response as JSON: {e}\nResponse: {response.text}")
    except Exception as e:
        raise ValueError(f"Failed to create BatchQueueRouting object: {e}")


def main():
    """Entry point for the application - Stage I: Information Extraction."""
    import argparse
    from pathlib import Path
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract structured information from FNOL text using LLM"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="inputs/sample_fnol.txt",
        help="Path to input file containing raw FNOL text (default: inputs/sample_fnol.txt)"
    )
    args = parser.parse_args()
    
    try:
        # Read FNOL text from input file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
            
        with open(input_path, 'r', encoding='utf-8') as f:
            fnol_text = f.read()
        
        if not fnol_text.strip():
            print(f"Error: Input file is empty: {input_path}", file=sys.stderr)
            sys.exit(1)
        
        print("=" * 80)
        print("STAGE I: INFORMATION EXTRACTION FROM FNOL")
        print("=" * 80)
        print(f"\nReading from: {input_path}")
        print("\nRaw FNOL Text:")
        print("-" * 80)
        print(fnol_text.strip())
        print("-" * 80)
        
        print("\n[STAGE I] Extracting information using Gemini 2.5 Flash...")
        batch_info = extract_fnol_information_batch(fnol_text)
        
        print("\n" + "=" * 80)
        print(f"STAGE I: EXTRACTED INFORMATION - {len(batch_info.claims)} CLAIMS FOUND")
        print("=" * 80)
        
        for idx, claim in enumerate(batch_info.claims, 1):
            print(f"\n--- CLAIM {idx} ---")
            print(f"Claim ID: {claim.claim_id or 'N/A'}")
            print(f"Customer: {claim.policyholder_name or 'N/A'}")
            if claim.vehicle and (claim.vehicle.year or claim.vehicle.make or claim.vehicle.model):
                vehicle_str = f"{claim.vehicle.year or ''} {claim.vehicle.make or ''} {claim.vehicle.model or ''}".strip()
                print(f"Vehicle: {vehicle_str}")
            if claim.damage and claim.damage.description:
                print(f"Damage: {claim.damage.description[:100]}{'...' if len(claim.damage.description) > 100 else ''}")
        
        print("\n" + "=" * 80)
        print("STAGE I: VALIDATION SUMMARY")
        print("=" * 80)
        validation_summary = batch_info.validate_all()
        print(f"Total Claims: {validation_summary['total_claims']}")
        print(f"Valid Claims: {validation_summary['valid_claims']}")
        print(f"Invalid Claims: {validation_summary['invalid_claims']}")
        if validation_summary['invalid_claims'] > 0:
            print("\nInvalid Claim Details:")
            for detail in validation_summary['invalid_details']:
                print(f"  Claim {detail['claim_id']}: Missing fields: {detail['missing_fields']}")
        
        # Stage II: Severity Assessment
        print("\n" + "=" * 80)
        print("STAGE II: SEVERITY ASSESSMENT & COST ESTIMATION")
        print("=" * 80)
        print("\n[STAGE II] Assessing claim severity and estimating costs...")
        
        severity_assessment = assess_claim_severity(batch_info)
        
        print("\n" + "=" * 80)
        print("STAGE II: ASSESSMENT RESULTS")
        print("=" * 80)
        
        for idx, assessment in enumerate(severity_assessment.assessments, 1):
            print(f"\n--- CLAIM {idx} ASSESSMENT ---")
            print(f"Claim ID: {assessment.claim_id}")
            print(f"Severity: {assessment.severity}")
            print(f"Estimated Cost: ${assessment.estimated_cost:,.2f}")
            if assessment.reasoning:
                print(f"Reasoning: {assessment.reasoning}")
        
        print("\n" + "=" * 80)
        print("STAGE II: SUMMARY STATISTICS")
        print("=" * 80)
        severity_breakdown = severity_assessment.get_severity_breakdown()
        print(f"Minor Claims: {severity_breakdown['Minor']}")
        print(f"Moderate Claims: {severity_breakdown['Moderate']}")
        print(f"Major Claims: {severity_breakdown['Major']}")
        print("=" * 80)
        
        # Stage III: Queue Routing
        print("\n" + "=" * 80)
        print("STAGE III: QUEUE ROUTING & PRIORITY ASSIGNMENT")
        print("=" * 80)
        print("\n[STAGE III] Routing claims to appropriate queues...")
        
        queue_routing = route_claims_to_queues(batch_info, severity_assessment)
        
        print("\n" + "=" * 80)
        print("STAGE III: ROUTING RESULTS")
        print("=" * 80)
        
        for idx, routing in enumerate(queue_routing.routings, 1):
            print(f"\n--- CLAIM {idx} ROUTING ---")
            print(f"Claim ID: {routing.claim_id}")
            print(f"Queue: {routing.queue}")
            print(f"Priority: {routing.priority}")
            if routing.reasoning:
                print(f"Reasoning: {routing.reasoning}")
        
        print("\n" + "=" * 80)
        print("STAGE III: ROUTING STATISTICS")
        print("=" * 80)
        queue_breakdown = queue_routing.get_queue_breakdown()
        print("Queue Assignments:")
        for queue, count in queue_breakdown.items():
            print(f"  {queue}: {count} claim(s)")
        
        priority_breakdown = queue_routing.get_priority_breakdown()
        print("\nPriority Distribution:")
        for priority in sorted(priority_breakdown.keys()):
            count = priority_breakdown[priority]
            print(f"  Priority {priority}: {count} claim(s)")
        print("=" * 80)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing FNOL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
