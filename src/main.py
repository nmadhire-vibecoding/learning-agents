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


def extract_fnol_information_batch(raw_text: str, enable_feedback_loop: bool = False) -> BatchFNOLInfo:
    """Extract structured FNOL information from raw text containing multiple claims.
    
    Stage I: Information Extraction (Batch Processing)
    This function uses an LLM to extract specific, structured information 
    from raw text that may contain multiple FNOL claims.
    
    If feedback loop is enabled, the LLM will review its own extraction and
    provide feedback, then refine the extraction based on that feedback.
    
    Args:
        raw_text: Raw text containing one or more FNOL claims
        enable_feedback_loop: If True, enables LLM self-review and refinement (default: False)
        
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
3. incident_location: Location where incident occurred (MUST be a physical/geographic location, not event descriptions like "in a hailstorm")
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
   - description: Detailed description of ONLY the physical damage observed (do NOT include cause or context)
   - location: Specific location on vehicle where damage occurred (e.g., "windshield", "rear bumper", "driver side doors")
   - severity: Severity (minor, moderate, severe) - infer based on damage extent
   - estimated_repair_cost: Estimated cost (numeric value only)
9. incident_description: Full description of how the incident occurred (can copy from original text)
10. other_parties_involved: Boolean - were other parties involved? (infer from context if not explicit)
11. police_report_filed: Boolean - was a police report filed? (infer from context if not explicit)

CRITICAL EXTRACTION RULES:
- Extract ALL available information from the text
- For damage.description: Focus ONLY on physical damage, not cause (e.g., "dented rear bumper and broken taillight" NOT "someone hit my car and dented...")
- For incident_location: Only use geographic/physical locations (e.g., "highway", "grocery store", "intersection"), set to null if only event description available
- For damage.location: Be specific about vehicle parts affected
- Infer boolean fields (other_parties_involved, police_report_filed) from context when possible
- Use null ONLY for information that is truly not present or inferable
- Ensure ALL JSON is perfectly valid - no extra fields, no typos, no random text

QUALITY CHECKLIST (aim for 10/10):
✓ All explicit information extracted
✓ damage.description focuses on physical damage only
✓ incident_location is a physical place or null
✓ damage.location is specific
✓ Boolean fields inferred when possible
✓ JSON is perfectly valid and parseable
✓ No missing extractable information

FNOL Text:
{raw_text}

Return ONLY a valid JSON object in this exact format:
{{
  "claims": [
    {{ ... claim 1 data ... }},
    {{ ... claim 2 data ... }},
    ...
  ]
}}

CRITICAL: Return ONLY the JSON object, no explanations, no markdown formatting, no extra text. The JSON must be perfectly parseable."""

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
        
        # Try to extract just the JSON object by finding the outermost braces
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            response_text = response_text[first_brace:last_brace+1]
        
        # Parse JSON
        extracted_data = json.loads(response_text)
        
        # Create BatchFNOLInfo object from extracted data
        batch_info = BatchFNOLInfo(**extracted_data)
        
        # Feedback loop: If enabled, ask LLM to review and improve the extraction
        if enable_feedback_loop:
            print("\n[FEEDBACK LOOP] Initiating LLM self-review...")
            batch_info = _apply_feedback_loop(client, raw_text, batch_info)
        
        return batch_info
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response.text}")
    except Exception as e:
        raise ValueError(f"Failed to create BatchFNOLInfo object: {e}")


def _apply_feedback_loop(client: genai.Client, original_text: str, initial_extraction: BatchFNOLInfo, max_iterations: int = 5) -> BatchFNOLInfo:
    """Apply feedback loop to review and improve the initial extraction.
    
    This function sends the initial extraction back to the LLM for review,
    receives feedback on what's missing or incorrect, and then refines the extraction.
    It will iterate until achieving a quality score of 10/10 or reaching max iterations.
    
    Args:
        client: Initialized Gemini client
        original_text: The original raw FNOL text
        initial_extraction: The initial BatchFNOLInfo extraction
        max_iterations: Maximum number of refinement iterations (default: 5)
        
    Returns:
        Refined BatchFNOLInfo object after feedback
    """
    current_extraction = initial_extraction
    iteration = 0
    target_score = 10
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n[FEEDBACK LOOP - Iteration {iteration}/{max_iterations}]")
        
        # Step 1: Ask LLM to review the extraction and provide feedback
        feedback_prompt = f"""You are a quality assurance reviewer for insurance claims extraction.

Original FNOL Text:
{original_text}

Extracted Data (JSON):
{current_extraction.model_dump_json(indent=2)}

Review the extracted data against the original text and provide feedback on:
1. Missing information that should have been extracted
2. Incorrect or inaccurate extractions
3. Fields that could be more complete or detailed
4. Any inconsistencies between the original text and extracted data
5. JSON structure and parseability issues

Return a JSON object with:
{{
  "feedback": "Overall assessment of the extraction quality",
  "issues": [
    {{
      "claim_id": "ID of claim with issue",
      "field": "name of field with issue",
      "issue": "description of the problem",
      "suggestion": "suggested correction or addition"
    }}
  ],
  "json_valid": true/false,
  "overall_quality_score": 1-10
}}

SCORING CRITERIA:
- 10/10: Perfect extraction - all available info extracted, JSON valid, no issues, follows all rules
- 9/10: Excellent but minor improvement possible (e.g., slight wording preference)
- 8/10: Very good but 1-2 small issues
- 7/10 or below: Missing data or rule violations

Be pragmatic: If the text doesn't provide information (like incident_date), that's acceptable.
Only deduct points for information that WAS available but not extracted, or rule violations.

Do not include markdown formatting, just raw JSON."""

        feedback_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=feedback_prompt
        )
        
        # Parse feedback
        feedback_text = feedback_response.text.strip()
        if feedback_text.startswith("```json"):
            feedback_text = feedback_text[7:]
        if feedback_text.startswith("```"):
            feedback_text = feedback_text[3:]
        if feedback_text.endswith("```"):
            feedback_text = feedback_text[:-3]
        feedback_text = feedback_text.strip()
        
        # Try to extract JSON by finding outermost braces
        first_brace = feedback_text.find('{')
        last_brace = feedback_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            feedback_text = feedback_text[first_brace:last_brace+1]
        
        try:
            feedback_data = json.loads(feedback_text)
            quality_score = feedback_data.get('overall_quality_score', 0)
            json_valid = feedback_data.get('json_valid', True)
            
            print(f"[FEEDBACK] Quality Score: {quality_score}/10")
            print(f"[FEEDBACK] JSON Valid: {json_valid}")
            print(f"[FEEDBACK] {feedback_data.get('feedback', 'No feedback provided')}")
            
            if feedback_data.get('issues'):
                print(f"[FEEDBACK] Found {len(feedback_data['issues'])} issues to address")
                for idx, issue in enumerate(feedback_data['issues'][:3], 1):  # Show first 3 issues
                    print(f"  {idx}. {issue.get('field', 'unknown')}: {issue.get('issue', 'no details')}")
            
            # Check if we've achieved target score
            if quality_score >= target_score and json_valid:
                print(f"[FEEDBACK] ✓ Target score of {target_score}/10 achieved!")
                return current_extraction
                
        except json.JSONDecodeError:
            print("[FEEDBACK] Warning: Could not parse feedback JSON, proceeding with refinement")
            feedback_data = {"feedback": "Unable to parse feedback", "issues": [], "json_valid": False, "overall_quality_score": 0}
            quality_score = 0
            json_valid = False
        
        # If we've reached max iterations but haven't achieved target, return best effort
        if iteration >= max_iterations:
            print(f"[FEEDBACK] Max iterations ({max_iterations}) reached. Final score: {quality_score}/10")
            return current_extraction
        
        # Step 2: Ask LLM to refine the extraction based on feedback
        print(f"[FEEDBACK] Refining extraction (attempt {iteration + 1})...")
        refinement_prompt = f"""You are an insurance claims processor refining an extraction based on feedback.

Original FNOL Text:
{original_text}

Current Extraction:
{current_extraction.model_dump_json(indent=2)}

Feedback Received:
{json.dumps(feedback_data, indent=2)}

CRITICAL: You must return ONLY valid, parseable JSON. No extra text, no markdown, no comments.

Based on the feedback, create an improved extraction addressing ALL issues identified.
Return ONLY a valid JSON object with this exact structure:
{{
  "claims": [
    {{ ... improved claim 1 data ... }},
    {{ ... improved claim 2 data ... }},
    ...
  ]
}}

Ensure:
1. All JSON is valid and parseable
2. No extra text or markdown formatting
3. All issues from feedback are addressed
4. No random text or fields are added

RETURN ONLY THE JSON OBJECT, NOTHING ELSE."""

        refinement_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=refinement_prompt
        )
        
        # Parse refined extraction
        refined_text = refinement_response.text.strip()
        if refined_text.startswith("```json"):
            refined_text = refined_text[7:]
        if refined_text.startswith("```"):
            refined_text = refined_text[3:]
        if refined_text.endswith("```"):
            refined_text = refined_text[:-3]
        refined_text = refined_text.strip()
        
        # Try to extract JSON by finding outermost braces
        first_brace = refined_text.find('{')
        last_brace = refined_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            refined_text = refined_text[first_brace:last_brace+1]
        
        # Validate JSON parseability
        try:
            refined_data = json.loads(refined_text)
            refined_batch_info = BatchFNOLInfo(**refined_data)
            print(f"[FEEDBACK] ✓ Refinement successful, JSON valid")
            # Update current extraction for next iteration
            current_extraction = refined_batch_info
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"[FEEDBACK] ✗ Refinement failed: {e}")
            print(f"[FEEDBACK] Keeping previous extraction and trying again...")
            # Keep current_extraction unchanged and continue loop
    
    # If we exit the loop without returning, return the last valid extraction
    print(f"[FEEDBACK] Returning best extraction after {max_iterations} iterations")
    return current_extraction


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
    parser.add_argument(
        "--feedback-loop",
        "-f",
        action="store_true",
        help="Enable feedback loop for Stage I extraction (LLM reviews and refines its own output)"
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
        if args.feedback_loop:
            print("[STAGE I] Feedback loop ENABLED - LLM will review and refine extraction")
        batch_info = extract_fnol_information_batch(fnol_text, enable_feedback_loop=args.feedback_loop)
        
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
