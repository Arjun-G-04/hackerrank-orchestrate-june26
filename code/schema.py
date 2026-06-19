from typing import Literal, Set
from pydantic import BaseModel, Field, field_validator

# Define enums exactly as per problem_statement.md
CLAIM_STATUS_VALUES = Literal["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE_VALUES = Literal[
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]

SEVERITY_VALUES = Literal["none", "low", "medium", "high", "unknown"]

RISK_FLAG_VALUES: Set[str] = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}

CAR_PARTS: Set[str] = {
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
}

LAPTOP_PARTS: Set[str] = {
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
}

PACKAGE_PARTS: Set[str] = {
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
}

ALL_PARTS: Set[str] = CAR_PARTS | LAPTOP_PARTS | PACKAGE_PARTS


class ClaimReviewResult(BaseModel):
    evidence_standard_met: bool = Field(
        description="true if the image set is sufficient to evaluate the claim; otherwise false"
    )
    evidence_standard_met_reason: str = Field(
        description="Short reason for the evidence decision"
    )
    risk_flags: str = Field(description="Semicolon-separated risk flags, or 'none'")
    issue_type: ISSUE_TYPE_VALUES = Field(description="Visible issue type")
    object_part: str = Field(
        description="Relevant object part. Must be valid for the claim_object type."
    )
    claim_status: CLAIM_STATUS_VALUES = Field(
        description="supported, contradicted, or not_enough_information"
    )
    claim_status_justification: str = Field(
        description="Concise explanation grounded in the image evidence"
    )
    supporting_image_ids: str = Field(
        description="Semicolon-separated image IDs (without extension) supporting the decision, or 'none'"
    )
    valid_image: bool = Field(
        description="true if the image set is usable for automated review; otherwise false"
    )
    severity: SEVERITY_VALUES = Field(description="none, low, medium, high, or unknown")

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, v: str) -> str:
        flags = [f.strip() for f in v.split(";")]
        for flag in flags:
            if flag not in RISK_FLAG_VALUES:
                raise ValueError(f"Invalid risk flag: {flag}")
        return v

    @field_validator("object_part")
    @classmethod
    def validate_object_part(cls, v: str) -> str:
        if v not in ALL_PARTS:
            raise ValueError(f"Invalid object part: {v}")
        return v


def validate_result_against_object(
    result: ClaimReviewResult, claim_object: str
) -> None:
    """
    Validates that the object_part and issue_type matches the claim_object.
    Mutates result to fallback or raise error if invalid.
    """
    part = result.object_part
    if claim_object == "car":
        valid_parts = CAR_PARTS
    elif claim_object == "laptop":
        valid_parts = LAPTOP_PARTS
    elif claim_object == "package":
        valid_parts = PACKAGE_PARTS
    else:
        valid_parts = set()

    if part not in valid_parts:
        # If the LLM generates a part that does not belong to the object family, fallback to 'unknown'
        result.object_part = "unknown"
