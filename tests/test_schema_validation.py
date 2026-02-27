"""Schema validation tests for generated product descriptions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.product_description import build_product_description


def test_product_description_builds_when_valid() -> None:
    result = build_product_description(
        title="Vintage Camera Lens",
        short_description="Clean vintage lens with smooth focus and minimal wear.",
        long_description=(
            "A well-maintained vintage camera lens suitable for collectors and hobbyists. "
            "Glass appears clear and housing shows light cosmetic wear only."
        ),
        key_features=["Clear optics", "Smooth focusing ring", "Classic build quality"],
        condition="Used - Good",
        estimated_category="Photography",
        search_facts=["Likely manual focus lens", "Compatible with vintage camera bodies"],
        confidence=0.82,
        source_image_path="artifacts/source_product_frame.png",
        polished_image_url="https://example.com/polished.png",
    )

    assert result.title == "Vintage Camera Lens"
    assert result.confidence == 0.82
    assert result.schema_version == "1.0"


def test_product_description_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        build_product_description(
            title="Headphones",
            short_description="Wireless over-ear headphones in good condition.",
            long_description=(
                "Comfortable over-ear wireless headphones with clear sound output "
                "and a long battery life."
            ),
            key_features=["Bluetooth", "Foldable design"],
            condition="Used - Good",
            estimated_category="Audio",
            search_facts=["Supports Bluetooth pairing"],
            confidence=1.5,
        )

