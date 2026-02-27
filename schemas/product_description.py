"""Structured schema for generated product pages."""

from __future__ import annotations

from pydantic import AnyUrl, BaseModel, Field


class ProductDescription(BaseModel):
    """Output schema for frontend product-page rendering."""

    schema_version: str = Field(default="1.0")
    title: str = Field(min_length=3, max_length=120)
    short_description: str = Field(min_length=10, max_length=240)
    long_description: str = Field(min_length=40, max_length=3000)
    key_features: list[str] = Field(default_factory=list, max_length=12)
    condition: str = Field(default="Used - Good", max_length=80)
    estimated_category: str = Field(default="General")
    search_facts: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)
    source_image_path: str | None = None
    polished_image_url: AnyUrl | None = None
    polished_image_path: str | None = None


def build_product_description(
    *,
    title: str,
    short_description: str,
    long_description: str,
    key_features: list[str],
    condition: str,
    estimated_category: str,
    search_facts: list[str],
    confidence: float,
    source_image_path: str | None = None,
    polished_image_url: str | None = None,
    polished_image_path: str | None = None,
) -> ProductDescription:
    """Factory helper to keep description creation explicit and validated."""
    return ProductDescription(
        title=title,
        short_description=short_description,
        long_description=long_description,
        key_features=key_features,
        condition=condition,
        estimated_category=estimated_category,
        search_facts=search_facts,
        confidence=confidence,
        source_image_path=source_image_path,
        polished_image_url=polished_image_url,
        polished_image_path=polished_image_path,
    )

