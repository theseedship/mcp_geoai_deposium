"""
Adapter: Deposium GeoAI → so4gp

This module provides conversion utilities to transform Deposium GeoAI outputs
(GeoJSON, time-series) into formats compatible with so4gp gradual pattern mining.

Example Usage:
    from examples.so4gp_integration import ndvi_profile_to_dataframe, run_graank

    # Get NDVI profile from Deposium
    profile = await geo.ndvi_profile(polygon, "2023-01-01", "2024-01-01")

    # Convert to pandas DataFrame
    df = ndvi_profile_to_dataframe(profile)

    # Run GRAANK algorithm
    patterns = run_graank(df, min_support=0.5)
"""

from typing import Any

import pandas as pd


def ndvi_profile_to_dataframe(
    profile_result: dict,
    include_geometry: bool = False,
) -> pd.DataFrame:
    """Convert geo.ndvi_profile output to pandas DataFrame for so4gp.

    Args:
        profile_result: Output from geo.ndvi_profile tool
        include_geometry: Whether to include geometry centroid as lat/lon

    Returns:
        DataFrame with columns: [date, ndvi_mean, ndvi_std, cloud_cover, ...]

    Example:
        >>> profile = {
        ...     "geometry": {"type": "Polygon", "coordinates": [...]},
        ...     "profile": [
        ...         {"date": "2024-01-15", "ndvi_mean": 0.72, "cloud_cover": 5},
        ...         {"date": "2024-02-01", "ndvi_mean": 0.68, "cloud_cover": 12},
        ...     ],
        ...     "statistics": {"mean": 0.70, ...}
        ... }
        >>> df = ndvi_profile_to_dataframe(profile)
        >>> df.columns
        Index(['date', 'ndvi_mean', 'cloud_cover', 'time_index'])
    """
    if "profile" not in profile_result:
        raise ValueError("Invalid profile result: missing 'profile' key")

    df = pd.DataFrame(profile_result["profile"])

    # Convert date to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        # Add numeric time column for pattern mining
        df["time_index"] = range(len(df))

    # Optionally add geometry centroid
    if include_geometry and "geometry" in profile_result:
        geom = profile_result["geometry"]
        if geom.get("type") == "Point":
            df["longitude"] = geom["coordinates"][0]
            df["latitude"] = geom["coordinates"][1]
        # For polygons, would need shapely to compute centroid

    return df


def change_detect_to_dataframe(
    change_result: dict,
    flatten_geometry: bool = True,
) -> pd.DataFrame:
    """Convert geo.change_detect output to pandas DataFrame for so4gp.

    Args:
        change_result: Output from geo.change_detect tool
        flatten_geometry: Replace geometry with centroid lat/lon

    Returns:
        DataFrame with columns: [change_type, confidence, area_ha, date, ...]

    Example:
        >>> changes = {
        ...     "type": "FeatureCollection",
        ...     "features": [
        ...         {
        ...             "geometry": {"type": "Polygon", ...},
        ...             "properties": {
        ...                 "change_type": "deforestation",
        ...                 "confidence": 0.89,
        ...                 "area_ha": 2.34
        ...             }
        ...         }
        ...     ]
        ... }
        >>> df = change_detect_to_dataframe(changes)
    """
    if "features" not in change_result:
        raise ValueError("Invalid change result: missing 'features' key")

    rows = []
    for feature in change_result["features"]:
        row = feature.get("properties", {}).copy()

        if flatten_geometry and "geometry" in feature:
            geom = feature["geometry"]
            # Simplified centroid for points/polygons
            if geom.get("type") == "Point":
                row["longitude"] = geom["coordinates"][0]
                row["latitude"] = geom["coordinates"][1]
            elif geom.get("type") == "Polygon":
                # Approximate centroid from first ring
                coords = geom["coordinates"][0]
                row["longitude"] = sum(c[0] for c in coords) / len(coords)
                row["latitude"] = sum(c[1] for c in coords) / len(coords)

        rows.append(row)

    return pd.DataFrame(rows)


def spatial_query_to_dataframe(
    spatial_result: dict,
    exclude_geometry: bool = True,
) -> pd.DataFrame:
    """Convert geo.spatial_query output to pandas DataFrame for so4gp.

    Args:
        spatial_result: Output from geo.spatial_query tool
        exclude_geometry: Whether to exclude geometry column

    Returns:
        DataFrame with feature properties as columns
    """
    if "features" not in spatial_result:
        raise ValueError("Invalid spatial result: missing 'features' key")

    rows = []
    for feature in spatial_result["features"]:
        row = feature.get("properties", {}).copy()
        if not exclude_geometry and "geometry" in feature:
            row["geometry"] = feature["geometry"]
        rows.append(row)

    return pd.DataFrame(rows)


def enrich_with_climate_data(
    df: pd.DataFrame,
    climate_source: str = "era5",
) -> pd.DataFrame:
    """Enrich time-series DataFrame with climate variables.

    NOTE: This is a placeholder. Full implementation would fetch from:
    - ERA5 Climate Reanalysis API
    - Local climate station data
    - Other climate data sources

    Args:
        df: DataFrame with 'date' and location columns
        climate_source: Source for climate data

    Returns:
        DataFrame enriched with precipitation, temperature, humidity
    """
    # Placeholder: In real implementation, would call climate API
    # For now, just return the original DataFrame
    print(f"[PLACEHOLDER] Would fetch climate data from {climate_source}")
    print("Columns needed: precipitation_mm, temperature_c, humidity_pct")
    return df


def run_graank(
    df: pd.DataFrame,
    min_support: float = 0.5,
    target_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run so4gp GRAANK on a DataFrame and return its gradual patterns.

    Delegates to the real so4gp 0.8.8 API via ``src.patterns.so4gp_adapter``.
    A gradual pattern is a CO-MONOTONIC itemset (attributes rising/falling
    together across rows), not an antecedent->consequent rule.

    If so4gp is not installed (it needs Python >=3.12), falls back to
    clearly-labelled MOCK patterns for the offline demo. Any *real* mining
    error propagates — it is NOT silently replaced by mock output.

    Args:
        df: Input DataFrame; non-numeric columns are dropped (so4gp needs
            numeric columns only).
        min_support: Minimum support threshold (0-1).
        target_columns: Columns to analyze (default: all numeric).

    Returns:
        List of gradual patterns, e.g.:
        {'pattern': '{precipitation↑, ndvi↑}', 'support': 0.82,
         'attributes': ['precipitation', 'ndvi'], 'gradients': ['+', '+']}
    """
    from src.patterns.so4gp_adapter import So4gpNotInstalled, mine_gradual_patterns

    if target_columns is not None:
        df = df[target_columns]
    df = df.select_dtypes(include=["number"])

    try:
        result = mine_gradual_patterns(df, min_support=min_support, algorithm="graank")
    except So4gpNotInstalled:
        print("so4gp not installed (needs Python >=3.12). Install: uv pip install '.[patterns]'")
        print("Returning MOCK patterns for demonstration only.")
        return _mock_graank_patterns(df, min_support, target_columns)

    patterns = []
    for pat in result["patterns"]:
        items = pat["gradual_items"]
        patterns.append(
            {
                "pattern": _format_gradual_pattern(items),
                "support": pat["support"],
                "attributes": [gi["attribute"] for gi in items],
                "gradients": [gi["direction"] for gi in items],
            }
        )
    return patterns


def _mock_graank_patterns(
    df: pd.DataFrame,
    min_support: float,
    target_columns: list[str] | None,
) -> list[dict[str, Any]]:
    """Generate mock patterns when so4gp is not available."""
    if target_columns is None:
        target_columns = df.select_dtypes(include=["number"]).columns.tolist()

    # Return example patterns based on common columns
    mock_patterns = []

    if any(col.startswith("ndvi") for col in target_columns):
        if "precipitation" in target_columns or "precip" in target_columns:
            mock_patterns.append(
                {
                    "pattern": "{precipitation↑, ndvi↑}",
                    "support": 0.78,
                    "attributes": ["precipitation", "ndvi"],
                    "gradients": ["+", "+"],
                }
            )

        if "temperature" in target_columns or "temp" in target_columns:
            mock_patterns.append(
                {
                    "pattern": "{temperature↑, ndvi↓}",
                    "support": 0.65,
                    "attributes": ["temperature", "ndvi"],
                    "gradients": ["+", "-"],
                }
            )

    if not mock_patterns:
        # Generic pattern with first two columns
        if len(target_columns) >= 2:
            mock_patterns.append(
                {
                    "pattern": f"{{{target_columns[0]}↑, {target_columns[1]}↑}}",
                    "support": 0.70,
                    "attributes": target_columns[:2],
                    "gradients": ["+", "+"],
                }
            )

    return mock_patterns


def _format_gradual_pattern(gradual_items: list[dict]) -> str:
    """Render a co-monotonic gradual itemset, e.g. '{precip↑, temp↓, ndvi↑}'.

    Gradual patterns are sets of co-varying (attribute, direction) items, not
    antecedent->consequent rules, so there is deliberately no '→' arrow.
    """
    arrows = {"+": "↑", "-": "↓", None: "?"}
    parts = [f"{gi['attribute']}{arrows.get(gi['direction'], '?')}" for gi in gradual_items]
    return "{" + ", ".join(parts) + "}"


# =============================================================================
# Example Pipeline
# =============================================================================


async def example_pipeline():
    """Example: Full pipeline from Deposium to so4gp patterns.

    This demonstrates the intended workflow once Phase 2 is implemented.
    """
    print("=" * 60)
    print("Deposium → so4gp Integration Example")
    print("=" * 60)

    # Step 1: Define area of interest
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-1.5, 51.0],
                [-1.0, 51.0],
                [-1.0, 51.5],
                [-1.5, 51.5],
                [-1.5, 51.0],
            ]
        ],
    }

    print("\n[1] Area of Interest defined")
    print(f"    Bounding box: {polygon['coordinates'][0][0]} to {polygon['coordinates'][0][2]}")

    # Step 2: Get NDVI profile (mock data mirroring the real geo.ndvi_profile
    # output shape, so the example runs offline without a STAC catalog)
    print("\n[2] Fetching NDVI profile...")
    print("    NOTE: using mock data for offline demo (geo.ndvi_profile output shape).")

    mock_profile = {
        "geometry": polygon,
        "profile": [
            {"date": "2024-01-15", "ndvi_mean": 0.72, "cloud_cover": 5},
            {"date": "2024-02-01", "ndvi_mean": 0.68, "cloud_cover": 12},
            {"date": "2024-02-17", "ndvi_mean": 0.65, "cloud_cover": 3},
            {"date": "2024-03-05", "ndvi_mean": 0.58, "cloud_cover": 8},
            {"date": "2024-03-21", "ndvi_mean": 0.52, "cloud_cover": 15},
            {"date": "2024-04-06", "ndvi_mean": 0.61, "cloud_cover": 7},
            {"date": "2024-04-22", "ndvi_mean": 0.69, "cloud_cover": 4},
            {"date": "2024-05-08", "ndvi_mean": 0.74, "cloud_cover": 9},
        ],
        "statistics": {"mean": 0.65, "min": 0.52, "max": 0.74},
    }

    # Step 3: Convert to DataFrame
    print("\n[3] Converting to DataFrame...")
    df = ndvi_profile_to_dataframe(mock_profile)
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # Step 4: Enrich with climate data (placeholder)
    print("\n[4] Enriching with climate data...")
    # Add mock climate data
    df["precipitation"] = [45, 52, 38, 25, 18, 42, 55, 48]
    df["temperature"] = [8, 9, 11, 14, 17, 15, 13, 12]
    print("    Added: precipitation, temperature")

    # Step 5: Run GRAANK
    print("\n[5] Running GRAANK pattern mining...")
    patterns = run_graank(
        df,
        min_support=0.5,
        target_columns=["ndvi_mean", "precipitation", "temperature"],
    )

    print(f"\n[6] Discovered {len(patterns)} patterns:")
    for i, p in enumerate(patterns, 1):
        print(f"    {i}. {p['pattern']}")
        print(f"       Support: {p['support']:.2f}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)

    return patterns


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_pipeline())
