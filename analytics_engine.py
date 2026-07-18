"""Dataset-agnostic profiling, exploratory analysis, and insight generation.

This module intentionally does not require a business-specific schema.  It
infers column roles from the uploaded data and only emits analyses supported
by the available columns.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


MAX_CONTEXT_RECORDS = 50
MAX_COLUMNS_PER_ANALYSIS = 10


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Return a dataframe as JSON-safe records."""

    selected = frame if columns is None else frame.loc[:, list(columns)]
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in selected.to_dict(orient="records")
    ]


def _normalise_name(value: Any) -> str:
    """Create a stable, lower-case name for semantic column matching."""

    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _safe_pct_change(current: float, previous: float) -> float | None:
    """Calculate a percentage change without dividing by zero."""

    if previous is None or not np.isfinite(previous) or previous == 0:
        return None
    return float((current - previous) / abs(previous) * 100)


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Trim column labels and make duplicate labels unique without imposing names."""

    frame = data.copy()
    used: dict[str, int] = {}
    labels: list[str] = []
    for position, label in enumerate(frame.columns, start=1):
        clean_label = str(label).strip() or f"unnamed_column_{position}"
        count = used.get(clean_label, 0)
        used[clean_label] = count + 1
        labels.append(clean_label if count == 0 else f"{clean_label}_{count + 1}")
    frame.columns = labels
    return frame


def _parse_rate(series: pd.Series, parser: str) -> float:
    """Return the fraction of non-null values successfully parsed."""

    values = series.dropna()
    if values.empty:
        return 0.0
    if parser == "numeric":
        parsed = pd.to_numeric(values, errors="coerce")
    else:
        try:
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        except (TypeError, ValueError):  # pandas < 2.0 compatibility
            parsed = pd.to_datetime(values, errors="coerce")
    return float(parsed.notna().mean())


def detect_column_types(data: pd.DataFrame) -> dict[str, Any]:
    """Infer numerical, categorical, datetime, boolean, and text columns."""

    frame = standardize_columns(data)
    numerical: list[str] = []
    categorical: list[str] = []
    datetime_columns: list[str] = []
    boolean: list[str] = []
    text: list[str] = []
    type_by_column: dict[str, str] = {}
    parse_rates: dict[str, dict[str, float]] = {}

    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        unique_count = int(non_null.nunique(dropna=True))
        unique_ratio = unique_count / max(len(non_null), 1)
        numeric_rate = _parse_rate(series, "numeric")
        datetime_rate = _parse_rate(series, "datetime")
        parse_rates[column] = {"numeric": numeric_rate, "datetime": datetime_rate}

        if pd.api.types.is_bool_dtype(series) or (
            not non_null.empty and set(non_null.astype(str).str.lower().unique()) <= {"true", "false", "yes", "no"}
        ):
            boolean.append(column)
            type_by_column[column] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series) or (
            not pd.api.types.is_numeric_dtype(series) and datetime_rate >= 0.8 and unique_count >= 2
        ):
            datetime_columns.append(column)
            type_by_column[column] = "datetime"
        elif pd.api.types.is_numeric_dtype(series) or numeric_rate >= 0.95 and unique_count >= 1:
            numerical.append(column)
            type_by_column[column] = "numerical"
        else:
            average_length = float(non_null.astype(str).str.len().mean()) if not non_null.empty else 0.0
            categorical_limit = min(50, max(10, int(max(len(non_null), 1) * 0.2)))
            text_name_hint = any(
                token in _normalise_name(column).split()
                for token in ("text", "note", "notes", "description", "comment", "comments", "message", "review", "feedback", "address", "body")
            )
            likely_free_text = unique_ratio >= 0.8 and average_length > 20
            if not text_name_hint and not likely_free_text and unique_count <= categorical_limit and average_length <= 80:
                categorical.append(column)
                type_by_column[column] = "categorical"
            else:
                text.append(column)
                type_by_column[column] = "text"

    return {
        "type_by_column": type_by_column,
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean,
        "text_columns": text,
        "identifier_like_columns": [
            column for column in frame.columns
            if frame[column].nunique(dropna=True) / max(frame[column].notna().sum(), 1) >= 0.95
        ],
        "parse_rates": parse_rates,
    }


def validate_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Validate only that the upload is a usable tabular CSV, not its schema."""

    if not isinstance(data, pd.DataFrame):
        return {"valid": False, "errors": ["The uploaded object is not a table."], "warnings": []}
    had_duplicate_columns = bool(data.columns.duplicated().any())
    frame = standardize_columns(data)
    warnings: list[str] = []
    if frame.empty:
        warnings.append("The uploaded CSV contains no rows. Only column metadata can be profiled.")
    if len(frame.columns) == 0:
        return {"valid": False, "errors": ["The uploaded CSV contains no columns."], "warnings": warnings}
    if had_duplicate_columns:
        warnings.append("Duplicate column names were renamed with a suffix.")
    all_empty = [str(column) for column in frame.columns if frame[column].isna().all()]
    if all_empty:
        warnings.append(f"Empty column(s) detected: {', '.join(all_empty[:10])}.")
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Apply generic, non-destructive cleaning based on inferred column types."""

    frame = standardize_columns(data).copy()
    detected = detect_column_types(frame)
    for column in detected["numerical_columns"]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in detected["datetime_columns"]:
        if not pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in frame.columns:
        if pd.api.types.is_string_dtype(frame[column]) or frame[column].dtype == object:
            frame[column] = frame[column].astype("string").str.strip()
    return frame.drop_duplicates().reset_index(drop=True)


def _numeric_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Summarize numeric distributions."""

    output: list[dict[str, Any]] = []
    for column in columns[:MAX_COLUMNS_PER_ANALYSIS]:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.dropna().empty:
            continue
        output.append(
            {
                "column": column,
                "count": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": _json_value(series.std()),
                "min": float(series.min()),
                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75)),
                "max": float(series.max()),
                "unique": int(series.nunique()),
            }
        )
    return output


def _categorical_summary(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Summarize the most common values for categorical fields."""

    output: list[dict[str, Any]] = []
    for column in columns[:MAX_COLUMNS_PER_ANALYSIS]:
        values = frame[column].astype("string").fillna("<missing>")
        counts = values.value_counts(dropna=False).head(10).rename_axis("value").reset_index(name="count")
        counts["share_pct"] = counts["count"] / max(len(frame), 1) * 100
        output.append({"column": column, "unique": int(values.nunique()), "top_values": _records(counts)})
    return output


def _grouped_summaries(frame: pd.DataFrame, categorical_columns: list[str], numerical_columns: list[str]) -> list[dict[str, Any]]:
    """Create reusable category-by-metric aggregates for comparison charts."""

    output: list[dict[str, Any]] = []
    for dimension in categorical_columns[:5]:
        groups = frame[dimension].astype("string").fillna("<missing>")
        for metric in numerical_columns[:5]:
            working = pd.DataFrame({"dimension": groups, "metric": pd.to_numeric(frame[metric], errors="coerce")}).dropna(subset=["metric"])
            if working.empty or working["dimension"].nunique() < 2:
                continue
            grouped = (
                working.groupby("dimension", as_index=False)["metric"]
                .agg(total="sum", average="mean", count="count")
                .sort_values("total", ascending=False)
                .head(15)
            )
            output.append(
                {
                    "dimension": dimension,
                    "metric": metric,
                    "records": _records(grouped),
                }
            )
    return output[:20]


def _correlations(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Return a compact correlation matrix and strongest pairs."""

    selected = columns[:MAX_COLUMNS_PER_ANALYSIS]
    if len(selected) < 2:
        return {"matrix": [], "strongest_pairs": []}
    matrix = frame[selected].corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []
    for index, first in enumerate(matrix.columns):
        for second in matrix.columns[index + 1:]:
            value = matrix.loc[first, second]
            if pd.notna(value):
                pairs.append({"column_1": first, "column_2": second, "correlation": float(value), "absolute_correlation": abs(float(value))})
    return {
        "columns": list(matrix.columns),
        "matrix": [{"column": row, **{str(column): _json_value(matrix.loc[row, column]) for column in matrix.columns}} for row in matrix.index],
        "strongest_pairs": sorted(pairs, key=lambda item: item["absolute_correlation"], reverse=True)[:20],
    }


def _time_series_analysis(frame: pd.DataFrame, datetime_columns: list[str], numerical_columns: list[str]) -> list[dict[str, Any]]:
    """Aggregate numeric fields by month for each usable datetime field."""

    output: list[dict[str, Any]] = []
    for date_column in datetime_columns[:3]:
        date_values = pd.to_datetime(frame[date_column], errors="coerce")
        for metric in numerical_columns[:5]:
            working = pd.DataFrame({"period": date_values.dt.to_period("M").dt.to_timestamp(), "value": pd.to_numeric(frame[metric], errors="coerce")}).dropna()
            if working["period"].nunique() < 2:
                continue
            trend = working.groupby("period", as_index=False)["value"].agg(["sum", "mean", "count"]).reset_index()
            trend = trend.rename(columns={"sum": "total", "mean": "average"}).sort_values("period")
            records = _records(trend)
            for record in records:
                record["period"] = pd.Timestamp(record["period"]).strftime("%Y-%m")
            first_total, last_total = float(trend["total"].iloc[0]), float(trend["total"].iloc[-1])
            change = _safe_pct_change(last_total, first_total)
            output.append(
                {
                    "date_column": date_column,
                    "metric": metric,
                    "granularity": "month",
                    "period_count": int(len(trend)),
                    "trend_direction": "increasing" if last_total > first_total else "declining" if last_total < first_total else "flat",
                    "overall_change_pct": change,
                    "records": records[-MAX_CONTEXT_RECORDS:],
                }
            )
    return output


def _anomalies(frame: pd.DataFrame, numerical_columns: list[str]) -> list[dict[str, Any]]:
    """Detect numeric outlier counts using the IQR rule."""

    output: list[dict[str, Any]] = []
    for column in numerical_columns[:MAX_COLUMNS_PER_ANALYSIS]:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            outliers = series.iloc[0:0]
        else:
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
        output.append({"column": column, "outlier_count": int(len(outliers)), "outlier_share_pct": float(len(outliers) / len(series) * 100), "method": "IQR"})
    return output


def _customer_insights(frame: pd.DataFrame, detected: dict[str, Any], numeric_summary: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize customer-like fields when their names are present, without requiring them."""

    keywords = ("customer", "client", "user", "member", "account", "buyer", "shopper")
    candidates = [column for column in frame.columns if any(keyword in _normalise_name(column).split() for keyword in keywords)]
    if not candidates:
        return {"available": False, "reason": "No customer-like column was detected."}
    customer_ids = [column for column in candidates if frame[column].nunique(dropna=True) > 1 and frame[column].nunique(dropna=True) / max(frame[column].notna().sum(), 1) > 0.2]
    return {
        "available": True,
        "customer_like_columns": candidates,
        "identifier_candidates": customer_ids,
        "summary": [
            {"column": column, "unique_values": int(frame[column].nunique(dropna=True)), "missing": int(frame[column].isna().sum())}
            for column in candidates
        ],
        "note": "Customer-level value or cohort analysis is limited because no explicit transaction grain or metric mapping is assumed.",
    }


def _generate_kpis(frame: pd.DataFrame, numeric_summary: list[dict[str, Any]], detected: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate a compact set of data-backed headline cards without fixed metrics."""

    kpis: list[dict[str, Any]] = [
        {"id": "row_count", "label": "Rows analysed", "value": int(len(frame)), "calculation": "count", "source": "dataset"},
        {"id": "column_count", "label": "Columns detected", "value": int(len(frame.columns)), "calculation": "count", "source": "dataset"},
    ]
    for item in numeric_summary[:4]:
        kpis.append(
            {
                "id": f"average_{item['column']}",
                "label": f"Average {item['column']}",
                "value": item["mean"],
                "calculation": "average",
                "column": item["column"],
                "source": "numeric_summary",
            }
        )
    if len(kpis) < 4 and detected["categorical_columns"]:
        first_category = detected["categorical_columns"][0]
        kpis.append(
            {
                "id": "category_values",
                "label": f"Unique {first_category}",
                "value": int(frame[first_category].nunique(dropna=True)),
                "calculation": "count_distinct",
                "column": first_category,
                "source": "profile",
            }
        )
    if len(kpis) < 4 and detected["datetime_columns"]:
        kpis.append(
            {
                "id": "date_fields",
                "label": "Datetime fields",
                "value": int(len(detected["datetime_columns"])),
                "calculation": "count",
                "source": "profile",
            }
        )
    return kpis[:6]


def _recommend_visualizations(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Recommend chart intents from the available analytical structures."""

    profile = results["profile"]
    numeric = profile["numerical_columns"]
    categorical = profile["categorical_columns"]
    recommendations: list[dict[str, Any]] = []
    trends = results.get("time_series_trends", [])
    for index, trend in enumerate(trends[:6], start=1):
        recommendations.append(
            {
                "id": f"trend_{index}",
                "kind": "time_series",
                "title": f"{trend['metric']} over time",
                "columns": [trend["date_column"], trend["metric"]],
                "priority": 1 if index == 1 else 2,
                "rationale": "A datetime field and numerical metric support a monthly trend chart.",
            }
        )
    if categorical and results.get("categorical_distributions"):
        for index, distribution in enumerate(results["categorical_distributions"][:6], start=1):
            recommendations.append(
                {
                    "id": f"category_{index}",
                    "kind": "categorical_bar",
                    "title": f"Top values in {distribution['column']}",
                    "columns": [distribution["column"]],
                    "priority": 3,
                    "rationale": "A categorical field supports a frequency comparison.",
                }
            )
            if distribution["unique"] <= 8:
                recommendations.append(
                    {
                        "id": f"category_pie_{index}",
                        "kind": "categorical_pie",
                        "title": f"Composition of {distribution['column']}",
                        "columns": [distribution["column"]],
                        "priority": 4,
                        "rationale": "A low-cardinality categorical field supports a composition view.",
                    }
                )
    for index, grouped in enumerate(results.get("grouped_summaries", [])[:8], start=1):
        recommendations.append(
            {
                "id": f"grouped_{index}",
                "kind": "grouped_bar",
                "title": f"{grouped['metric']} by {grouped['dimension']}",
                "columns": [grouped["dimension"], grouped["metric"]],
                "priority": 4,
                "rationale": "A categorical and numerical field support an aggregated comparison.",
            }
        )
    for index, summary in enumerate(results.get("numeric_summary", [])[:6], start=1):
        recommendations.append(
            {
                "id": f"distribution_{index}",
                "kind": "histogram",
                "title": f"Distribution of {summary['column']}",
                "columns": [summary["column"]],
                "priority": 5,
                "rationale": "A numerical field supports distribution and outlier review.",
            }
        )
        recommendations.append(
            {
                "id": f"boxplot_{index}",
                "kind": "boxplot",
                "title": f"Spread of {summary['column']}",
                "columns": [summary["column"]],
                "priority": 6,
                "rationale": "A numerical field supports range and outlier review with a boxplot.",
            }
        )
    if len(numeric) >= 2 and results.get("correlations", {}).get("matrix"):
        recommendations.append(
            {
                "id": "correlations",
                "kind": "correlation_heatmap",
                "title": "Numerical relationships",
                "columns": numeric[:MAX_COLUMNS_PER_ANALYSIS],
                "priority": 7,
                "rationale": "Multiple numerical fields support correlation analysis.",
            }
        )
    scatter_pairs = [(numeric[0], numeric[1])] if len(numeric) >= 2 else []
    if len(numeric) >= 4:
        scatter_pairs.extend([(numeric[0], numeric[2]), (numeric[1], numeric[3])])
    for index, pair in enumerate(scatter_pairs, start=1):
        recommendations.append(
            {
                "id": f"relationship_{index}",
                "kind": "scatter",
                "title": f"{pair[0]} vs {pair[1]}",
                "columns": list(pair),
                "priority": 8,
                "rationale": "Two numerical fields support an exploratory relationship plot.",
            }
        )
    return recommendations[:30]


def _generate_recommendations(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn verified findings and quality signals into cautious next steps."""

    recommendations: list[dict[str, Any]] = []
    for trend in results.get("time_series_trends", [])[:2]:
        if trend.get("overall_change_pct") is not None:
            recommendations.append(
                {
                    "title": f"Monitor {trend['metric']}",
                    "message": f"Review the {trend['metric']} trend over time and investigate the drivers behind the {trend['trend_direction']} pattern.",
                    "evidence": {key: value for key, value in trend.items() if key != "records"},
                }
            )
    for anomaly in [item for item in results.get("anomalies", []) if item.get("outlier_count")][:2]:
        recommendations.append(
            {
                "title": f"Review outliers in {anomaly['column']}",
                "message": "Validate whether flagged observations represent genuine exceptional events or data-quality issues before using them for decisions.",
                "evidence": anomaly,
            }
        )
    missing = results.get("data_quality", {}).get("missing_values", {})
    if missing:
        recommendations.append(
            {
                "title": "Improve data completeness",
                "message": "Review fields with missing values before relying on segment comparisons or downstream models.",
                "evidence": {"missing_values": missing},
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "title": "Continue monitoring the available metrics",
                "message": "The dashboard did not detect a specific action signal from the available fields.",
                "evidence": {"available_analyses": results.get("available_analyses", [])},
            }
        )
    return recommendations[:5]


def _availability(detected: dict[str, Any], results: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Describe which analytical modules could and could not run."""

    available: list[str] = []
    unavailable: dict[str, str] = {}
    numeric = detected["numerical_columns"]
    categorical = detected["categorical_columns"]
    dates = detected["datetime_columns"]
    if numeric:
        available.extend(["numeric_summary", "distributions"])
    else:
        unavailable["numeric_summary"] = "No numerical columns were detected."
        unavailable["distributions"] = "No numerical columns were detected."
    if len(numeric) >= 2:
        available.append("correlation_analysis")
    else:
        unavailable["correlation_analysis"] = "At least two numerical columns are required."
    if categorical:
        available.append("categorical_distributions")
    else:
        unavailable["categorical_distributions"] = "No categorical columns were detected."
    if dates and numeric:
        available.append("time_series_trends")
    elif not dates:
        unavailable["time_series_trends"] = "No datetime column was detected."
    else:
        unavailable["time_series_trends"] = "A datetime column exists, but no numerical metric is available."
    if results["customer_insights"].get("available"):
        available.append("customer_insights")
    else:
        unavailable["customer_insights"] = results["customer_insights"].get("reason", "No customer-like fields were detected.")
    if results["anomalies"]:
        available.append("anomaly_detection")
    else:
        unavailable["anomaly_detection"] = "At least four non-null observations in a numerical column are required."
    return sorted(set(available)), unavailable


def analyze_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Run all analyses that are supported by the uploaded dataset."""

    validation = validate_dataset(data)
    if not validation["valid"]:
        raise ValueError(" ".join(validation.get("errors", ["The dataset is not valid."])))
    frame = clean_dataset(data)
    detected = detect_column_types(frame)
    numeric = detected["numerical_columns"]
    categorical = detected["categorical_columns"]
    dates = detected["datetime_columns"]
    numeric_summary = _numeric_summary(frame, numeric)
    results: dict[str, Any] = {
        "profile": {
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "columns": [str(column) for column in frame.columns],
            "type_by_column": detected["type_by_column"],
            "numerical_columns": numeric,
            "categorical_columns": categorical,
            "datetime_columns": dates,
            "boolean_columns": detected["boolean_columns"],
            "text_columns": detected["text_columns"],
            "identifier_like_columns": detected["identifier_like_columns"],
        },
        "data_quality": {
            "rows_received": int(len(data)),
            "rows_after_cleaning": int(len(frame)),
            "rows_removed": int(len(data) - len(frame)),
            "duplicate_rows_received": validation.get("duplicate_rows", 0),
            "missing_values": {str(column): int(count) for column, count in frame.isna().sum().items() if count},
            "validation_warnings": validation.get("warnings", []),
        },
        "numeric_summary": numeric_summary,
        "categorical_distributions": _categorical_summary(frame, categorical),
        "grouped_summaries": _grouped_summaries(frame, categorical, numeric),
        "correlations": _correlations(frame, numeric),
        "time_series_trends": _time_series_analysis(frame, dates, numeric),
        "anomalies": _anomalies(frame, numeric),
        "customer_insights": _customer_insights(frame, detected, numeric_summary),
    }
    results["available_analyses"], results["unavailable_analyses"] = _availability(detected, results)
    results["kpis"] = _generate_kpis(frame, numeric_summary, detected)
    results["visualization_recommendations"] = _recommend_visualizations(results)
    results["dashboard_metadata"] = {
        "title": "DataWonder Executive Dashboard",
        "subtitle": f"Automatically generated from {len(frame):,} rows and {len(frame.columns):,} detected fields",
        "sections": ["kpis", "main_charts", "supporting_charts", "correlations", "data_quality", "insights", "recommendations", "ai_copilot"],
        "chart_count": len(results["visualization_recommendations"]),
        "generation_mode": "automatic",
    }
    results["insights"] = generate_insights(results)
    results["recommendations"] = _generate_recommendations(results)
    return results


def generate_insights(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate deterministic findings only from computed analytical outputs."""

    insights: list[dict[str, Any]] = []
    numeric = results.get("numeric_summary", [])
    if numeric:
        widest = max(numeric, key=lambda item: item["max"] - item["min"])
        insights.append({
            "type": "distribution",
            "title": "Largest observed numeric range",
            "message": f"{widest['column']} spans from {widest['min']:,.2f} to {widest['max']:,.2f} across {widest['count']:,} non-null values.",
            "evidence": widest,
        })
    correlations = results.get("correlations", {}).get("strongest_pairs", [])
    if correlations:
        pair = correlations[0]
        direction = "positive" if pair["correlation"] > 0 else "negative"
        insights.append({
            "type": "relationship",
            "title": "Strongest numerical relationship",
            "message": f"{pair['column_1']} and {pair['column_2']} have a {direction} correlation of {pair['correlation']:.2f}. Correlation does not establish causation.",
            "evidence": pair,
        })
    trends = results.get("time_series_trends", [])
    if trends:
        strongest = max((item for item in trends if item.get("overall_change_pct") is not None), key=lambda item: abs(item["overall_change_pct"]), default=None)
        if strongest:
            insights.append({
                "type": "trend",
                "title": "Largest time-series movement",
                "message": f"Monthly total of {strongest['metric']} was {strongest['trend_direction']} by {abs(strongest['overall_change_pct']):.1f}% over the observed period using {strongest['date_column']}.",
                "evidence": {key: value for key, value in strongest.items() if key != "records"},
            })
    distributions = results.get("categorical_distributions", [])
    if distributions:
        distribution = distributions[0]
        top = distribution["top_values"][0] if distribution["top_values"] else None
        if top:
            insights.append({
                "type": "composition",
                "title": "Most common category value",
                "message": f"{top['value']} is the most common value in {distribution['column']}, representing {top['share_pct']:.1f}% of rows.",
                "evidence": {"column": distribution["column"], **top},
            })
    anomaly = max(results.get("anomalies", []), key=lambda item: item["outlier_count"], default=None)
    if anomaly and anomaly["outlier_count"]:
        insights.append({
            "type": "anomaly",
            "title": "Potential numerical outliers",
            "message": f"{anomaly['column']} contains {anomaly['outlier_count']:,} IQR outlier(s), or {anomaly['outlier_share_pct']:.1f}% of its non-null values.",
            "evidence": anomaly,
        })
    return insights


def build_ai_context(results: dict[str, Any], insights: list[dict[str, Any]] | None = None) -> str:
    """Serialize verified profile, analyses, availability, and findings for the AI."""

    context = {
        "profile": results.get("profile", {}),
        "data_quality": results.get("data_quality", {}),
        "available_analyses": results.get("available_analyses", []),
        "unavailable_analyses": results.get("unavailable_analyses", {}),
        "numeric_summary": results.get("numeric_summary", []),
        "categorical_distributions": results.get("categorical_distributions", []),
        "grouped_summaries": results.get("grouped_summaries", []),
        "correlations": results.get("correlations", {}),
        "time_series_trends": results.get("time_series_trends", []),
        "anomalies": results.get("anomalies", []),
        "customer_insights": results.get("customer_insights", {}),
        "kpis": results.get("kpis", []),
        "visualization_recommendations": results.get("visualization_recommendations", []),
        "dashboard_metadata": results.get("dashboard_metadata", {}),
        "recommendations": results.get("recommendations", []),
        "verified_insights": insights if insights is not None else results.get("insights", []),
    }
    return json.dumps(context, indent=2, ensure_ascii=False, allow_nan=False)
