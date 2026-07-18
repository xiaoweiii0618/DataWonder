"""DataWonder: automatic executive dashboard generation from any CSV."""

from __future__ import annotations

import hashlib
import json
import textwrap
from html import escape
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is installed from requirements in the app runtime
    Image = ImageDraw = ImageFont = None

try:
    from streamlit_sortables import sort_items
except ImportError:  # pragma: no cover - dependency is installed from requirements in the app runtime
    sort_items = None

from ai_agent import answer_question
from analytics_engine import analyze_dataset, build_ai_context, clean_dataset, validate_dataset


st.set_page_config(page_title="DataWonder | Automatic Dashboard", page_icon="📊", layout="wide")


MIN_DASHBOARD_COLUMNS = 1
MAX_DASHBOARD_COLUMNS = 3
DEFAULT_DASHBOARD_COLUMNS = 2


def _initialise_state() -> None:
    """Create the small amount of state needed for the dashboard workflow."""

    defaults = {
        "dataset_signature": None,
        "raw_data": None,
        "cleaned_data": None,
        "analytics": None,
        "analytics_context": "",
        "chat_history": [],
        "upload_error": None,
        "edit_mode": False,
        "dashboard_theme": "Executive",
        "chart_overrides": {},
        "custom_sections": [],
        "custom_visualizations": [],
        "story_text": "",
        "dashboard_title": "",
        "dashboard_subtitle": "",
        "dashboard_layout": {"columns": DEFAULT_DASHBOARD_COLUMNS, "lanes": [], "customized": False},
        "dashboard_columns_selector": DEFAULT_DASHBOARD_COLUMNS,
        "dashboard_column_notice": None,
        "dashboard_filters": {},
        "loaded_config_signature": None,
        "selected_auto_chart_ids": None,
        "customize_auto_chart_ids": [],
        "selected_generated_chart_ids": None,
        "customize_kpi_ids": [],
        "custom_kpis": [],
        "kpi_overrides": {},
        "kpi_builder_version": 0,
        "customize_summary_section_ids": [],
        "chart_editor_version": 0,
        "chart_editor_preview": None,
        "chart_generator_draft": None,
        "chart_generator_last_selections": {},
        "saved_charts": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_chart_generator_widgets() -> None:
    """Clear field widgets whose options depend on the currently uploaded dataset."""

    for key in list(st.session_state.keys()):
        if str(key).startswith("chart_generator_"):
            st.session_state.pop(key, None)
    for key in ("generate_chart_button", "save_generated_chart", "add_generated_chart_to_dashboard"):
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def _process_dataset(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cache deterministic cleaning and dashboard metadata generation."""

    cleaned = clean_dataset(raw_data)
    return cleaned, analyze_dataset(cleaned)


def _clear_dataset(signature: str, error: dict) -> None:
    """Clear stale dashboard state when the selected upload is unusable."""

    _reset_chart_generator_widgets()
    for widget_key in ("customize_kpi_selector", "kpi_edit_selection"):
        st.session_state.pop(widget_key, None)
    st.session_state.update(
        dataset_signature=signature,
        raw_data=None,
        cleaned_data=None,
        analytics=None,
        analytics_context="",
        chat_history=[],
        upload_error=error,
        edit_mode=False,
        chart_overrides={},
        custom_sections=[],
        custom_visualizations=[],
        story_text="",
        dashboard_title="",
        dashboard_subtitle="",
        dashboard_layout={"columns": DEFAULT_DASHBOARD_COLUMNS, "lanes": [], "customized": False},
        dashboard_columns_selector=DEFAULT_DASHBOARD_COLUMNS,
        dashboard_column_notice=None,
        dashboard_filters={},
        loaded_config_signature=None,
        selected_auto_chart_ids=None,
        customize_auto_chart_ids=[],
        selected_generated_chart_ids=None,
        customize_kpi_ids=[],
        custom_kpis=[],
        kpi_overrides={},
        kpi_builder_version=0,
        customize_summary_section_ids=[],
        chart_editor_version=0,
        chart_editor_preview=None,
        chart_generator_draft=None,
        chart_generator_last_selections={},
        saved_charts=[],
    )


def _load_upload(uploaded_file) -> None:
    """Load any CSV and generate its dashboard automatically."""

    payload = uploaded_file.getvalue()
    signature = hashlib.sha256(payload).hexdigest()
    if signature == st.session_state.dataset_signature:
        return

    try:
        _reset_chart_generator_widgets()
        for widget_key in ("customize_kpi_selector", "kpi_edit_selection"):
            st.session_state.pop(widget_key, None)
        raw = pd.read_csv(BytesIO(payload))
        validation = validate_dataset(raw)
        if not validation["valid"]:
            _clear_dataset(signature, validation)
            return
        cleaned, analytics = _process_dataset(raw)
        st.session_state.update(
            dataset_signature=signature,
            raw_data=raw,
            cleaned_data=cleaned,
            analytics=analytics,
            analytics_context=build_ai_context(analytics),
            chat_history=[],
            upload_error=None,
            edit_mode=False,
            chart_overrides={},
            custom_sections=[],
            custom_visualizations=[],
            story_text="",
            dashboard_title=analytics.get("dashboard_metadata", {}).get("title", "DataWonder Executive Dashboard"),
            dashboard_subtitle=analytics.get("dashboard_metadata", {}).get("subtitle", ""),
            dashboard_layout={"columns": DEFAULT_DASHBOARD_COLUMNS, "lanes": [], "customized": False},
            dashboard_columns_selector=DEFAULT_DASHBOARD_COLUMNS,
            dashboard_column_notice=None,
            dashboard_filters={},
            loaded_config_signature=None,
            selected_auto_chart_ids=None,
            customize_auto_chart_ids=[],
            selected_generated_chart_ids=None,
            customize_kpi_ids=[],
            custom_kpis=[],
            kpi_overrides={},
            kpi_builder_version=0,
            customize_summary_section_ids=[],
            chart_editor_version=0,
            chart_generator_draft=None,
            chart_generator_last_selections={},
            saved_charts=[],
        )
    except Exception as exc:  # noqa: BLE001 - convert parsing failures to UI feedback
        _clear_dataset(signature, {"errors": [f"Could not process the CSV: {exc}"], "warnings": []})


def _format_value(value: object) -> str:
    """Format KPI values consistently without assuming their business meaning."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return f"{value:,.2f}"
        return f"{value:,.0f}"
    return str(value)


THEMES = {
    "Professional": {
        "background": "#f6f8fb", "surface": "#ffffff", "accent": "#2563eb", "text": "#172033", "muted": "#64748b", "border": "#dbe3ef", "grid": "#e2e8f0", "table_header": "#eff6ff",
        "chart_colors": ["#2563eb", "#0ea5e9", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6"],
        "heatmap_colorscale": [[0, "#dbeafe"], [0.5, "#ffffff"], [1, "#1d4ed8"]],
        "plotly_template": "plotly_white",
    },
    "Executive": {
        "background": "#f4f5f7", "surface": "#ffffff", "accent": "#2563eb", "text": "#111827", "muted": "#64748b", "border": "#d8dee8", "grid": "#e2e8f0", "table_header": "#f1f5f9",
        "chart_colors": ["#2563eb", "#0f766e", "#64748b", "#d97706", "#7c3aed", "#0891b2"],
        "heatmap_colorscale": [[0, "#cbd5e1"], [0.5, "#ffffff"], [1, "#1d4ed8"]],
        "plotly_template": "plotly_white",
    },
    "Marketing": {
        "background": "#fff7ed", "surface": "#ffffff", "accent": "#ea580c", "text": "#431407", "muted": "#9a3412", "border": "#fed7aa", "grid": "#fdba74", "table_header": "#ffedd5",
        "chart_colors": ["#ea580c", "#f97316", "#f59e0b", "#ec4899", "#8b5cf6", "#14b8a6"],
        "heatmap_colorscale": [[0, "#fed7aa"], [0.5, "#ffffff"], [1, "#c2410c"]],
        "plotly_template": "plotly_white",
    },
    "Minimal": {
        "background": "#ffffff", "surface": "#ffffff", "accent": "#475569", "text": "#0f172a", "muted": "#64748b", "border": "#e2e8f0", "grid": "#e2e8f0", "table_header": "#f8fafc",
        "chart_colors": ["#475569", "#64748b", "#94a3b8", "#cbd5e1", "#334155", "#0f172a"],
        "heatmap_colorscale": [[0, "#e2e8f0"], [0.5, "#ffffff"], [1, "#334155"]],
        "plotly_template": "plotly_white",
    },
    "Dark Mode": {
        "background": "#111827", "surface": "#1f2937", "accent": "#60a5fa", "text": "#f9fafb", "muted": "#cbd5e1", "border": "#374151", "grid": "#4b5563", "table_header": "#374151",
        "chart_colors": ["#60a5fa", "#22d3ee", "#34d399", "#fbbf24", "#f472b6", "#c084fc"],
        "heatmap_colorscale": [[0, "#172554"], [0.5, "#1f2937"], [1, "#f8fafc"]],
        "plotly_template": "plotly_dark",
    },
}


class ThemeManager:
    """Centralized theme access for the Streamlit page, figures, tables, and exports."""

    _SAFE_FALLBACK_PALETTE = ["#2563eb", "#0f766e", "#d97706", "#7c3aed", "#db2777", "#0891b2"]
    _SAFE_FALLBACK_HEATMAP = [[0, "#dbeafe"], [0.5, "#ffffff"], [1, "#1d4ed8"]]
    _EXTRA_PALETTES = {
        "Set2": px.colors.qualitative.Set2,
        "Blues": px.colors.sequential.Blues,
        "Viridis": px.colors.sequential.Viridis,
    }

    def __init__(self, themes: dict[str, dict]):
        self.themes = themes

    def active(self, theme_name: str | None = None) -> dict:
        name = theme_name or st.session_state.get("dashboard_theme", "Executive")
        return self.themes.get(name, self.themes["Executive"])

    def palette(self, color_theme: str = "Default") -> list[str]:
        if color_theme in self._EXTRA_PALETTES:
            return list(self._EXTRA_PALETTES[color_theme])
        return list(self.active().get("chart_colors") or self._SAFE_FALLBACK_PALETTE)

    def heatmap_colorscale(self) -> list[list[object]]:
        return self.active().get("heatmap_colorscale") or self._SAFE_FALLBACK_HEATMAP

    def kpi_color(self, color_name: str = "Auto (theme)") -> str:
        """Resolve KPI accent and icon colors from the active global theme palette."""

        theme = self.active()
        palette = self.palette()
        colors = {
            "Auto (theme)": theme["accent"],
            "Primary": palette[0],
            "Secondary": palette[1 % len(palette)],
            "Accent": theme["accent"],
            "Positive": palette[2 % len(palette)],
            "Warning": palette[3 % len(palette)],
        }
        return colors.get(color_name, theme["accent"])

    @staticmethod
    def _rgba(color: str, alpha: float) -> str:
        """Convert a hex color to rgba for area fills without relying on Plotly defaults."""

        value = str(color).lstrip("#")
        if len(value) == 3:
            value = "".join(character * 2 for character in value)
        try:
            red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
        except (TypeError, ValueError):
            return color
        return f"rgba({red},{green},{blue},{alpha})"

    def apply_page(self, theme_name: str | None = None) -> None:
        theme = self.active(theme_name)
        st.markdown(
            f"""
            <style>
            :root {{
                --dw-background: {theme['background']};
                --dw-surface: {theme['surface']};
                --dw-accent: {theme['accent']};
                --dw-text: {theme['text']};
                --dw-muted: {theme['muted']};
                --dw-border: {theme['border']};
                --dw-table-header: {theme['table_header']};
            }}
            .stApp {{ background: var(--dw-background); color: var(--dw-text); }}
            [data-testid="stMetric"] {{ background: var(--dw-surface); border: 1px solid var(--dw-border); border-radius: 14px; padding: 14px; }}
            [data-testid="stMetricLabel"] {{ color: var(--dw-muted); }}
            [data-testid="stMetricValue"] {{ color: var(--dw-accent); }}
            [data-testid="stMetricDelta"] {{ color: var(--dw-accent); }}
            [data-testid="stDataFrame"] {{ border: 1px solid var(--dw-border); border-radius: 12px; overflow: hidden; }}
            .dw-section {{ background: var(--dw-surface); color: var(--dw-text); border-radius: 14px; padding: 12px 18px; margin: 10px 0; border: 1px solid var(--dw-border); }}
            .dw-kpi-card {{ background: var(--dw-surface); color: var(--dw-text); border: 1px solid var(--dw-border); border-radius: 14px; padding: 14px; min-height: 104px; box-shadow: 0 6px 18px rgba(15, 23, 42, .08); }}
            .dw-kpi-card .dw-kpi-icon {{ color: var(--dw-kpi-accent, var(--dw-accent)); font-size: 20px; }}
            .dw-kpi-card .dw-kpi-label {{ color: var(--dw-muted); font-size: 13px; margin-top: 6px; }}
            .dw-kpi-card .dw-kpi-value {{ color: var(--dw-kpi-accent, var(--dw-accent)); font-size: 25px; font-weight: 700; margin-top: 6px; }}
            .dw-kpi-card .dw-kpi-description {{ color: var(--dw-muted); font-size: 12px; margin-top: 4px; }}
            .dw-layout-handle {{ color: var(--dw-muted); font-size: 12px; font-weight: 600; letter-spacing: .02em; margin-bottom: 8px; }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    def apply_figure(self, figure, palette: list[str] | None = None, chart_style: str = "Auto", theme: dict | None = None):
        theme = theme or self.active()
        chosen_palette = list(palette or theme.get("chart_colors") or self._SAFE_FALLBACK_PALETTE) or list(self._SAFE_FALLBACK_PALETTE)
        template = theme.get("plotly_template", "plotly_white")
        paper_background = theme["surface"]
        plot_background = theme["surface"]
        if chart_style == "Minimal":
            template = "plotly_white"
        elif chart_style == "Dark":
            template = "plotly_dark"
            paper_background = theme["background"]
            plot_background = theme["background"]
        figure.update_layout(
            template=template,
            colorway=chosen_palette,
            paper_bgcolor=paper_background,
            plot_bgcolor=plot_background,
            font={"color": theme["text"]},
            title={"font": {"color": theme["text"]}},
            xaxis={
                "color": theme["text"],
                "gridcolor": theme["grid"],
                "linecolor": theme["border"],
                "zerolinecolor": theme["border"],
                "tickfont": {"color": theme["text"]},
                "title": {"font": {"color": theme["text"]}},
            },
            yaxis={
                "color": theme["text"],
                "gridcolor": theme["grid"],
                "linecolor": theme["border"],
                "zerolinecolor": theme["border"],
                "tickfont": {"color": theme["text"]},
                "title": {"font": {"color": theme["text"]}},
            },
            legend={
                "font": {"color": theme["text"]},
                "bgcolor": paper_background,
                "bordercolor": theme["border"],
                "borderwidth": 0,
            },
            hoverlabel={"bgcolor": paper_background, "font": {"color": theme["text"]}},
        )
        for index, trace in enumerate(figure.data):
            trace_type = getattr(trace, "type", None)
            series_color = chosen_palette[index % len(chosen_palette)]
            if trace_type == "heatmap":
                trace.colorscale = theme.get("heatmap_colorscale") or self._SAFE_FALLBACK_HEATMAP
                if getattr(trace, "colorbar", None) is not None:
                    trace.colorbar.tickfont = {"color": theme["text"]}
                    trace.colorbar.outlinecolor = theme["border"]
                    if getattr(trace.colorbar, "title", None) is not None:
                        trace.colorbar.title.font = {"color": theme["text"]}
            elif trace_type in {"pie", "treemap", "sunburst", "icicle", "funnelarea"} and getattr(trace, "marker", None) is not None:
                values = getattr(trace, "values", None)
                labels = getattr(trace, "labels", None)
                count = len(values) if values is not None else len(labels) if labels is not None else len(chosen_palette)
                trace.marker.colors = [chosen_palette[item % len(chosen_palette)] for item in range(count)]
                if getattr(trace.marker, "line", None) is not None:
                    trace.marker.line.color = theme["surface"]
                    trace.marker.line.width = 1
            elif trace_type in {"bar", "histogram", "box", "scatter", "funnel"} and getattr(trace, "marker", None) is not None:
                trace.marker.color = series_color
                trace.marker.line.color = theme["surface"]
                trace.marker.line.width = 0.5
            if getattr(trace, "line", None) is not None and trace_type in {"line", "scatter", "bar", "histogram", "box"}:
                trace.line.color = series_color
            if trace_type == "scatter" and getattr(trace, "fill", None) not in (None, "none"):
                trace.fillcolor = self._rgba(series_color, 0.32)
        return figure

    def apply_export_figure(self, figure, palette: list[str] | None = None, chart_style: str = "Auto", theme_name: str | None = None):
        """Re-apply the active theme immediately before HTML, PNG, or PDF rendering."""

        theme = self.active(theme_name)
        return self.apply_figure(figure, palette=palette, chart_style=chart_style, theme=theme)

    def style_dataframe(self, frame: pd.DataFrame):
        """Return a pandas Styler so Streamlit tables use the active theme colors."""

        theme = self.active()
        return (
            frame.style
            .set_properties(**{"background-color": theme["surface"], "color": theme["text"], "border-color": theme["border"]})
            .set_table_styles(
                [
                    {"selector": "th", "props": [("background-color", theme["table_header"]), ("color", theme["text"]), ("border-color", theme["border"])]},
                    {"selector": "td", "props": [("border-color", theme["border"])]},
                ]
            )
        )

    def export_settings(self) -> dict:
        """Return serializable theme settings for configuration exports."""

        theme = self.active()
        return {
            "name": st.session_state.get("dashboard_theme", "Executive"),
            "chart_colors": list(theme.get("chart_colors", [])),
            "plotly_template": theme.get("plotly_template", "plotly_white"),
        }


THEME_MANAGER = ThemeManager(THEMES)

VISUALIZATION_LIBRARY = [
    {"label": "Line Chart", "kind": "line", "category": "Trend", "keywords": "trend time series movement"},
    {"label": "Area Chart", "kind": "area", "category": "Trend", "keywords": "trend time series movement"},
    {"label": "Rolling Average", "kind": "rolling_average", "category": "Trend", "keywords": "trend smoothing moving average"},
    {"label": "Bar Chart", "kind": "bar", "category": "Comparison", "keywords": "compare ranking"},
    {"label": "Grouped Bar Chart", "kind": "grouped_bar", "category": "Comparison", "keywords": "compare group aggregation"},
    {"label": "Stacked Bar Chart", "kind": "stacked_bar", "category": "Comparison", "keywords": "compare group stacked composition"},
    {"label": "Histogram", "kind": "histogram", "category": "Distribution", "keywords": "distribution spread numeric"},
    {"label": "Boxplot", "kind": "boxplot", "category": "Distribution", "keywords": "distribution outlier spread"},
    {"label": "Scatter Plot", "kind": "scatter", "category": "Relationship", "keywords": "relationship correlation numeric"},
    {"label": "Correlation Heatmap", "kind": "heatmap", "category": "Relationship", "keywords": "relationship correlation numeric"},
    {"label": "Ranking Chart", "kind": "ranking", "category": "Ranking", "keywords": "ranking top bottom"},
    {"label": "Pie Chart", "kind": "pie", "category": "Composition", "keywords": "composition share proportion"},
    {"label": "Treemap", "kind": "treemap", "category": "Composition", "keywords": "composition hierarchy share"},
]


DATASET_SUMMARY_OPTIONS = [
    ("overview", "Dataset overview"),
    ("column_names", "Column names"),
    ("data_types", "Data types"),
    ("missing_values", "Missing value summary"),
    ("duplicate_records", "Duplicate records summary"),
    ("numeric_statistics", "Numerical statistics"),
    ("categorical_summary", "Categorical variable summary"),
    ("unique_value_counts", "Unique value counts"),
    ("data_quality", "Data quality report"),
    ("correlation_summary", "Correlation summary"),
    ("distribution_summary", "Distribution summary"),
]


KPI_AGGREGATIONS = ["Sum", "Average", "Median", "Count", "Unique Count", "Minimum", "Maximum", "Standard Deviation"]
KPI_NUMBER_FORMATS = ["Number", "Percentage", "Currency"]
KPI_ICONS = ["None", "▦", "Σ", "◉", "⌁", "✓", "★"]
KPI_THEME_COLORS = ["Auto (theme)", "Primary", "Secondary", "Accent", "Positive", "Warning"]


def _apply_theme(theme_name: str) -> None:
    """Apply the active theme to the Streamlit page."""

    THEME_MANAGER.apply_page(theme_name)


def _dashboard_chart_theme() -> dict:
    """Return the active dashboard theme used by page and export charts."""

    return THEME_MANAGER.active()


def _apply_dashboard_chart_theme(figure, palette: list[str] | None = None, chart_style: str = "Auto"):
    """Apply the centralized theme to any Plotly figure."""

    return THEME_MANAGER.apply_figure(figure, palette=palette, chart_style=chart_style)


def _chart_type_options(spec: dict) -> list[tuple[str, str]]:
    """Return safe chart-type choices appropriate for a generated chart."""

    kind = spec["kind"]
    if kind == "time_series":
        return [("Auto", "time_series"), ("Line chart", "time_series_line"), ("Area chart", "time_series_area"), ("Bar chart", "time_series_bar")]
    if kind in {"categorical_bar", "categorical_pie"}:
        return [("Auto", kind), ("Bar chart", "categorical_bar"), ("Pie chart", "categorical_pie")]
    if kind == "grouped_bar":
        return [("Auto", "grouped_bar"), ("Total comparison", "grouped_bar"), ("Average comparison", "grouped_average")]
    if kind in {"histogram", "boxplot"}:
        return [("Auto", kind), ("Histogram", "histogram"), ("Boxplot", "boxplot")]
    return [("Auto", kind), ("Scatter plot", "scatter")]


def _aggregate_series(frame: pd.DataFrame, dimension: str, metric: str | None, aggregation: str) -> pd.DataFrame:
    """Aggregate a selected dimension/metric pair for a user-added chart."""

    work = frame.copy()
    work[dimension] = work[dimension].astype("string").fillna("<missing>")
    if metric and metric in work.columns:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        operation = {"Sum": "sum", "Average": "mean", "Median": "median", "Minimum": "min", "Maximum": "max", "Standard Deviation": "std"}.get(aggregation)
        if aggregation == "Count":
            grouped = work.groupby(dimension, as_index=False)[metric].count()
        elif aggregation == "Unique Count":
            grouped = work.groupby(dimension, as_index=False)[metric].nunique()
        else:
            work = work.dropna(subset=[metric])
            grouped = work.groupby(dimension, as_index=False)[metric].agg(operation or "sum")
        return grouped.rename(columns={metric: "value"}).sort_values("value", ascending=False)
    counts = work.groupby(dimension, as_index=False).size()
    return counts.rename(columns={"size": "value"}).sort_values("value", ascending=False)


def _aggregate_grouped(frame: pd.DataFrame, group_keys: list[str], metric: str, aggregation: str) -> pd.DataFrame:
    """Aggregate a metric for one or more grouping fields without silently changing the requested operation."""

    work = frame.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    grouped_source = work.groupby(group_keys, as_index=False)[metric]
    if aggregation == "Count":
        grouped = grouped_source.count()
    elif aggregation == "Unique Count":
        grouped = grouped_source.nunique()
    else:
        operation = {"Sum": "sum", "Average": "mean", "Median": "median", "Minimum": "min", "Maximum": "max", "Standard Deviation": "std"}.get(aggregation, "sum")
        grouped = grouped_source.agg(operation)
    return grouped.rename(columns={metric: "value"})


def _apply_dashboard_filters(frame: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Apply the same persisted dashboard filters used by page and export views."""

    work = frame.copy()
    if not isinstance(filters, dict):
        return work
    for column, values in filters.items():
        if column not in work.columns or values in (None, [], ""):
            continue
        allowed = {str(value) for value in values} if isinstance(values, (list, tuple, set)) else {str(values)}
        work = work[work[column].astype("string").isin(allowed)].copy()
    return work


def _analytics_for_current_filters(analytics: dict) -> dict:
    """Recompute analytical outputs when a persisted dashboard filter is active."""

    filters = st.session_state.get("dashboard_filters", {})
    if not isinstance(filters, dict) or not any(value not in (None, [], "") for value in filters.values()):
        return analytics
    filtered_data = _apply_dashboard_filters(st.session_state.cleaned_data, filters)
    return analyze_dataset(filtered_data)


def _chart_colors(spec: dict) -> list[str]:
    """Return a reusable color palette selected in Chart Generator."""

    return THEME_MANAGER.palette(spec.get("color_theme", "Default"))


def _apply_chart_style(figure, spec: dict):
    """Apply dashboard theme plus any explicit Chart Generator styling."""

    return _apply_dashboard_chart_theme(figure, palette=_chart_colors(spec), chart_style=spec.get("chart_style", "Auto"))


def _manual_visualization_figure(spec: dict):
    """Render a visualization created through Chart Generator."""

    data = _apply_dashboard_filters(st.session_state.cleaned_data, st.session_state.get("dashboard_filters", {}))
    chart_kind = spec["kind"]
    x_column = spec.get("x_column")
    y_column = spec.get("y_column")
    group_column = spec.get("group_column") or spec.get("color_column")
    group_columns = [column for column in (spec.get("group_columns") or ([group_column] if group_column else [])) if column in data.columns]
    facet_column = spec.get("facet_column") if spec.get("facet_column") in data.columns else None
    size_column = spec.get("size_column")
    aggregation = spec.get("aggregation", "Sum")
    title = spec.get("title", spec.get("label", "Custom visualization"))
    filter_column = spec.get("filter_column")
    filter_value = spec.get("filter_value", "<all>")
    if filter_column in data.columns and filter_value != "<all>":
        data = data[data[filter_column].astype("string") == str(filter_value)].copy()
    if x_column not in data.columns and chart_kind not in {"heatmap"}:
        return None
    palette = _chart_colors(spec)

    def _add_dimension_labels(work: pd.DataFrame) -> pd.DataFrame:
        """Combine multiple selected grouping fields into one chart-safe label."""

        result = work.copy()
        if group_columns:
            result["group"] = result[group_columns].astype("string").fillna("<missing>").agg(" · ".join, axis=1)
        if facet_column:
            result["facet"] = result[facet_column].astype("string").fillna("<missing>")
        return result

    if chart_kind in {"line", "area", "rolling_average"}:
        if x_column not in data.columns or y_column not in data.columns:
            return None
        dates = pd.to_datetime(data[x_column], errors="coerce")
        values = pd.to_numeric(data[y_column], errors="coerce")
        work = pd.DataFrame({"period": dates.dt.to_period("M").dt.to_timestamp(), "value": values})
        if group_columns:
            work[group_columns] = data[group_columns].reset_index(drop=True)
        if facet_column:
            work[facet_column] = data[facet_column].reset_index(drop=True)
        work = _add_dimension_labels(work)
        work = work.dropna(subset=["period", "value"])
        if work.empty:
            return None
        group_keys = ["period"] + (["group"] if "group" in work.columns else [])
        group_keys += ["facet"] if "facet" in work.columns else []
        grouped = _aggregate_grouped(work, group_keys, "value", aggregation)
        if chart_kind == "rolling_average":
            if "group" in grouped.columns:
                grouped["value"] = grouped.groupby("group")["value"].transform(lambda series: series.rolling(3, min_periods=1).mean())
            else:
                grouped["value"] = grouped["value"].rolling(3, min_periods=1).mean()
        options = {"x": "period", "y": "value", "title": title, "color_discrete_sequence": palette}
        if "group" in grouped.columns:
            options["color"] = "group"
        if "facet" in grouped.columns:
            options["facet_col"] = "facet"
        if chart_kind == "area":
            return _apply_chart_style(px.area(grouped, **options), spec)
        return _apply_chart_style(px.line(grouped, markers=True, **options), spec)
    if chart_kind in {"bar", "grouped_bar", "stacked_bar", "ranking"}:
        work = data.copy()
        work[x_column] = work[x_column].astype("string").fillna("<missing>")
        work = _add_dimension_labels(work)
        group_keys = [x_column]
        if "group" in work.columns:
            group_keys.append("group")
        if "facet" in work.columns:
            group_keys.append("facet")
        if y_column in work.columns:
            grouped = _aggregate_grouped(work, group_keys, y_column, aggregation)
        else:
            grouped = work.groupby(group_keys, as_index=False).size().rename(columns={"size": "value"})
        options = {"x": x_column, "y": "value", "title": title, "labels": {"value": aggregation}, "color_discrete_sequence": palette}
        if "group" in grouped.columns:
            options["color"] = "group"
        if "facet" in grouped.columns:
            options["facet_col"] = "facet"
        figure = px.bar(grouped.head(80 if "group" in grouped.columns else 30), barmode="stack" if chart_kind == "stacked_bar" else "group", **options)
        if chart_kind == "ranking":
            figure.update_layout(xaxis={"categoryorder": "total descending"})
        return _apply_chart_style(figure, spec)
    if chart_kind == "pie":
        grouped = _aggregate_series(data, x_column, y_column, aggregation if y_column else "Count")
        return _apply_chart_style(px.pie(grouped.head(20), names=x_column, values="value", title=title, hole=0.35, color_discrete_sequence=palette), spec)
    if chart_kind == "treemap":
        path = list(dict.fromkeys([*group_columns, x_column]))
        if not path:
            return None
        if y_column in data.columns:
            work = data.copy()
            grouped = _aggregate_grouped(work, path, y_column, aggregation)
        else:
            grouped = data.groupby(path, as_index=False).size().rename(columns={"size": "value"})
        return _apply_chart_style(px.treemap(grouped, path=path, values="value", title=title, color_discrete_sequence=palette), spec)
    if chart_kind == "histogram":
        target = y_column or x_column
        return _apply_chart_style(px.histogram(data, x=target, nbins=30, title=title, color_discrete_sequence=palette), spec) if target in data.columns else None
    if chart_kind == "boxplot":
        target = y_column or x_column
        return _apply_chart_style(px.box(data, y=target, points="outliers", title=title, color_discrete_sequence=palette), spec) if target in data.columns else None
    if chart_kind == "scatter":
        if x_column in data.columns and y_column in data.columns:
            options = {"x": x_column, "y": y_column, "title": title, "opacity": 0.75, "color_discrete_sequence": palette}
            if group_columns:
                scatter_data = data.copy()
                scatter_data["group"] = scatter_data[group_columns].astype("string").fillna("<missing>").agg(" · ".join, axis=1)
                options["color"] = "group"
            else:
                scatter_data = data
            if facet_column:
                options["facet_col"] = facet_column
            if size_column in data.columns:
                options["size"] = size_column
            return _apply_chart_style(px.scatter(scatter_data, **options), spec)
        return None
    if chart_kind == "heatmap":
        selected_heatmap = [column for column in spec.get("heatmap_columns", []) if column in data.columns]
        numeric = data[selected_heatmap] if selected_heatmap else data.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return None
        corr = numeric.corr()
        figure = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale=THEME_MANAGER.heatmap_colorscale(), zmin=-1, zmax=1))
        figure.update_layout(title=title)
        return _apply_chart_style(figure, spec)
    return None


def _render_custom_visualization(spec: dict, chart_key: str) -> None:
    """Render a user-added visualization."""

    figure = _manual_visualization_figure(spec)
    if figure is not None:
        figure.update_layout(margin={"l": 20, "r": 20, "t": 55, "b": 20})
        st.plotly_chart(figure, use_container_width=True, key=chart_key)


def _chart_field_label(profile: dict, column: str) -> str:
    """Show inferred type beside a raw field name without changing its value."""

    if str(column).startswith("<"):
        return str(column)
    type_name = profile.get("type_by_column", {}).get(column, "field")
    return f"{column}  ·  {type_name}"


def _chart_selectbox(label: str, options: list[str], profile: dict, key: str, default: str | None = None, help_text: str | None = None) -> str | None:
    """Render a searchable, type-labelled selectbox with a valid remembered value."""

    options = list(dict.fromkeys(options))
    if not options:
        return None
    remembered = st.session_state.get(key, default)
    if remembered not in options:
        st.session_state.pop(key, None)
        remembered = default if default in options else options[0]
    return st.selectbox(
        label,
        options,
        index=options.index(remembered) if remembered in options else 0,
        format_func=lambda item: _chart_field_label(profile, item),
        key=key,
        help=help_text or "Type in this box to search all available fields.",
    )


def _chart_multiselect(label: str, options: list[str], profile: dict, key: str, default: list[str] | None = None, help_text: str | None = None) -> list[str]:
    """Render a searchable multi-select while discarding stale field names safely."""

    options = list(dict.fromkeys(options))
    remembered = st.session_state.get(key, default or [])
    if not isinstance(remembered, list) or any(item not in options for item in remembered):
        st.session_state.pop(key, None)
        remembered = [item for item in (default or []) if item in options]
    return st.multiselect(
        label,
        options,
        default=remembered,
        format_func=lambda item: _chart_field_label(profile, item),
        key=key,
        help=help_text or "Type to search, then select one or more fields.",
    )


def _chart_option_selectbox(label: str, options: list[str], key: str, default: str | None = None) -> str:
    """Render a remembered non-column option without retaining an invalid value."""

    options = list(dict.fromkeys(options))
    remembered = st.session_state.get(key, default)
    if remembered not in options:
        st.session_state.pop(key, None)
        remembered = default if default in options else options[0]
    return st.selectbox(label, options, index=options.index(remembered), key=key)


def _render_add_visualization(analytics: dict) -> None:
    """Build a flexible chart-generator draft from every compatible dataset field."""

    profile = analytics["profile"]
    numeric = list(profile.get("numerical_columns", []))
    dates = list(profile.get("datetime_columns", []))
    all_columns = list(profile.get("columns", []))
    type_by_column = profile.get("type_by_column", {})
    dimension_columns = [
        column for column in all_columns
        if type_by_column.get(column) in {"categorical", "boolean", "datetime", "text"}
    ]
    memory = st.session_state.get("chart_generator_last_selections", {})

    query = st.text_input("Search chart types", placeholder="Try: trend, distribution, relationship, ranking", key="chart_generator_search")
    category = st.selectbox("Visualization category", ["All", "Trend", "Distribution", "Comparison", "Relationship", "Ranking", "Composition"], key="chart_generator_category")
    eligible: list[dict] = []
    for item in VISUALIZATION_LIBRARY:
        searchable = f"{item['label']} {item['category']} {item['keywords']}".lower()
        query_match = not query.strip() or query.strip().lower() in searchable
        category_match = category == "All" or item["category"] == category
        kind = item["kind"]
        requires_data = (
            (kind in {"line", "area", "rolling_average"} and dates and numeric)
            or (kind in {"histogram", "boxplot"} and numeric)
            or (kind in {"scatter", "heatmap"} and len(numeric) >= 2)
            or (kind in {"bar", "ranking", "grouped_bar", "stacked_bar"} and all_columns)
            or (kind in {"pie", "treemap"} and all_columns)
        )
        if query_match and category_match and requires_data:
            eligible.append(item)
    if not eligible:
        st.info("No compatible chart type was found for the search and available columns. Try another search or category.")
        return

    chart_type_options = [item["label"] for item in eligible]
    if st.session_state.get("chart_generator_recommendation") not in chart_type_options:
        st.session_state.pop("chart_generator_recommendation", None)
    selected_label = st.selectbox("Chart type", chart_type_options, key="chart_generator_recommendation")
    selected = next(item for item in eligible if item["label"] == selected_label)
    st.caption(f"This chart is compatible with the detected {selected['category'].lower()} fields. All eligible dataset columns are available below.")
    kind = selected["kind"]
    prior = memory.get(kind, {}) if isinstance(memory.get(kind, {}), dict) else {}

    if kind in {"line", "area", "rolling_average"}:
        x_options, y_options = dates, numeric
    elif kind in {"bar", "grouped_bar", "stacked_bar", "ranking", "pie", "treemap"}:
        x_options = all_columns
        y_options = ["<count rows>", *numeric] if kind in {"bar", "grouped_bar", "stacked_bar", "treemap", "pie"} else ["<count rows>", *numeric]
    elif kind in {"histogram", "boxplot"}:
        x_options, y_options = numeric, ["<not required>"]
    elif kind == "scatter":
        x_options, y_options = numeric, numeric
    else:
        x_options, y_options = [], ["<not required>"]

    if kind == "heatmap":
        st.info("Heatmap uses numerical columns only. Select the fields to include in the correlation matrix.")
        heatmap_columns = _chart_multiselect("Heatmap numerical columns", numeric, profile, f"chart_generator_heatmap_{kind}", prior.get("heatmap_columns", numeric))
        x_column = y_column = None
    else:
        if not x_options:
            st.info(f"This chart requires a compatible X-axis field, but no eligible column was detected. Available types: {', '.join(sorted(set(type_by_column.values())))}.")
            return
        if not y_options:
            st.info("This chart requires a compatible numerical Y-axis field, but none is available in this dataset.")
            return
        x_column = _chart_selectbox("X-axis / category column", x_options, profile, f"chart_generator_x_{kind}", prior.get("x_column"))
        y_choice = _chart_selectbox("Y-axis / metric column", y_options, profile, f"chart_generator_y_{kind}", prior.get("y_column") or ("<not required>" if "<not required>" in y_options else "<count rows>"))
        y_column = None if not y_choice or y_choice.startswith("<") else y_choice

    aggregation_options = ["Count"] if y_column is None else ["Sum", "Average", "Median", "Count", "Unique Count", "Minimum", "Maximum", "Standard Deviation"]
    aggregation = _chart_option_selectbox("Aggregation", aggregation_options, f"chart_generator_aggregation_{kind}", prior.get("aggregation"))
    title = st.text_input("Chart title", value=prior.get("title", selected["label"]), key=f"chart_generator_title_{kind}")

    group_columns: list[str] = []
    facet_column = None
    size_column = None
    filter_choice = "<none>"
    filter_value = "<all>"
    with st.expander("Optional grouping, faceting, filters, and style"):
        available_dimensions = [column for column in dimension_columns if column not in {x_column, y_column}]
        if kind in {"line", "area", "rolling_average", "bar", "grouped_bar", "stacked_bar", "ranking", "scatter", "treemap"}:
            group_columns = _chart_multiselect("Group / color by", available_dimensions, profile, f"chart_generator_groups_{kind}", prior.get("group_columns", []), "Select multiple categorical, datetime, boolean, or text fields to combine into a series.")
            facet_options = ["<none>"] + [column for column in available_dimensions if column not in group_columns]
            facet_choice = _chart_selectbox("Facet by", facet_options, profile, f"chart_generator_facet_{kind}", prior.get("facet_column") or "<none>", "Split the chart into small panels by this field when supported.")
            facet_column = None if not facet_choice or facet_choice == "<none>" else facet_choice
        if kind == "scatter":
            size_options = ["<none>"] + numeric
            size_choice = _chart_selectbox("Marker size", size_options, profile, f"chart_generator_size_{kind}", prior.get("size_column") or "<none>")
            size_column = None if not size_choice or size_choice == "<none>" else size_choice
        filter_choice = _chart_selectbox("Filter column", ["<none>", *all_columns], profile, f"chart_generator_filter_{kind}", prior.get("filter_column") or "<none>")
        if filter_choice and filter_choice != "<none>":
            filter_values = sorted(st.session_state.cleaned_data[filter_choice].dropna().astype("string").unique().tolist())
            filter_value = _chart_option_selectbox("Filter value", ["<all>", *filter_values[:500]], f"chart_generator_filter_value_{kind}", prior.get("filter_value", "<all>"))
        color_theme = _chart_option_selectbox("Color theme", ["Default", "Set2", "Blues", "Viridis"], f"chart_generator_color_theme_{kind}", prior.get("color_theme", "Default"))
        chart_style = _chart_option_selectbox("Chart style", ["Auto", "Minimal", "Dark"], f"chart_generator_style_{kind}", prior.get("chart_style", "Auto"))

    preview_rows = []
    for role, column in [("X-axis", x_column), ("Y-axis", y_column), ("Facet", facet_column), ("Filter", filter_choice if filter_choice != "<none>" else None), ("Size", size_column)]:
        if column:
            preview_rows.append({"Role": role, "Column": column, "Detected type": type_by_column.get(column, "field")})
    for column in group_columns:
        preview_rows.append({"Role": "Group / color", "Column": column, "Detected type": type_by_column.get(column, "field")})
    if kind == "heatmap":
        for column in heatmap_columns:
            preview_rows.append({"Role": "Heatmap", "Column": column, "Detected type": type_by_column.get(column, "field")})
    with st.expander("Preview selected fields", expanded=True):
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
        sample_columns = list(dict.fromkeys([row["Column"] for row in preview_rows if row["Column"] in all_columns]))
        if sample_columns:
            st.caption("Sample values from the selected fields")
            st.dataframe(THEME_MANAGER.style_dataframe(st.session_state.cleaned_data[sample_columns].head(5)), use_container_width=True, hide_index=True)

    if st.button("Generate chart", type="primary", key="generate_chart_button"):
        draft = {
            "id": "chart_generator_preview",
            "label": selected["label"],
            "kind": kind,
            "category": selected["category"],
            "x_column": x_column,
            "y_column": y_column,
            "heatmap_columns": heatmap_columns if kind == "heatmap" else [],
            "aggregation": aggregation,
            "title": title,
            "group_columns": group_columns,
            "group_column": group_columns[0] if group_columns else None,
            "color_column": group_columns[0] if group_columns else None,
            "facet_column": facet_column,
            "size_column": size_column,
            "filter_column": None if filter_choice == "<none>" else filter_choice,
            "filter_value": filter_value,
            "color_theme": color_theme,
            "chart_style": chart_style,
        }
        st.session_state.chart_generator_draft = draft
        last_selections = dict(st.session_state.get("chart_generator_last_selections", {}))
        last_selections[kind] = {key: draft.get(key) for key in ["x_column", "y_column", "heatmap_columns", "aggregation", "title", "group_columns", "facet_column", "size_column", "filter_column", "filter_value", "color_theme", "chart_style"]}
        st.session_state.chart_generator_last_selections = last_selections


def _render_chart_generator(analytics: dict) -> None:
    """Render the optional chart-builder section inside the dashboard."""

    with st.container(border=True):
        st.caption("Create additional visualizations when the automatic dashboard does not cover your question.")
        steps = st.columns(5)
        for column, label in zip(steps, ["1. Chart type", "2. Data columns", "3. Settings", "4. Generate", "5. Preview"]):
            column.markdown(f"**{label}**")

        with st.container(border=True):
            _render_add_visualization(analytics)

        draft = st.session_state.chart_generator_draft
        if draft:
            st.subheader("Preview result")
            _render_custom_visualization(draft, "chart_generator_preview")
            save_col, add_col = st.columns(2)
            with save_col:
                if st.button("Save chart", key="save_generated_chart"):
                    saved = {**draft, "id": f"saved_{len(st.session_state.saved_charts) + 1}"}
                    st.session_state.saved_charts.append(saved)
                    st.success(f"Saved: {draft['title']}")
            with add_col:
                if st.button("Add to Customize Dashboard", key="add_generated_chart_to_dashboard"):
                    custom_id = f"custom_{len(st.session_state.custom_visualizations) + 1}"
                    st.session_state.custom_visualizations.append({**draft, "id": custom_id})
                    selected_generated = st.session_state.selected_generated_chart_ids
                    if selected_generated is None:
                        st.session_state.selected_generated_chart_ids = [item["id"] for item in st.session_state.custom_visualizations]
                    else:
                        st.session_state.selected_generated_chart_ids = [*selected_generated, custom_id]
                    st.success(f"Added to Customize Dashboard: {draft['title']}")

        if st.session_state.saved_charts:
            st.subheader("Saved charts")
            for index, spec in enumerate(st.session_state.saved_charts):
                with st.container(border=True):
                    st.markdown(f"**{spec['title']}** · {spec['label']}")
                    _render_custom_visualization(spec, f"saved_chart_{index}")
                    if st.button("Delete saved chart", key=f"delete_saved_chart_{index}"):
                        st.session_state.saved_charts.pop(index)
                        st.rerun()


def _dashboard_chart_component_id(kind: str, chart_id: str) -> str:
    """Create a stable layout ID for an automatic or generated chart."""

    return f"chart::{kind}::{chart_id}"


def _dashboard_component_specs(analytics: dict) -> list[dict]:
    """Return every final-dashboard component as one reorderable layout item."""

    components: list[dict] = []
    kpi_by_id = {spec["id"]: spec for spec in _all_kpi_specs(analytics)}
    for kpi_id in _ordered_kpi_ids():
        spec = kpi_by_id.get(kpi_id)
        if spec:
            title = st.session_state.get("kpi_overrides", {}).get(kpi_id, {}).get("title", spec.get("title", spec.get("label", "KPI")))
            components.append({"id": f"kpi::{kpi_id}", "type": "kpi", "label": str(title), "kpi_id": kpi_id})

    summary_labels = dict(DATASET_SUMMARY_OPTIONS)
    for section_id in _ordered_dataset_summary_ids():
        components.append({"id": f"summary::{section_id}", "type": "summary", "label": summary_labels.get(section_id, section_id), "section_id": section_id})

    for kind, spec in _selected_dashboard_items(analytics):
        components.append({
            "id": _dashboard_chart_component_id(kind, spec["id"]),
            "type": "chart",
            "label": spec.get("title", spec.get("label", "Chart")),
            "chart_kind": kind,
            "chart_id": spec["id"],
            "spec": spec,
        })

    for index, section in enumerate(st.session_state.get("custom_sections", [])):
        components.append({
            "id": f"section::{index}",
            "type": "section",
            "label": section.get("name", f"Custom section {index + 1}"),
            "section": section,
        })

    if st.session_state.get("story_text", "").strip():
        components.append({"id": "text::story", "type": "text", "label": "Storytelling notes"})
    if analytics.get("insights") or analytics.get("recommendations"):
        components.append({"id": "insights::summary", "type": "insights", "label": "AI insights and recommendations"})
    components.append({"id": "quality::report", "type": "quality", "label": "Data quality report"})
    return components


def _dashboard_column_count(layout: dict | None = None) -> int:
    """Return a safe user-selected dashboard column count."""

    layout = layout if isinstance(layout, dict) else st.session_state.get("dashboard_layout", {})
    layout = layout if isinstance(layout, dict) else {}
    try:
        return max(MIN_DASHBOARD_COLUMNS, min(int(layout.get("columns", DEFAULT_DASHBOARD_COLUMNS)), MAX_DASHBOARD_COLUMNS))
    except (TypeError, ValueError):
        return DEFAULT_DASHBOARD_COLUMNS


def _default_dashboard_lanes(component_ids: list[str], columns: int) -> list[list[str]]:
    """Return an unbalanced canvas so new components never displace existing ones."""

    # A new dashboard deliberately starts with all components in the first lane.
    # This avoids silently making placement decisions for the user; they can move
    # every item to a different column with the layout canvas.
    return [list(component_ids), *([[] for _ in range(max(1, columns) - 1)])]


def _dashboard_layout_lanes(analytics: dict) -> list[list[dict]]:
    """Return the exact saved drag-and-drop layout without automatic packing.

    Existing components never move lanes automatically.  Components selected
    after a user has arranged the canvas are appended to the bottom of the first
    lane, which gives them a predictable starting place without disturbing the
    user's layout.
    """

    components = _dashboard_component_specs(analytics)
    component_ids = [component["id"] for component in components]
    by_id = {component["id"]: component for component in components}
    stored_layout = st.session_state.get("dashboard_layout", {})
    stored_layout = stored_layout if isinstance(stored_layout, dict) else {}
    columns = _dashboard_column_count(stored_layout)
    raw_lanes = stored_layout.get("lanes", [])
    raw_lanes = raw_lanes if isinstance(raw_lanes, list) else []
    all_lanes = [list(lane) if isinstance(lane, list) else [] for lane in raw_lanes[:MAX_DASHBOARD_COLUMNS]]
    # A malformed/legacy configuration must not silently discard components in a
    # populated lane. Show that lane instead of migrating its components.
    last_populated_lane = max((index + 1 for index, lane in enumerate(all_lanes) if lane), default=0)
    columns = max(columns, last_populated_lane)
    lanes = all_lanes[:columns]
    customized = bool(stored_layout.get("customized", False))

    if not customized or not lanes:
        lanes = _default_dashboard_lanes(component_ids, columns)
    else:
        while len(lanes) < columns:
            lanes.append([])

        seen: set[str] = set()
        reconciled_lanes: list[list[str]] = []
        for lane in lanes:
            reconciled_lane = []
            for component_id in lane:
                if component_id in by_id and component_id not in seen:
                    reconciled_lane.append(component_id)
                    seen.add(component_id)
            reconciled_lanes.append(reconciled_lane)
        for component_id in component_ids:
            if component_id not in seen:
                reconciled_lanes[0].append(component_id)
        lanes = reconciled_lanes

    normalized_layout = {"columns": columns, "lanes": lanes, "customized": customized}
    if stored_layout != normalized_layout:
        st.session_state.dashboard_layout = normalized_layout
    return [[by_id[component_id] for component_id in lane if component_id in by_id] for lane in lanes]


def _dashboard_layout_label(component: dict, duplicate_count: int = 0) -> str:
    """Create a readable sortable label with an explicit drag-handle glyph."""

    icon = {"kpi": "▣", "chart": "▥", "summary": "▤", "insights": "✦", "text": "T", "section": "§", "quality": "✓"}.get(component["type"], "⋮")
    suffix = f" · {component['id']}" if duplicate_count else ""
    return f"↕  {icon}  {component['label']}{suffix}"


def _change_dashboard_columns() -> None:
    """Resize the empty canvas only; never relocate components between columns."""

    try:
        requested_columns = int(st.session_state.get("dashboard_columns_selector", DEFAULT_DASHBOARD_COLUMNS))
    except (TypeError, ValueError):
        requested_columns = DEFAULT_DASHBOARD_COLUMNS
    requested_columns = max(MIN_DASHBOARD_COLUMNS, min(requested_columns, MAX_DASHBOARD_COLUMNS))

    layout = st.session_state.get("dashboard_layout", {})
    layout = layout if isinstance(layout, dict) else {}
    current_columns = _dashboard_column_count(layout)
    raw_lanes = layout.get("lanes", [])
    raw_lanes = raw_lanes if isinstance(raw_lanes, list) else []
    lanes = [list(lane) if isinstance(lane, list) else [] for lane in raw_lanes[:current_columns]]
    while len(lanes) < current_columns:
        lanes.append([])

    if requested_columns < current_columns and any(lanes[requested_columns:]):
        st.session_state.dashboard_columns_selector = current_columns
        st.session_state.dashboard_column_notice = "Move every component out of the columns you want to remove before reducing the column count."
        return

    st.session_state.dashboard_layout = {
        "columns": requested_columns,
        "lanes": lanes[:requested_columns] + ([[] for _ in range(max(0, requested_columns - len(lanes)))]),
        "customized": bool(layout.get("customized", False)),
    }
    st.session_state.dashboard_column_notice = None


def _render_dashboard_layout_editor(analytics: dict) -> None:
    """Render a visual multi-column drag-and-drop canvas for the final dashboard."""

    component_lanes = _dashboard_layout_lanes(analytics)
    components = [component for lane in component_lanes for component in lane]
    if not components:
        st.info("Select KPI cards, charts, summaries, or text blocks to arrange the final dashboard.")
        return

    label_to_id: dict[str, str] = {}
    labels_by_id: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    for component in components:
        base = _dashboard_layout_label(component)
        label_counts[base] = label_counts.get(base, 0) + 1
    for component in components:
        base = _dashboard_layout_label(component)
        label = _dashboard_layout_label(component, label_counts[base] > 1)
        label_to_id[label] = component["id"]
        labels_by_id[component["id"]] = label

    with st.expander("Layout canvas", expanded=True):
        st.caption("Drop above or below an item to keep it in that column, or drop inside another column to move it there. Columns never rebalance automatically.")
        st.caption("Click and hold the ⠿ handle, then drop a component in any column. This canvas updates the dashboard preview and every export immediately.")
        if sort_items is not None:
            theme = THEME_MANAGER.active()
            canvas_lanes = [
                {"header": f"Column {index + 1}", "items": [labels_by_id[component["id"]] for component in lane]}
                for index, lane in enumerate(component_lanes)
            ]
            custom_style = f"""
            .sortable-component.vertical {{ display: grid; grid-template-columns: repeat({len(component_lanes)}, minmax(0, 1fr)); gap: 12px; background: {theme['background']}; border: 1px solid {theme['border']}; border-radius: 14px; padding: 10px; }}
            .sortable-component.vertical .sortable-container {{ min-width: 0; height: auto; margin: 0; padding: 0; background: {theme['surface']}; border: 1px solid {theme['border']}; border-radius: 10px; overflow: visible; }}
            .sortable-container-header {{ background: {theme['table_header']}; color: {theme['muted']}; font-size: 12px; font-weight: 700; letter-spacing: .04em; padding: 9px 12px; text-transform: uppercase; }}
            .sortable-container-body {{ height: auto; min-height: 132px; padding: 7px; overflow: visible; }}
            .sortable-container-body:has(.sortable-ghost) {{ outline: 2px dashed {theme['accent']}; outline-offset: -3px; background: {theme['accent']}12; }}
            .sortable-item, .sortable-item:hover {{ min-width: 0; overflow-wrap: anywhere; white-space: normal; background: {theme['surface']}; color: {theme['text']}; border: 1px solid {theme['border']}; border-radius: 9px; box-shadow: 0 1px 2px rgba(15, 23, 42, .06); margin: 6px 0; padding: 11px 12px; cursor: grab; }}
            .sortable-item:active {{ cursor: grabbing; }}
            .sortable-item.dragging, .sortable-ghost {{ background: {theme['accent']}22; border: 1px dashed {theme['accent']}; box-shadow: 0 8px 18px {theme['accent']}33; opacity: .78; }}
            @media (max-width: 560px) {{ .sortable-component.vertical {{ grid-template-columns: 1fr; }} }}
            """
            canvas_key = hashlib.sha1(json.dumps(canvas_lanes, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
            sorted_canvas = sort_items(canvas_lanes, multi_containers=True, direction="vertical", custom_style=custom_style, key=f"dashboard_layout_canvas_{canvas_key}")
            sorted_lanes = [
                [label_to_id[label] for label in lane.get("items", []) if label in label_to_id]
                for lane in sorted_canvas
                if isinstance(lane, dict)
            ]
            if len(sorted_lanes) == len(component_lanes) and sorted_lanes != [[component["id"] for component in lane] for lane in component_lanes]:
                st.session_state.dashboard_layout = {"columns": len(component_lanes), "lanes": sorted_lanes, "customized": True}
                st.rerun()
        else:
            st.error("The visual drag-and-drop canvas is unavailable because streamlit-sortables is not installed. Install requirements.txt and restart Streamlit.")
        if st.button("Reset layout", key="reset_dashboard_layout"):
            st.session_state.dashboard_layout = {
                "columns": len(component_lanes),
                "lanes": _default_dashboard_lanes([component["id"] for component in components], len(component_lanes)),
                "customized": False,
            }
            st.rerun()


def _summary_context(analytics: dict) -> tuple[pd.DataFrame, dict]:
    """Return the current filtered data and matching analytics for summary sections."""

    effective_analytics = _analytics_for_current_filters(analytics)
    frame = _apply_dashboard_filters(st.session_state.cleaned_data, st.session_state.get("dashboard_filters", {}))
    return frame, effective_analytics


def _dataset_summary_payload(section_id: str, analytics: dict) -> dict:
    """Build one dataset summary section from the available, inferred dataset outputs."""

    frame, effective_analytics = _summary_context(analytics)
    profile = effective_analytics.get("profile", {})
    quality = effective_analytics.get("data_quality", {})
    if section_id == "overview":
        size_mb = frame.memory_usage(deep=True).sum() / (1024 * 1024)
        return {"kind": "metrics", "items": [("Rows", f"{len(frame):,}"), ("Columns", f"{len(frame.columns):,}"), ("In-memory size", f"{size_mb:.2f} MB")]}
    if section_id == "column_names":
        return {"kind": "table", "frame": pd.DataFrame({"Column": profile.get("columns", [])})}
    if section_id == "data_types":
        return {"kind": "table", "frame": pd.DataFrame([{"Column": column, "Detected type": kind} for column, kind in profile.get("type_by_column", {}).items()])}
    if section_id == "missing_values":
        missing = quality.get("missing_values", {})
        if not missing:
            return {"kind": "notice", "message": "No missing values were detected in the available data."}
        return {"kind": "table", "frame": pd.DataFrame([{"Column": column, "Missing values": count, "Missing %": round(count / max(len(frame), 1) * 100, 2)} for column, count in missing.items()])}
    if section_id == "duplicate_records":
        return {
            "kind": "metrics",
            "items": [
                ("Rows received", f"{quality.get('rows_received', len(frame)):,}"),
                ("Rows after cleaning", f"{quality.get('rows_after_cleaning', len(frame)):,}"),
                ("Duplicate rows", f"{quality.get('duplicate_rows_received', 0):,}"),
                ("Rows removed", f"{quality.get('rows_removed', 0):,}"),
            ],
        }
    if section_id == "numeric_statistics":
        numeric = effective_analytics.get("numeric_summary", [])
        if not numeric:
            return {"kind": "notice", "message": "No numerical columns are available for statistical summaries."}
        return {"kind": "table", "frame": pd.DataFrame([{"Column": item["column"], "Mean": item["mean"], "Median": item["median"], "Min": item["min"], "Max": item["max"], "Std dev": item["std"]} for item in numeric])}
    if section_id == "categorical_summary":
        records = []
        for item in effective_analytics.get("categorical_distributions", []):
            for value in item.get("top_values", []):
                records.append({"Column": item["column"], "Value": value.get("value"), "Count": value.get("count"), "Share %": value.get("share_pct")})
        if not records:
            return {"kind": "notice", "message": "No categorical columns are available for frequency summaries."}
        return {"kind": "table", "frame": pd.DataFrame(records)}
    if section_id == "unique_value_counts":
        return {"kind": "table", "frame": pd.DataFrame([{"Column": column, "Unique values": int(frame[column].nunique(dropna=True))} for column in frame.columns])}
    if section_id == "data_quality":
        available = [{"Analysis": name, "Status": "Available", "Reason": ""} for name in effective_analytics.get("available_analyses", [])]
        unavailable = [{"Analysis": name, "Status": "Skipped", "Reason": reason} for name, reason in effective_analytics.get("unavailable_analyses", {}).items()]
        records = available + unavailable
        if quality.get("validation_warnings"):
            records.append({"Analysis": "Validation warnings", "Status": "Review", "Reason": " ".join(quality["validation_warnings"])})
        return {"kind": "table", "frame": pd.DataFrame(records, columns=["Analysis", "Status", "Reason"])}
    if section_id == "correlation_summary":
        pairs = effective_analytics.get("correlations", {}).get("strongest_pairs", [])
        if not pairs:
            return {"kind": "notice", "message": "At least two usable numerical columns are required for correlation analysis."}
        return {"kind": "table", "frame": pd.DataFrame([{"Column 1": item["column_1"], "Column 2": item["column_2"], "Correlation": item["correlation"]} for item in pairs])}
    if section_id == "distribution_summary":
        numeric = effective_analytics.get("numeric_summary", [])
        if not numeric:
            return {"kind": "notice", "message": "No numerical columns are available for distribution summaries."}
        anomaly_map = {item["column"]: item.get("outlier_count", 0) for item in effective_analytics.get("anomalies", [])}
        return {"kind": "table", "frame": pd.DataFrame([{"Column": item["column"], "Min": item["min"], "Q1": item["q1"], "Median": item["median"], "Q3": item["q3"], "Max": item["max"], "Outliers": anomaly_map.get(item["column"], 0)} for item in numeric])}
    return {"kind": "notice", "message": "This summary section is unavailable for the current dataset."}


def _recommended_dataset_summary_ids(analytics: dict) -> list[str]:
    """Choose preview sections from detected data, without selecting them for Customize Dashboard."""

    profile = analytics.get("profile", {})
    recommendations = ["overview", "column_names", "data_types", "missing_values", "duplicate_records", "unique_value_counts", "data_quality"]
    if analytics.get("numeric_summary"):
        recommendations.extend(["numeric_statistics", "distribution_summary"])
    if analytics.get("categorical_distributions"):
        recommendations.append("categorical_summary")
    if analytics.get("correlations", {}).get("strongest_pairs"):
        recommendations.append("correlation_summary")
    allowed = {section_id for section_id, _ in DATASET_SUMMARY_OPTIONS}
    return [section_id for section_id, _ in DATASET_SUMMARY_OPTIONS if section_id in set(recommendations) and section_id in allowed]


def _ordered_dataset_summary_ids() -> list[str]:
    """Return selected summary IDs; the canvas owns their visible placement."""

    return list(dict.fromkeys(st.session_state.get("customize_summary_section_ids", []) or []))


def _render_dataset_summary(analytics: dict, selected_ids: list[str], heading: str) -> None:
    """Render selected or preview dataset summary sections with themed tables and cards."""

    if not selected_ids:
        return
    labels = dict(DATASET_SUMMARY_OPTIONS)
    st.subheader(heading)
    for section_id in selected_ids:
        payload = _dataset_summary_payload(section_id, analytics)
        st.markdown(f"#### {labels.get(section_id, section_id)}")
        if payload["kind"] == "metrics":
            columns = st.columns(min(len(payload["items"]), 4))
            for index, (label, value) in enumerate(payload["items"]):
                columns[index % len(columns)].metric(label, value)
        elif payload["kind"] == "table":
            st.dataframe(THEME_MANAGER.style_dataframe(payload["frame"]), use_container_width=True, hide_index=True)
        else:
            st.markdown(f'<div class="dw-section">{escape(payload["message"])}</div>', unsafe_allow_html=True)


def _render_dataset_summary_component(analytics: dict, section_id: str) -> None:
    """Render one summary component so it can participate in the global layout."""

    labels = dict(DATASET_SUMMARY_OPTIONS)
    payload = _dataset_summary_payload(section_id, analytics)
    st.subheader(labels.get(section_id, section_id))
    if payload["kind"] == "metrics":
        columns = st.columns(min(len(payload["items"]), 4))
        for index, (label, value) in enumerate(payload["items"]):
            columns[index % len(columns)].metric(label, value)
    elif payload["kind"] == "table":
        st.dataframe(THEME_MANAGER.style_dataframe(payload["frame"]), use_container_width=True, hide_index=True)
    else:
        st.markdown(f'<div class="dw-section">{escape(payload["message"])}</div>', unsafe_allow_html=True)


def _render_insights_component(analytics: dict) -> None:
    """Render insights and recommendations as one reorderable component."""

    st.subheader("AI insights and recommendations")
    insights = analytics.get("insights", [])
    if insights:
        st.markdown("#### Key insights")
        for insight in insights:
            st.info(f"**{insight['title']}** — {insight['message']}")
    else:
        st.info("No additional insights were generated from the available fields.")
    recommendations = analytics.get("recommendations", [])
    if recommendations:
        st.markdown("#### Recommendations")
        for recommendation in recommendations:
            st.info(f"**{recommendation['title']}** — {recommendation['message']}")


def _render_dashboard_component(component: dict, analytics: dict, index: int) -> None:
    """Render one component in the user's saved order."""

    with st.container(border=True):
        st.markdown(f'<div class="dw-layout-handle">{escape(str(component["label"]))}</div>', unsafe_allow_html=True)
        component_type = component["type"]
        if component_type == "kpi":
            kpi_by_id = {spec["id"]: spec for spec in _all_kpi_specs(analytics)}
            spec = kpi_by_id.get(component["kpi_id"])
            if spec:
                merged = {**spec, **st.session_state.get("kpi_overrides", {}).get(component["kpi_id"], {})}
                st.markdown(_kpi_card_html(merged, _kpi_display_data(merged, _analytics_for_current_filters(analytics))), unsafe_allow_html=True)
        elif component_type == "summary":
            _render_dataset_summary_component(analytics, component["section_id"])
        elif component_type == "chart":
            if component["chart_kind"] == "auto":
                _render_recommended_chart(component["spec"], analytics, f"layout_auto_{index}", st.session_state.chart_overrides.get(component["chart_id"]))
            else:
                _render_custom_visualization(component["spec"], f"layout_generated_{index}")
        elif component_type == "section":
            section = component["section"]
            recommendations = {item["id"]: item for item in analytics.get("visualization_recommendations", [])}
            spec = recommendations.get(section.get("chart_id"))
            if spec:
                st.subheader(section.get("name", "Custom section"))
                _render_recommended_chart(spec, analytics, f"layout_section_{index}", st.session_state.chart_overrides.get(spec["id"]))
        elif component_type == "text":
            st.subheader("Storytelling notes")
            st.markdown(f'<div class="dw-section">{escape(st.session_state.get("story_text", "")).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        elif component_type == "insights":
            _render_insights_component(analytics)
        elif component_type == "quality":
            _render_quality(analytics)


def _render_ordered_dashboard_components(analytics: dict) -> None:
    """Render the final dashboard using the persisted drag-and-drop lanes."""

    component_lanes = _dashboard_layout_lanes(analytics)
    if not any(component_lanes):
        st.info("No dashboard components are selected yet.")
        return
    columns = st.columns(len(component_lanes))
    for lane_index, lane in enumerate(component_lanes):
        with columns[lane_index]:
            for component_index, component in enumerate(lane):
                _render_dashboard_component(component, analytics, lane_index * 100 + component_index)


def _dataset_summary_html(analytics: dict, selected_ids: list[str], theme: dict) -> str:
    """Render selected summary sections to theme-aware HTML for dashboard export."""

    labels = dict(DATASET_SUMMARY_OPTIONS)
    sections = []
    for section_id in selected_ids:
        payload = _dataset_summary_payload(section_id, analytics)
        title = escape(labels.get(section_id, section_id))
        if payload["kind"] == "metrics":
            body = "".join(f'<div class="summary-metric"><div class="summary-label">{escape(str(label))}</div><div class="summary-value">{escape(str(value))}</div></div>' for label, value in payload["items"])
            content = f'<div class="summary-metrics">{body}</div>'
        elif payload["kind"] == "table":
            content = f'<div class="dw-table-scroll">{payload["frame"].to_html(index=False, escape=True, classes="summary-table", border=0)}</div>'
        else:
            content = f'<div class="summary-notice">{escape(payload["message"])}</div>'
        sections.append(f'<section class="summary-section"><h3>{title}</h3>{content}</section>')
    return "".join(sections)


def _render_customize_panel(analytics: dict, show_toggle: bool = True) -> None:
    """Expose guided customization for the final storytelling dashboard."""

    if show_toggle:
        button_label = "Close customization" if st.session_state.edit_mode else "Customize Dashboard"
        if st.button(button_label, type="secondary", key="customize_dashboard_button"):
            st.session_state.edit_mode = not st.session_state.edit_mode

        if not st.session_state.edit_mode:
            return
    else:
        st.session_state.edit_mode = True

    recommendations = sorted(analytics.get("visualization_recommendations", []), key=lambda item: item.get("priority", 99))
    chart_ids = [item["id"] for item in recommendations]
    labels = {item["id"]: item["title"] for item in recommendations}

    with st.container(border=True):
        st.markdown("### Dashboard composition")
        st.caption("Auto Dashboard is a preview. Nothing is added here until you explicitly select it.")
        st.session_state.dashboard_title = st.text_input("Dashboard title", value=st.session_state.dashboard_title or analytics.get("dashboard_metadata", {}).get("title", "DataWonder Executive Dashboard"), key="dashboard_title_input")
        st.session_state.dashboard_subtitle = st.text_input("Dashboard subtitle", value=st.session_state.dashboard_subtitle or analytics.get("dashboard_metadata", {}).get("subtitle", ""), key="dashboard_subtitle_input")
        st.session_state.dashboard_theme = st.selectbox("Theme", list(THEMES), index=list(THEMES).index(st.session_state.dashboard_theme), key="theme_selector")
        st.caption("The selected theme controls dashboard surfaces, chart palettes, and exported chart colors.")
        current_columns = _dashboard_column_count()
        if st.session_state.get("dashboard_columns_selector") not in range(MIN_DASHBOARD_COLUMNS, MAX_DASHBOARD_COLUMNS + 1):
            st.session_state.dashboard_columns_selector = current_columns
        st.selectbox(
            "Dashboard columns",
            list(range(MIN_DASHBOARD_COLUMNS, MAX_DASHBOARD_COLUMNS + 1)),
            key="dashboard_columns_selector",
            on_change=_change_dashboard_columns,
            help="You can only remove an empty column. Move its components yourself before reducing the column count.",
        )
        if st.session_state.get("dashboard_column_notice"):
            st.warning(st.session_state.dashboard_column_notice)
        st.caption("Columns expand vertically as needed. Components never auto-pack, overlap, or move to another column.")

        if chart_ids:
            customize_auto = st.session_state.customize_auto_chart_ids or []
            customize_auto = st.multiselect(
                "Auto Dashboard chart library",
                chart_ids,
                default=[item for item in customize_auto if item in chart_ids],
                format_func=lambda item: labels[item],
                key="customize_auto_chart_selector",
            )
            st.session_state.customize_auto_chart_ids = list(customize_auto)

        custom_chart_ids = [item["id"] for item in st.session_state.custom_visualizations]
        if custom_chart_ids:
            custom_labels = {item["id"]: item.get("title", item.get("label", item["id"])) for item in st.session_state.custom_visualizations}
            selected_generated = st.session_state.selected_generated_chart_ids
            default_generated = [] if selected_generated is None else [item for item in selected_generated if item in custom_chart_ids]
            st.session_state.selected_generated_chart_ids = st.multiselect("Generated charts to include", custom_chart_ids, default=default_generated, format_func=lambda item: custom_labels[item], key="selected_generated_chart_selector")

        _render_kpi_customize_section(analytics)

        with st.expander("Dataset Summary"):
            summary_labels = dict(DATASET_SUMMARY_OPTIONS)
            st.caption("Choose the sections to show in the final dashboard. Remove a section to hide it; the preview above remains unchanged.")
            selected_summary = st.multiselect(
                "Summary sections to include",
                list(summary_labels),
                default=[item for item in st.session_state.customize_summary_section_ids if item in summary_labels],
                format_func=lambda item: summary_labels[item],
                key="customize_dataset_summary_selector",
            )
            st.session_state.customize_summary_section_ids = list(selected_summary)

        with st.expander("+ Add analysis section"):
            section_name = st.text_input("Section name", placeholder="Customer performance", key="new_section_name")
            analysis_type = st.selectbox("Analysis type", ["Trend", "Comparison", "Distribution", "Relationship", "Ranking"], key="new_section_type")
            matches = [item for item in recommendations if (analysis_type == "Trend" and item["kind"] == "time_series") or (analysis_type == "Comparison" and item["kind"] in {"grouped_bar", "categorical_bar"}) or (analysis_type == "Distribution" and item["kind"] in {"histogram", "boxplot", "categorical_bar"}) or (analysis_type == "Relationship" and item["kind"] in {"scatter", "correlation_heatmap"}) or (analysis_type == "Ranking" and item["kind"] in {"categorical_bar", "grouped_bar"})]
            if matches:
                selected_section_chart = st.selectbox("Recommended chart", [item["id"] for item in matches], format_func=lambda item: next(chart["title"] for chart in matches if chart["id"] == item), key="new_section_chart")
                if st.button("Add section", key="add_analysis_section") and section_name.strip():
                    st.session_state.custom_sections.append({"name": section_name.strip(), "analysis_type": analysis_type, "chart_id": selected_section_chart})
                    selected_auto_ids = list(st.session_state.customize_auto_chart_ids or [])
                    if selected_section_chart not in selected_auto_ids:
                        st.session_state.customize_auto_chart_ids = [*selected_auto_ids, selected_section_chart]
                    st.success(f"Added section: {section_name.strip()}")
            else:
                st.info("No compatible generated chart is available for this analysis type.")

        st.text_area("Storytelling notes", placeholder="Add context for a customer or executive presentation.", key="story_text")
        _render_dashboard_layout_editor(analytics)


def _render_sidebar() -> None:
    """Render upload and deployment-owned configuration status."""

    st.sidebar.header("Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload any CSV dataset", type=["csv"])
    st.sidebar.caption("DataWonder automatically detects fields, analyses the data, and builds the dashboard.")
    if uploaded_file is not None:
        _load_upload(uploaded_file)

    error = st.session_state.get("upload_error")
    if error:
        for message in error.get("errors", []):
            st.sidebar.error(message)
        for message in error.get("warnings", []):
            st.sidebar.warning(message)


def _recommended_kpi_specs(analytics: dict) -> list[dict]:
    """Normalize analytics-engine KPI recommendations into editable card specs."""

    icon_by_calculation = {"count": "▦", "average": "◉", "count_distinct": "⌁"}
    specs = []
    for item in analytics.get("kpis", []):
        calculation = item.get("calculation", "count")
        specs.append(
            {
                **item,
                "kind": "recommended",
                "title": item.get("label", item.get("id", "KPI")),
                "description": f"Recommended {calculation.replace('_', ' ')} from the detected dataset fields.",
                "number_format": "Number",
                "decimals": 2 if calculation == "average" else 0,
                "icon": icon_by_calculation.get(calculation, "▦"),
                "theme_color": "Auto (theme)",
                "target": None,
                "compare_previous": False,
            }
        )
    return specs


def _all_kpi_specs(analytics: dict) -> list[dict]:
    """Return recommended and user-created KPI card specs available to Customize Dashboard."""

    return [*_recommended_kpi_specs(analytics), *st.session_state.get("custom_kpis", [])]


def _format_kpi_value(value: object, spec: dict) -> str:
    """Format KPI numbers according to the selected card settings."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    decimals = max(0, min(int(spec.get("decimals", 0)), 6))
    number_format = spec.get("number_format", "Number")
    if number_format == "Percentage":
        return f"{number * 100:,.{decimals}f}%"
    if number_format == "Currency":
        return f"${number:,.{decimals}f}"
    return f"{number:,.{decimals}f}" if decimals else f"{number:,.0f}"


def _aggregate_kpi_series(data: pd.DataFrame, column: str, aggregation: str) -> float | int | None:
    """Calculate a KPI using only the selected column and aggregation."""

    if column not in data.columns:
        return None
    series = pd.to_numeric(data[column], errors="coerce")
    if aggregation == "Count":
        return int(series.notna().sum())
    if aggregation == "Unique Count":
        return int(series.nunique(dropna=True))
    series = series.dropna()
    if series.empty:
        return None
    operation = {
        "Sum": "sum",
        "Average": "mean",
        "Median": "median",
        "Minimum": "min",
        "Maximum": "max",
        "Standard Deviation": "std",
    }.get(aggregation, "mean")
    value = getattr(series, operation)()
    return float(value) if pd.notna(value) else None


def _calculate_custom_kpi(spec: dict) -> dict:
    """Calculate a custom KPI, optional filter, period comparison, and target delta."""

    data = _apply_dashboard_filters(st.session_state.cleaned_data, st.session_state.get("dashboard_filters", {}))
    filter_column = spec.get("filter_column")
    filter_value = spec.get("filter_value", "<all>")
    if filter_column in data.columns and filter_value not in (None, "", "<all>"):
        data = data[data[filter_column].astype("string") == str(filter_value)].copy()
    column = spec.get("column")
    date_column = spec.get("date_column")
    current_value = _aggregate_kpi_series(data, column, spec.get("aggregation", "Average"))
    comparison_pct = None
    if spec.get("compare_previous") and date_column in data.columns:
        dates = pd.to_datetime(data[date_column], errors="coerce")
        periods = dates.dt.to_period("M")
        valid_periods = periods.dropna()
        if not valid_periods.empty:
            latest_period = valid_periods.max()
            previous_period = latest_period - 1
            current_data = data[periods == latest_period]
            previous_data = data[periods == previous_period]
            current_value = _aggregate_kpi_series(current_data, column, spec.get("aggregation", "Average"))
            previous_value = _aggregate_kpi_series(previous_data, column, spec.get("aggregation", "Average"))
            if current_value is not None and previous_value not in (None, 0):
                comparison_pct = (current_value - previous_value) / abs(previous_value) * 100
    target = spec.get("target")
    target_delta = None if target in (None, "") or current_value is None else current_value - float(target)
    return {"value": current_value, "comparison_pct": comparison_pct, "target_delta": target_delta}


def _kpi_display_data(spec: dict, analytics: dict) -> dict:
    if spec.get("kind") == "custom":
        return _calculate_custom_kpi(spec)
    return {"value": spec.get("value"), "comparison_pct": None, "target_delta": None}


def _kpi_card_html(spec: dict, display_data: dict) -> str:
    """Render the same theme-aware KPI card markup for Streamlit and HTML export."""

    accent = THEME_MANAGER.kpi_color(spec.get("theme_color", "Auto (theme)"))
    icon = "" if spec.get("icon", "None") == "None" else f'<div class="dw-kpi-icon">{escape(str(spec.get("icon")))}</div>'
    title = escape(str(spec.get("title", spec.get("label", "KPI"))))
    description = escape(str(spec.get("description", "")))
    value = escape(_format_kpi_value(display_data.get("value"), spec))
    comparison = display_data.get("comparison_pct")
    comparison_html = "" if comparison is None else f'<div class="dw-kpi-description">Previous period: {comparison:+.1f}%</div>'
    target_delta = display_data.get("target_delta")
    target_html = "" if target_delta is None else f'<div class="dw-kpi-description">Target variance: {target_delta:+,.2f}</div>'
    description_html = f'<div class="dw-kpi-description">{description}</div>' if description else ""
    return f'<div class="dw-kpi-card" style="--dw-kpi-accent:{accent}">{icon}<div class="dw-kpi-label">{title}</div><div class="dw-kpi-value">{value}</div>{description_html}{comparison_html}{target_html}</div>'


def _ordered_kpi_ids() -> list[str]:
    """Return selected KPI IDs; the canvas owns their visible placement."""

    return list(dict.fromkeys(st.session_state.get("customize_kpi_ids", []) or []))


def _render_kpi_cards(analytics: dict, selected_ids: list[str] | None = None, heading: str = "KPI cards") -> None:
    """Render KPI recommendations as preview or only explicitly selected final cards."""

    effective_analytics = _analytics_for_current_filters(analytics)
    recommended = _recommended_kpi_specs(effective_analytics)
    all_specs = [*recommended, *st.session_state.get("custom_kpis", [])]
    by_id = {spec["id"]: spec for spec in all_specs}
    ids = [spec["id"] for spec in recommended] if selected_ids is None else [item for item in selected_ids if item in by_id]
    if not ids:
        if selected_ids is not None:
            st.info("No KPI cards are selected for this final dashboard.")
        else:
            st.info("No KPI cards could be generated from the available fields.")
        return
    st.subheader(heading)
    columns = st.columns(min(len(ids), 4))
    for index, kpi_id in enumerate(ids):
        spec = {**by_id[kpi_id], **st.session_state.get("kpi_overrides", {}).get(kpi_id, {})}
        with columns[index % len(columns)]:
            st.markdown(_kpi_card_html(spec, _kpi_display_data(spec, effective_analytics)), unsafe_allow_html=True)


def _render_kpi_customize_section(analytics: dict) -> None:
    """Provide KPI selection, editing, and custom KPI creation controls."""

    with st.expander("KPI Cards"):
        st.caption("Auto Dashboard KPI cards are previews. Select only the cards you want in the final dashboard.")
        specs = _all_kpi_specs(analytics)
        labels = {spec["id"]: st.session_state.get("kpi_overrides", {}).get(spec["id"], {}).get("title", spec.get("title", spec.get("label", spec["id"]))) for spec in specs}
        ids = [spec["id"] for spec in specs]
        selected = st.multiselect(
            "KPI cards to include",
            ids,
            default=[item for item in st.session_state.customize_kpi_ids if item in ids],
            format_func=lambda item: labels[item],
            key="customize_kpi_selector",
        )
        st.session_state.customize_kpi_ids = list(selected)

        with st.container(border=True):
            st.markdown("#### Edit selected KPI card")
            if not selected:
                st.caption("Select a KPI card above to edit its title, format, icon, target, or accent color.")
            else:
                selected_id = st.selectbox("KPI card to edit", selected, format_func=lambda item: labels[item], key="kpi_edit_selection")
                current = {**next(spec for spec in specs if spec["id"] == selected_id), **st.session_state.kpi_overrides.get(selected_id, {})}
                version = st.session_state.kpi_builder_version
                title = st.text_input("KPI title", value=current.get("title", current.get("label", "KPI")), key=f"kpi_edit_title_{selected_id}_{version}")
                description = st.text_input("Description", value=current.get("description", ""), key=f"kpi_edit_description_{selected_id}_{version}")
                format_col, decimal_col = st.columns(2)
                with format_col:
                    number_format = st.selectbox("Number format", KPI_NUMBER_FORMATS, index=KPI_NUMBER_FORMATS.index(current.get("number_format", "Number")) if current.get("number_format", "Number") in KPI_NUMBER_FORMATS else 0, key=f"kpi_edit_format_{selected_id}_{version}")
                with decimal_col:
                    decimals = st.number_input("Decimal places", min_value=0, max_value=6, value=int(current.get("decimals", 0)), step=1, key=f"kpi_edit_decimals_{selected_id}_{version}")
                icon_col, color_col = st.columns(2)
                with icon_col:
                    icon = st.selectbox("Icon", KPI_ICONS, index=KPI_ICONS.index(current.get("icon", "None")) if current.get("icon", "None") in KPI_ICONS else 0, key=f"kpi_edit_icon_{selected_id}_{version}")
                with color_col:
                    theme_color = st.selectbox("Theme color", KPI_THEME_COLORS, index=KPI_THEME_COLORS.index(current.get("theme_color", "Auto (theme)")) if current.get("theme_color", "Auto (theme)") in KPI_THEME_COLORS else 0, key=f"kpi_edit_color_{selected_id}_{version}")
                target_enabled = st.checkbox("Add target / benchmark", value=current.get("target") is not None, key=f"kpi_edit_target_enabled_{selected_id}_{version}")
                target = st.number_input("Target value", value=float(current.get("target") or 0), key=f"kpi_edit_target_{selected_id}_{version}") if target_enabled else None
                if st.button("Save KPI settings", type="primary", key=f"save_kpi_settings_{selected_id}", use_container_width=True):
                    st.session_state.kpi_overrides[selected_id] = {"title": title, "description": description, "number_format": number_format, "decimals": int(decimals), "icon": icon, "theme_color": theme_color, "target": target}
                    st.session_state.kpi_builder_version += 1
                    st.rerun()

        with st.container(border=True):
            st.markdown("#### Create custom KPI card")
            profile = analytics.get("profile", {})
            numeric_columns = profile.get("numerical_columns", [])
            all_columns = profile.get("columns", [])
            date_columns = profile.get("datetime_columns", [])
            if not numeric_columns:
                st.info("No numerical columns are available for a custom KPI card.")
            else:
                version = st.session_state.kpi_builder_version
                metric = st.selectbox("Numerical column", numeric_columns, key=f"kpi_builder_metric_{version}")
                aggregation = st.selectbox("Aggregation", KPI_AGGREGATIONS, index=1, key=f"kpi_builder_aggregation_{version}")
                title = st.text_input("Custom KPI title", value=f"{aggregation} {metric}", key=f"kpi_builder_title_{version}")
                description = st.text_input("Custom KPI description", value=f"Calculated from {metric}.", key=f"kpi_builder_description_{version}")
                filter_col, filter_val = st.columns(2)
                with filter_col:
                    filter_column = st.selectbox("Filter column", ["<none>", *all_columns], key=f"kpi_builder_filter_column_{version}")
                filter_value = "<all>"
                if filter_column != "<none>":
                    values = sorted({str(value) for value in st.session_state.cleaned_data[filter_column].dropna().head(200).tolist()})
                    with filter_val:
                        filter_value = st.selectbox("Filter value", ["<all>", *values], key=f"kpi_builder_filter_value_{version}")
                date_choice = st.selectbox("Time column for previous-period comparison", ["<none>", *date_columns], key=f"kpi_builder_date_{version}")
                compare_previous = st.checkbox("Compare with previous month", disabled=date_choice == "<none>", key=f"kpi_builder_compare_{version}")
                target_enabled = st.checkbox("Add target / benchmark", key=f"kpi_builder_target_enabled_{version}")
                target = st.number_input("Target value", value=0.0, key=f"kpi_builder_target_{version}") if target_enabled else None
                format_col, decimal_col = st.columns(2)
                with format_col:
                    number_format = st.selectbox("Number format", KPI_NUMBER_FORMATS, key=f"kpi_builder_format_{version}")
                with decimal_col:
                    decimals = st.number_input("Decimal places", min_value=0, max_value=6, value=2, step=1, key=f"kpi_builder_decimals_{version}")
                icon_col, color_col = st.columns(2)
                with icon_col:
                    icon = st.selectbox("Icon", KPI_ICONS, key=f"kpi_builder_icon_{version}")
                with color_col:
                    theme_color = st.selectbox("Theme color", KPI_THEME_COLORS, key=f"kpi_builder_color_{version}")
                if st.button("Add KPI to Customize Dashboard", type="primary", key=f"add_custom_kpi_{version}", use_container_width=True):
                    seed = json.dumps({"metric": metric, "aggregation": aggregation, "title": title, "filter_column": filter_column, "filter_value": filter_value, "date": date_choice, "count": len(st.session_state.custom_kpis)}, sort_keys=True)
                    custom_id = f"custom_kpi_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:10]}"
                    spec = {"id": custom_id, "kind": "custom", "title": title, "description": description, "column": metric, "aggregation": aggregation, "filter_column": None if filter_column == "<none>" else filter_column, "filter_value": filter_value, "date_column": None if date_choice == "<none>" else date_choice, "compare_previous": compare_previous, "target": target, "number_format": number_format, "decimals": int(decimals), "icon": icon, "theme_color": theme_color}
                    st.session_state.custom_kpis.append(spec)
                    st.session_state.customize_kpi_ids = [*st.session_state.customize_kpi_ids, custom_id]
                    st.session_state.kpi_builder_version += 1
                    st.rerun()


def _render_kpis(analytics: dict) -> None:
    """Render recommended KPI cards as an Auto Dashboard preview."""

    _render_kpi_cards(analytics, selected_ids=None, heading="Recommended KPI cards preview")


def _render_auto_chart_editor(analytics: dict) -> None:
    """Edit automatic chart presentation directly from the Auto Dashboard sheet."""

    recommendations = sorted(analytics.get("visualization_recommendations", []), key=lambda item: item.get("priority", 99))
    if not recommendations:
        return
    chart_ids = [item["id"] for item in recommendations]
    labels = {item["id"]: item["title"] for item in recommendations}
    with st.expander("Edit Auto Dashboard charts"):
        st.caption("Change the visualization type or title here. Customize Dashboard is reserved for final selection and storytelling.")
        selected_chart = st.selectbox("Chart to edit", chart_ids, format_func=lambda item: labels[item], key="auto_chart_to_customize")
        spec = next(item for item in recommendations if item["id"] == selected_chart)
        choices = _chart_type_options(spec)
        choice_labels = [item[0] for item in choices]
        current_override = st.session_state.chart_overrides.get(selected_chart, {})
        current_kind = current_override.get("kind", spec["kind"])
        current_index = next((index for index, item in enumerate(choices) if item[1] == current_kind), 0)
        editor_version = st.session_state.chart_editor_version
        selected_label = st.selectbox("Visualization", choice_labels, index=current_index, key=f"auto_chart_type_{selected_chart}_{editor_version}")
        selected_kind = dict(choices)[selected_label]
        title = st.text_input("Chart title", value=current_override.get("title", spec["title"]), key=f"auto_chart_title_{selected_chart}_{editor_version}")
        preview_col, save_col, reset_col = st.columns(3)
        preview_override = {"kind": selected_kind, "title": title}
        with preview_col:
            if st.button("Preview changes", key=f"preview_auto_chart_{selected_chart}", use_container_width=True):
                st.session_state.chart_editor_preview = {"chart_id": selected_chart, "override": preview_override}
                st.rerun()
        with save_col:
            if st.button("Save chart changes", type="primary", key=f"save_auto_chart_{selected_chart}", use_container_width=True):
                st.session_state.chart_overrides[selected_chart] = preview_override
                st.session_state.chart_editor_preview = None
                st.rerun()
        with reset_col:
            if st.button("Reset this chart", key=f"reset_auto_chart_{selected_chart}", use_container_width=True):
                st.session_state.chart_overrides.pop(selected_chart, None)
                st.session_state.chart_editor_preview = None
                st.session_state.chart_editor_version += 1
                st.rerun()

        preview = st.session_state.get("chart_editor_preview")
        if preview and preview.get("chart_id") == selected_chart:
            st.markdown("#### Preview")
            if preview.get("override", {}) != preview_override:
                st.info("The editor settings changed after the last preview. Click Preview changes to refresh it.")
            else:
                preview_figure = _recommended_figure(spec, analytics, preview.get("override", {}))
                if preview_figure is not None:
                    preview_figure.update_layout(margin={"l": 20, "r": 20, "t": 55, "b": 20})
                    st.plotly_chart(preview_figure, use_container_width=True, key=f"auto_chart_preview_{selected_chart}_{editor_version}")
                    st.caption("Preview only. Click Save chart changes to apply this chart to Auto Dashboard and exports.")
                else:
                    st.info("This visualization cannot be rendered with the available data.")


def _trend_figure(spec: dict, analytics: dict, render_kind: str | None = None, title: str | None = None):
    """Build a chart for an engine-recommended time-series visualization."""

    metric = spec["columns"][1]
    date_column = spec["columns"][0]
    trend = next(
        (item for item in analytics.get("time_series_trends", []) if item["metric"] == metric and item["date_column"] == date_column),
        None,
    )
    if not trend:
        return None
    frame = pd.DataFrame(trend["records"])
    chart_title = title or spec["title"]
    if render_kind == "time_series_line":
        return px.line(frame, x="period", y="total", markers=True, title=chart_title, labels={"period": date_column, "total": metric})
    if render_kind == "time_series_bar":
        return px.bar(frame, x="period", y="total", title=chart_title, labels={"period": date_column, "total": metric})
    return px.area(frame, x="period", y="total", markers=True, title=chart_title, labels={"period": date_column, "total": metric})


def _categorical_figure(spec: dict, analytics: dict, title: str | None = None):
    """Build a chart for an engine-recommended categorical visualization."""

    column = spec["columns"][0]
    distribution = next((item for item in analytics.get("categorical_distributions", []) if item["column"] == column), None)
    if not distribution:
        return None
    frame = pd.DataFrame(distribution["top_values"])
    figure = px.bar(frame, x="count", y="value", orientation="h", title=title or spec["title"], labels={"count": "Rows", "value": column})
    figure.update_layout(yaxis={"categoryorder": "total ascending"})
    return figure


def _categorical_pie_figure(spec: dict, analytics: dict, title: str | None = None):
    """Build a composition chart for a low-cardinality categorical field."""

    column = spec["columns"][0]
    distribution = next((item for item in analytics.get("categorical_distributions", []) if item["column"] == column), None)
    if not distribution:
        return None
    frame = pd.DataFrame(distribution["top_values"])
    return px.pie(frame, names="value", values="count", title=title or spec["title"], hole=0.35)


def _grouped_figure(spec: dict, analytics: dict, render_kind: str | None = None, title: str | None = None):
    """Build an aggregation chart for a categorical/numerical pair."""

    dimension, metric = spec["columns"]
    grouped = next(
        (item for item in analytics.get("grouped_summaries", []) if item["dimension"] == dimension and item["metric"] == metric),
        None,
    )
    if not grouped:
        return None
    frame = pd.DataFrame(grouped["records"])
    measure = "average" if render_kind == "grouped_average" else "total"
    return px.bar(frame, x="dimension", y=measure, title=title or spec["title"], labels={"dimension": dimension, measure: f"{measure.title()} {metric}"})


def _numeric_figure(spec: dict, analytics: dict, render_kind: str | None = None, title: str | None = None):
    """Build a histogram or scatter plot from the cleaned arbitrary dataset."""

    data = st.session_state.cleaned_data
    columns = spec["columns"]
    if any(column not in data.columns for column in columns):
        return None
    kind = render_kind or spec["kind"]
    chart_title = title or spec["title"]
    if kind == "histogram":
        return px.histogram(data, x=columns[0], nbins=30, title=chart_title)
    if kind == "boxplot":
        return px.box(data, y=columns[0], points="outliers", title=chart_title)
    if kind == "scatter":
        return px.scatter(data, x=columns[0], y=columns[1], title=chart_title, opacity=0.75)
    return None


def _correlation_figure(analytics: dict):
    """Build a correlation heatmap only when the engine found one."""

    matrix_records = analytics.get("correlations", {}).get("matrix", [])
    if not matrix_records:
        return None
    matrix = pd.DataFrame(matrix_records).set_index("column")
    figure = go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale=THEME_MANAGER.heatmap_colorscale(), zmin=-1, zmax=1))
    figure.update_layout(title="Numerical relationships")
    return figure


def _recommended_figure(spec: dict, analytics: dict, override: dict | None = None):
    """Return the Plotly figure for a generated chart and optional edit override."""

    override = override or {}
    render_kind = override.get("kind", spec["kind"])
    title = override.get("title") or spec["title"]
    figure = None
    if spec["kind"] == "time_series":
        figure = _trend_figure(spec, analytics, render_kind, title)
    elif render_kind == "categorical_pie":
        figure = _categorical_pie_figure(spec, analytics, title)
    elif spec["kind"] in {"categorical_bar", "categorical_pie"}:
        figure = _categorical_figure(spec, analytics, title)
    elif spec["kind"] == "grouped_bar":
        figure = _grouped_figure(spec, analytics, render_kind, title)
    elif spec["kind"] in {"histogram", "boxplot", "scatter"}:
        figure = _numeric_figure(spec, analytics, render_kind, title)
    elif spec["kind"] == "correlation_heatmap":
        figure = _correlation_figure(analytics)
        if figure is not None:
            figure.update_layout(title=title)
    return _apply_dashboard_chart_theme(figure) if figure is not None else None


def _render_recommended_chart(spec: dict, analytics: dict, chart_key: str, override: dict | None = None) -> None:
    """Render one chart selected by analytics_engine recommendation metadata."""

    figure = _recommended_figure(spec, analytics, override)
    if figure is not None:
        figure.update_layout(margin={"l": 20, "r": 20, "t": 55, "b": 20})
        st.plotly_chart(figure, use_container_width=True, key=chart_key)


def _render_charts(analytics: dict) -> None:
    """Render main and supporting charts from automatic recommendations."""

    recommendations = sorted(analytics.get("visualization_recommendations", []), key=lambda item: item.get("priority", 99))
    if not recommendations:
        st.info("No compatible visualizations could be generated from the available fields.")
        return

    visible_ids = [item["id"] for item in recommendations[:8]]
    visible = [item for item in recommendations if item["id"] in visible_ids]
    if not visible:
        st.info("No recommended charts could be displayed from the available fields.")
        return

    st.subheader("Main charts")
    main = visible[:2]
    columns = st.columns(min(len(main), 2))
    for index, spec in enumerate(main):
        with columns[index % len(columns)]:
            _render_recommended_chart(spec, analytics, f"main_chart_{index}", st.session_state.chart_overrides.get(spec["id"]))

    supporting = visible[2:]
    if supporting:
        st.subheader("Supporting charts")
        columns = st.columns(2)
        for index, spec in enumerate(supporting):
            with columns[index % 2]:
                _render_recommended_chart(spec, analytics, f"supporting_chart_{index}", st.session_state.chart_overrides.get(spec["id"]))


def _selected_dashboard_items(analytics: dict) -> list[tuple[str, dict]]:
    """Return manually selected Auto/Chart Generator items for the layout canvas."""

    recommendations = {item["id"]: item for item in analytics.get("visualization_recommendations", [])}
    selected_auto = list(st.session_state.customize_auto_chart_ids or [])
    selected_generated = list(st.session_state.selected_generated_chart_ids or [])
    generated_by_id = {item["id"]: item for item in st.session_state.custom_visualizations}
    items: list[tuple[str, dict]] = []

    for chart_id in selected_auto:
        if chart_id in recommendations:
            items.append(("auto", recommendations[chart_id]))
    for chart_id in selected_generated:
        if chart_id in generated_by_id:
            items.append(("generated", generated_by_id[chart_id]))
    return items


def _render_custom_visualizations(selected_only: bool = False) -> None:
    """Render charts added through Chart Generator."""

    if not st.session_state.custom_visualizations:
        return
    charts = st.session_state.custom_visualizations
    if selected_only:
        by_id = {item["id"]: item for item in charts}
        charts = [by_id[chart_id] for chart_id in (st.session_state.selected_generated_chart_ids or []) if chart_id in by_id]
    if not charts:
        st.info("No generated charts are selected for the final dashboard.")
        return
    st.subheader("Added visualizations")
    columns = st.columns(_dashboard_column_count())
    for index, spec in enumerate(charts):
        with columns[index % len(columns)]:
            _render_custom_visualization(spec, f"custom_visualization_{index}")


def _render_selected_dashboard_charts(analytics: dict) -> None:
    """Render the final dashboard using only explicitly selected charts."""

    items = _selected_dashboard_items(analytics)
    if not items:
        st.info("No charts are selected yet. Open the chart library above to choose charts for this final dashboard.")
        return
    st.subheader("Selected dashboard charts")
    columns = st.columns(_dashboard_column_count())
    for index, (kind, spec) in enumerate(items):
        with columns[index % len(columns)]:
            if kind == "auto":
                _render_recommended_chart(spec, analytics, f"selected_dashboard_auto_{index}", st.session_state.chart_overrides.get(spec["id"]))
            else:
                _render_custom_visualization(spec, f"selected_dashboard_generated_{index}")


def _render_custom_sections(analytics: dict) -> None:
    """Render user-added storytelling/analysis sections from existing chart specs."""

    if st.session_state.story_text.strip():
        st.subheader("Storytelling notes")
        st.markdown(f'<div class="dw-section">{escape(st.session_state.story_text).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    recommendations = {item["id"]: item for item in analytics.get("visualization_recommendations", [])}
    for index, section in enumerate(st.session_state.custom_sections):
        spec = recommendations.get(section.get("chart_id"))
        if not spec:
            continue
        st.subheader(section.get("name", f"Custom section {index + 1}"))
        _render_recommended_chart(spec, analytics, f"custom_section_{index}", st.session_state.chart_overrides.get(spec["id"]))


def _prepare_export_figure(kind: str, spec: dict, figure):
    """Apply the exact active theme again at the export boundary."""

    if figure is None:
        return None
    palette = _chart_colors(spec) if kind == "generated" else None
    chart_style = spec.get("chart_style", "Auto") if kind == "generated" else "Auto"
    return THEME_MANAGER.apply_export_figure(
        figure,
        palette=palette,
        chart_style=chart_style,
        theme_name=st.session_state.get("dashboard_theme", "Executive"),
    )


def _add_kpi_export_annotations(figure, analytics: dict):
    """Add the selected KPI cards above PNG/PDF chart exports using the active theme."""

    selected_ids = _ordered_kpi_ids()
    if not selected_ids:
        return figure
    effective_analytics = _analytics_for_current_filters(analytics)
    by_id = {spec["id"]: spec for spec in _all_kpi_specs(effective_analytics)}
    selected_specs = [
        {**by_id[kpi_id], **st.session_state.get("kpi_overrides", {}).get(kpi_id, {})}
        for kpi_id in selected_ids
        if kpi_id in by_id
    ]
    if not selected_specs:
        return figure

    themed_figure = go.Figure(figure)
    theme = _dashboard_chart_theme()
    card_count = min(len(selected_specs), 4)
    for index, spec in enumerate(selected_specs[:card_count]):
        display_data = _kpi_display_data(spec, effective_analytics)
        accent = THEME_MANAGER.kpi_color(spec.get("theme_color", "Auto (theme)"))
        title = escape(str(spec.get("title", spec.get("label", "KPI"))))
        value = escape(_format_kpi_value(display_data.get("value"), spec))
        themed_figure.add_annotation(
            x=(index + 0.5) / card_count,
            y=1.14,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=f"<b>{title}</b><br><span style='font-size:20px'><b>{value}</b></span>",
            font={"color": theme["text"], "size": 11},
            bgcolor=theme["surface"],
            bordercolor=accent,
            borderwidth=1,
            borderpad=8,
        )
    margin = themed_figure.layout.margin.to_plotly_json() if themed_figure.layout.margin else {}
    margin["t"] = max(int(margin.get("t", 0) or 0), 132)
    themed_figure.update_layout(margin=margin)
    return themed_figure


def _dashboard_insights_html(analytics: dict) -> str:
    """Render AI insights and recommendations for ordered dashboard export."""

    blocks = ["<section class=\"summary-section\"><h3>AI insights and recommendations</h3>"]
    insights = analytics.get("insights", [])
    if insights:
        blocks.append("<h4>Key insights</h4>")
        blocks.extend(f"<p><strong>{escape(str(item.get('title', 'Insight')))}</strong> — {escape(str(item.get('message', '')))}</p>" for item in insights)
    else:
        blocks.append("<p class=\"summary-notice\">No additional insights were generated from the available fields.</p>")
    recommendations = analytics.get("recommendations", [])
    if recommendations:
        blocks.append("<h4>Recommendations</h4>")
        blocks.extend(f"<p><strong>{escape(str(item.get('title', 'Recommendation')))}</strong> — {escape(str(item.get('message', '')))}</p>" for item in recommendations)
    blocks.append("</section>")
    return "".join(blocks)


def _dashboard_quality_html(analytics: dict, theme: dict) -> str:
    """Render a compact data-quality component for ordered export."""

    quality = analytics.get("data_quality", {})
    rows = quality.get("rows_after_cleaning", quality.get("rows_received", 0))
    missing = quality.get("missing_values", {})
    missing_text = "No missing values were detected." if not missing else ", ".join(f"{escape(str(column))} ({count:,})" for column, count in missing.items())
    return f"<section class=\"summary-section\"><h3>Data quality report</h3><div class=\"summary-metrics\"><div class=\"summary-metric\"><div class=\"summary-label\">Rows received</div><div class=\"summary-value\">{quality.get('rows_received', rows):,}</div></div><div class=\"summary-metric\"><div class=\"summary-label\">Rows after cleaning</div><div class=\"summary-value\">{rows:,}</div></div><div class=\"summary-metric\"><div class=\"summary-label\">Duplicate rows</div><div class=\"summary-value\">{quality.get('duplicate_rows_received', 0):,}</div></div></div><p class=\"summary-notice\"><strong>Missing values:</strong> {missing_text}</p></section>"


def _responsive_plotly_html(figure) -> str:
    """Return a Plotly export that is strictly constrained to its dashboard column."""

    figure.update_layout(autosize=True, width=None)
    chart = pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
        default_width="100%",
        default_height="440px",
    )
    return f'<div class="dw-chart-frame">{chart}</div>'


def _build_html_snapshot(analytics: dict) -> str:
    """Create a portable Plotly HTML snapshot for the current dashboard view."""

    export_analytics = _analytics_for_current_filters(analytics)
    chart_html_by_id: dict[str, str] = {}
    for kind, spec in _selected_dashboard_items(export_analytics):
        figure = _recommended_figure(spec, export_analytics, st.session_state.chart_overrides.get(spec["id"])) if kind == "auto" else _manual_visualization_figure(spec)
        figure = _prepare_export_figure(kind, spec, figure)
        if figure is not None:
            chart_html_by_id[_dashboard_chart_component_id(kind, spec["id"])] = _responsive_plotly_html(figure)
    metadata = export_analytics.get("dashboard_metadata", {})
    title = escape(st.session_state.dashboard_title or metadata.get("title", "DataWonder Dashboard"))
    story = escape(st.session_state.story_text).replace("\n", "<br>")
    theme = _dashboard_chart_theme()
    kpi_by_id = {spec["id"]: spec for spec in _all_kpi_specs(export_analytics)}
    lane_html: list[str] = []
    recommendations = {item["id"]: item for item in export_analytics.get("visualization_recommendations", [])}
    for lane in _dashboard_layout_lanes(export_analytics):
        component_html: list[str] = []
        for component in lane:
            component_id = component["id"]
            component_type = component["type"]
            content = ""
            if component_type == "kpi" and component["kpi_id"] in kpi_by_id:
                spec = {**kpi_by_id[component["kpi_id"]], **st.session_state.get("kpi_overrides", {}).get(component["kpi_id"], {})}
                content = _kpi_card_html(spec, _kpi_display_data(spec, export_analytics))
            elif component_type == "summary":
                content = _dataset_summary_html(export_analytics, [component["section_id"]], theme)
            elif component_type == "chart":
                content = chart_html_by_id.get(component_id, "")
            elif component_type == "section":
                section = component["section"]
                spec = recommendations.get(section.get("chart_id"))
                if spec:
                    figure = _prepare_export_figure("auto", spec, _recommended_figure(spec, export_analytics, st.session_state.chart_overrides.get(spec["id"])))
                    if figure is not None:
                        content = f"<h3>{escape(str(section.get('name', 'Custom section')))}</h3>{_responsive_plotly_html(figure)}"
            elif component_type == "text":
                content = f"<section class=\"summary-section\"><h3>Storytelling notes</h3><p>{story}</p></section>"
            elif component_type == "insights":
                content = _dashboard_insights_html(export_analytics)
            elif component_type == "quality":
                content = _dashboard_quality_html(export_analytics, theme)
            if content:
                component_html.append(f'<section class="layout-component" data-component-id="{escape(component_id)}"><div class="dw-layout-handle">{escape(str(component["label"]))}</div><div class="dw-component-content">{content}</div></section>')
        lane_html.append(f'<div class="dashboard-lane" style="display:flex;flex-direction:column;gap:18px">{"".join(component_html)}</div>')
    export_layout_css = """
    *, *::before, *::after { box-sizing: border-box; }
    .dashboard-grid, .dashboard-lane, .layout-component, .dw-component-content { min-width: 0; max-width: 100%; }
    .dashboard-lane { width: 100%; }
    .layout-component { width: 100%; overflow: hidden; }
    .dw-component-content, .dw-chart-frame { width: 100%; min-width: 0; max-width: 100%; }
    .dw-chart-frame { overflow: hidden; }
    .dw-chart-frame .plotly-graph-div, .dw-chart-frame .plot-container, .dw-chart-frame .svg-container { width: 100% !important; max-width: 100% !important; min-width: 0 !important; }
    .dw-table-scroll { width: 100%; max-width: 100%; overflow-x: auto; overflow-y: hidden; }
    .summary-table { width: max-content; min-width: 100%; max-width: none; }
    .layout-component p, .layout-component h1, .layout-component h2, .layout-component h3, .layout-component h4, .layout-component td, .layout-component th { overflow-wrap: anywhere; word-break: break-word; }
    """
    layout_html = f"<style>{export_layout_css}</style>" + ("".join(lane_html) or '<p class="summary-notice">No dashboard components are selected.</p>')
    layout_columns = len(lane_html)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title><style>body{{font-family:Arial,sans-serif;background:{theme['background']};color:{theme['text']};margin:32px}}h1{{margin-bottom:4px}}.dashboard-grid{{display:grid;grid-template-columns:repeat({max(1, min(int(layout_columns), 3))},minmax(0,1fr));gap:18px;align-items:start}}.layout-component,.dw-kpi-card,.summary-section{{background:{theme['surface']};border:1px solid {theme['border']};border-radius:14px;padding:14px;min-width:0}}.dw-layout-handle{{color:{theme['muted']};font-size:12px;font-weight:600;margin-bottom:8px}}.summary-metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}.kpi-label,.dw-kpi-label,.summary-label{{color:{theme['muted']};font-size:13px}}.kpi-value,.dw-kpi-value,.summary-value{{color:{theme['accent']};font-size:24px;font-weight:700;margin-top:6px}}.dw-kpi-icon,.dw-kpi-value{{color:var(--dw-kpi-accent,{theme['accent']})}}.dw-kpi-icon{{font-size:20px}}.dw-kpi-description{{color:{theme['muted']};font-size:12px;margin-top:4px}}.summary-section{{margin:0}}.summary-section h3{{margin-top:0}}.summary-table{{width:100%;border-collapse:collapse;color:{theme['text']}}}.summary-table th{{background:{theme['table_header']};color:{theme['text']};text-align:left}}.summary-table th,.summary-table td{{border:1px solid {theme['border']};padding:8px}}.summary-notice{{color:{theme['muted']};padding:8px 0}}@media(max-width:900px){{.dashboard-grid{{grid-template-columns:1fr}}.summary-metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}</style></head>
<body><h1>{title}</h1><p>{escape(metadata.get('subtitle', 'Automatically generated dashboard'))}</p><div class="dashboard-grid">{layout_html}</div></body></html>"""


def _dashboard_configuration() -> dict:
    """Return all user-controlled dashboard state needed for save/load."""

    return {
        "dashboard_title": st.session_state.dashboard_title,
        "dashboard_subtitle": st.session_state.dashboard_subtitle,
        "theme": st.session_state.dashboard_theme,
        "theme_settings": THEME_MANAGER.export_settings(),
        "selected_auto_chart_ids": st.session_state.selected_auto_chart_ids,
        "customize_auto_chart_ids": st.session_state.customize_auto_chart_ids,
        "selected_generated_chart_ids": st.session_state.selected_generated_chart_ids,
        "customize_kpi_ids": st.session_state.customize_kpi_ids,
        "custom_kpis": st.session_state.custom_kpis,
        "kpi_overrides": st.session_state.kpi_overrides,
        "customize_summary_section_ids": st.session_state.customize_summary_section_ids,
        "chart_overrides": st.session_state.chart_overrides,
        "custom_visualizations": st.session_state.custom_visualizations,
        "saved_charts": st.session_state.saved_charts,
        "custom_sections": st.session_state.custom_sections,
        "story_text": st.session_state.story_text,
        "filters": st.session_state.dashboard_filters,
        "layout": st.session_state.dashboard_layout,
        "chart_generator_last_selections": st.session_state.chart_generator_last_selections,
    }


def _apply_dashboard_configuration(configuration: dict) -> None:
    """Restore safe, user-controlled dashboard settings from JSON."""

    if not isinstance(configuration, dict):
        raise ValueError("Configuration must be a JSON object.")
    theme = configuration.get("theme", "Executive")
    if theme not in THEMES:
        theme = "Executive"
    st.session_state.dashboard_title = str(configuration.get("dashboard_title", configuration.get("title", st.session_state.dashboard_title)))
    st.session_state.dashboard_subtitle = str(configuration.get("dashboard_subtitle", configuration.get("subtitle", st.session_state.dashboard_subtitle)))
    st.session_state.dashboard_theme = theme
    st.session_state.selected_auto_chart_ids = configuration.get("selected_auto_chart_ids", configuration.get("visible_chart_ids"))
    customize_ids = configuration.get("customize_auto_chart_ids", [])
    st.session_state.customize_auto_chart_ids = customize_ids if isinstance(customize_ids, list) else []
    generated_ids = configuration.get("selected_generated_chart_ids", [])
    st.session_state.selected_generated_chart_ids = generated_ids if isinstance(generated_ids, list) else []
    kpi_ids = configuration.get("customize_kpi_ids", [])
    st.session_state.customize_kpi_ids = kpi_ids if isinstance(kpi_ids, list) else []
    custom_kpis = configuration.get("custom_kpis", [])
    st.session_state.custom_kpis = custom_kpis if isinstance(custom_kpis, list) else []
    kpi_overrides = configuration.get("kpi_overrides", {})
    st.session_state.kpi_overrides = kpi_overrides if isinstance(kpi_overrides, dict) else {}
    st.session_state.kpi_builder_version += 1
    summary_ids = configuration.get("customize_summary_section_ids", [])
    st.session_state.customize_summary_section_ids = summary_ids if isinstance(summary_ids, list) else []
    st.session_state.chart_overrides = configuration.get("chart_overrides", {}) if isinstance(configuration.get("chart_overrides", {}), dict) else {}
    st.session_state.chart_editor_preview = None
    st.session_state.chart_editor_version += 1
    st.session_state.custom_visualizations = configuration.get("custom_visualizations", []) if isinstance(configuration.get("custom_visualizations", []), list) else []
    st.session_state.saved_charts = configuration.get("saved_charts", []) if isinstance(configuration.get("saved_charts", []), list) else []
    st.session_state.custom_sections = configuration.get("custom_sections", []) if isinstance(configuration.get("custom_sections", []), list) else []
    st.session_state.story_text = str(configuration.get("story_text", ""))
    st.session_state.dashboard_filters = configuration.get("filters", {}) if isinstance(configuration.get("filters", {}), dict) else {}
    raw_layout = configuration.get("layout", {})
    raw_layout = raw_layout if isinstance(raw_layout, dict) else {}
    layout_columns = _dashboard_column_count(raw_layout)
    raw_lanes = raw_layout.get("lanes", [])
    raw_lanes = raw_lanes if isinstance(raw_lanes, list) else []
    layout_was_customized = bool(raw_layout.get("customized", configuration.get("dashboard_layout_customized", bool(raw_lanes))))
    all_lanes = [
        [component_id for component_id in lane if isinstance(component_id, str)] if isinstance(lane, list) else []
        for lane in raw_lanes[:MAX_DASHBOARD_COLUMNS]
    ]
    layout_columns = max(layout_columns, max((index + 1 for index, lane in enumerate(all_lanes) if lane), default=0))
    lanes = all_lanes[:layout_columns]
    legacy_order = configuration.get("dashboard_component_order", raw_layout.get("order", []))
    if not any(lanes) and isinstance(legacy_order, list):
        lanes = _default_dashboard_lanes([component_id for component_id in legacy_order if isinstance(component_id, str)], layout_columns)
    while len(lanes) < layout_columns:
        lanes.append([])
    st.session_state.dashboard_layout = {
        "columns": layout_columns,
        "lanes": lanes,
        "customized": layout_was_customized,
    }
    st.session_state.dashboard_columns_selector = layout_columns
    st.session_state.dashboard_column_notice = None
    last_selections = configuration.get("chart_generator_last_selections", {})
    st.session_state.chart_generator_last_selections = last_selections if isinstance(last_selections, dict) else {}


def _reset_dashboard_state(analytics: dict) -> None:
    """Reset presentation and exploration state without touching the uploaded data."""

    metadata = analytics.get("dashboard_metadata", {})
    st.session_state.update(
        edit_mode=False,
        dashboard_theme="Executive",
        chart_overrides={},
        custom_sections=[],
        custom_visualizations=[],
        story_text="",
        dashboard_title=metadata.get("title", "DataWonder Executive Dashboard"),
        dashboard_subtitle=metadata.get("subtitle", ""),
        dashboard_layout={"columns": DEFAULT_DASHBOARD_COLUMNS, "lanes": [], "customized": False},
        dashboard_columns_selector=DEFAULT_DASHBOARD_COLUMNS,
        dashboard_column_notice=None,
        dashboard_filters={},
        loaded_config_signature=None,
        selected_auto_chart_ids=None,
        customize_auto_chart_ids=[],
        selected_generated_chart_ids=None,
        customize_kpi_ids=[],
        custom_kpis=[],
        kpi_overrides={},
        kpi_builder_version=0,
        customize_summary_section_ids=[],
        chart_editor_version=0,
        chart_editor_preview=None,
        chart_generator_draft=None,
        chart_generator_last_selections={},
    )


def _refresh_analysis() -> None:
    """Re-run cleaning and analytics against the current uploaded dataset."""

    raw_data = st.session_state.get("raw_data")
    if raw_data is None:
        return
    cleaned = clean_dataset(raw_data.copy())
    analytics = analyze_dataset(cleaned)
    st.session_state.cleaned_data = cleaned
    st.session_state.analytics = analytics
    st.session_state.analytics_context = build_ai_context(analytics)


def _theme_rgb(color: str) -> tuple[int, int, int]:
    """Convert a theme hex color to an RGB tuple for raster exports."""

    value = str(color).lstrip("#")
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    try:
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
    except (TypeError, ValueError):
        return (255, 255, 255)


def _export_font(size: int, bold: bool = False):
    """Load a portable font for image exports, falling back to Pillow's default."""

    if ImageFont is None:
        return None
    candidates = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _build_raster_dashboard(analytics: dict, image_format: str) -> bytes:
    """Render the persisted dashboard grid into a theme-aware PNG or PDF canvas."""

    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for PNG/PDF dashboard export. Install outputs/requirements.txt first.")
    export_analytics = _analytics_for_current_filters(analytics)
    component_lanes = _dashboard_layout_lanes(export_analytics)
    theme = _dashboard_chart_theme()
    columns = max(1, len(component_lanes))
    card_width, card_height, gap, margin, header_height = 520, 390, 18, 32, 100
    rows = max(1, max((len(lane) for lane in component_lanes), default=0))
    canvas = Image.new("RGB", (margin * 2 + columns * card_width + (columns - 1) * gap, margin * 2 + header_height + rows * card_height + (rows - 1) * gap), _theme_rgb(theme["background"]))
    draw = ImageDraw.Draw(canvas)
    title_font = _export_font(28, True)
    subtitle_font = _export_font(13)
    label_font = _export_font(12, True)
    value_font = _export_font(24, True)
    body_font = _export_font(12)
    title = st.session_state.get("dashboard_title") or export_analytics.get("dashboard_metadata", {}).get("title", "DataWonder Dashboard")
    subtitle = st.session_state.get("dashboard_subtitle") or export_analytics.get("dashboard_metadata", {}).get("subtitle", "")
    draw.text((margin, margin), str(title), fill=_theme_rgb(theme["text"]), font=title_font)
    draw.text((margin, margin + 42), str(subtitle), fill=_theme_rgb(theme["muted"]), font=subtitle_font)

    def draw_lines(x: int, y: int, text: str, width: int = 62, max_lines: int = 13, font=None, fill=None, line_gap: int = 5) -> None:
        wrapped = textwrap.wrap(str(text), width=width) or [""]
        draw.multiline_text((x, y), "\n".join(wrapped[:max_lines]), font=font or body_font, fill=fill or _theme_rgb(theme["text"]), spacing=line_gap)

    def chart_image(component: dict):
        try:
            if component["type"] == "chart":
                kind = component["chart_kind"]
                spec = component["spec"]
                figure = _recommended_figure(spec, export_analytics, st.session_state.chart_overrides.get(spec["id"])) if kind == "auto" else _manual_visualization_figure(spec)
            else:
                spec = {item["id"]: item for item in export_analytics.get("visualization_recommendations", [])}.get(component["section"].get("chart_id"))
                figure = _recommended_figure(spec, export_analytics, st.session_state.chart_overrides.get(spec["id"])) if spec else None
            figure = _prepare_export_figure("auto" if component["type"] == "section" else component.get("chart_kind", "auto"), spec or {}, figure)
            if figure is None:
                return None
            image_bytes = pio.to_image(figure, format="png", width=card_width - 36, height=285)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image.thumbnail((card_width - 36, 285))
            return image
        except Exception:
            return None

    for column, lane in enumerate(component_lanes):
        for row, component in enumerate(lane):
            x = margin + column * (card_width + gap)
            y = margin + header_height + row * (card_height + gap)
            draw.rounded_rectangle((x, y, x + card_width, y + card_height), radius=14, fill=_theme_rgb(theme["surface"]), outline=_theme_rgb(theme["border"]), width=2)
            draw.text((x + 16, y + 14), str(component["label"]), fill=_theme_rgb(theme["muted"]), font=label_font)
            content_x, content_y = x + 16, y + 48
            component_type = component["type"]
            if component_type == "kpi":
                spec = next((item for item in _all_kpi_specs(export_analytics) if item["id"] == component["kpi_id"]), None)
                if spec:
                    spec = {**spec, **st.session_state.get("kpi_overrides", {}).get(component["kpi_id"], {})}
                    display_data = _kpi_display_data(spec, export_analytics)
                    draw.text((content_x, content_y + 20), _format_kpi_value(display_data.get("value"), spec), fill=_theme_rgb(THEME_MANAGER.kpi_color(spec.get("theme_color", "Auto (theme)"))), font=value_font)
                    draw_lines(content_x, content_y + 62, spec.get("description", ""), width=64, max_lines=3, fill=_theme_rgb(theme["muted"]))
            elif component_type in {"chart", "section"}:
                image = chart_image(component)
                if image is not None:
                    canvas.paste(image, (content_x, content_y + 6))
                else:
                    draw_lines(content_x, content_y + 20, "This visualization could not be rendered in the current export runtime.", max_lines=5)
            elif component_type == "summary":
                payload = _dataset_summary_payload(component["section_id"], export_analytics)
                if payload["kind"] == "metrics":
                    metric_text = "   ".join(f"{label}: {value}" for label, value in payload["items"])
                    draw_lines(content_x, content_y + 20, metric_text, width=60, max_lines=6)
                elif payload["kind"] == "table":
                    draw_lines(content_x, content_y + 10, payload["frame"].head(9).to_string(index=False), width=68, max_lines=15)
                else:
                    draw_lines(content_x, content_y + 20, payload["message"], max_lines=8)
            elif component_type == "text":
                draw_lines(content_x, content_y + 18, st.session_state.get("story_text", ""), max_lines=16)
            elif component_type == "insights":
                messages = [f"{item.get('title', 'Insight')}: {item.get('message', '')}" for item in export_analytics.get("insights", [])]
                messages.extend(f"{item.get('title', 'Recommendation')}: {item.get('message', '')}" for item in export_analytics.get("recommendations", []))
                draw_lines(content_x, content_y + 8, "\n".join(messages) or "No additional insights were generated.", width=66, max_lines=16)
            elif component_type == "quality":
                quality = export_analytics.get("data_quality", {})
                quality_text = f"Rows received: {quality.get('rows_received', 0):,}\nRows after cleaning: {quality.get('rows_after_cleaning', 0):,}\nDuplicate rows: {quality.get('duplicate_rows_received', 0):,}\nMissing values: {len(quality.get('missing_values', {}))} field(s)"
                draw_lines(content_x, content_y + 18, quality_text, max_lines=8)

    output = BytesIO()
    canvas.save(output, format="PDF" if image_format == "pdf" else "PNG", resolution=150)
    return output.getvalue()


def _toolbar_export_payload(analytics: dict, export_format: str):
    """Build the selected toolbar export payload, when supported by the runtime."""

    if export_format == "HTML":
        return _build_html_snapshot(analytics), "datawonder_dashboard.html", "text/html"
    if export_format == "CSV (Processed Dataset)":
        return st.session_state.cleaned_data.to_csv(index=False), "datawonder_processed_dataset.csv", "text/csv"
    if export_format in {"PNG Image", "PDF"}:
        image_format = "png" if export_format == "PNG Image" else "pdf"
        payload = _build_raster_dashboard(analytics, image_format)
        return payload, f"datawonder_dashboard.{image_format}", "image/png" if image_format == "png" else "application/pdf"
    raise ValueError("PowerPoint export is planned for a future release.")


def _render_toolbar(analytics: dict, sheet_key: str, title: str, description: str) -> None:
    """Render a clean sheet header with one consolidated Export entry point."""

    with st.container():
        header_columns = st.columns([7, 1])
        with header_columns[0]:
            st.markdown(f"### {title}")
            st.caption(description)
        with header_columns[1]:
            with st.popover("Export ▾", use_container_width=True):
                st.markdown("#### Export Dashboard")
                st.download_button(
                    "Download HTML Dashboard",
                    _build_html_snapshot(analytics),
                    file_name="datawonder_dashboard.html",
                    mime="text/html",
                    key=f"export_html_{sheet_key}",
                    use_container_width=True,
                )
                st.download_button(
                    "Download Dashboard Configuration",
                    json.dumps(_dashboard_configuration(), indent=2, ensure_ascii=False),
                    file_name="datawonder_dashboard_config.json",
                    mime="application/json",
                    key=f"export_config_{sheet_key}",
                    use_container_width=True,
                )
                with st.expander("More export formats"):
                    export_format = st.selectbox("Format", ["CSV (Processed Dataset)", "PDF", "PNG Image", "PowerPoint (Future)"], key=f"export_more_format_{sheet_key}")
                    if export_format == "PowerPoint (Future)":
                        st.caption("PowerPoint export coming soon.")
                    else:
                        try:
                            payload, filename, mime = _toolbar_export_payload(analytics, export_format)
                            st.download_button("Download selected format", payload, file_name=filename, mime=mime, key=f"export_more_download_{sheet_key}", use_container_width=True)
                        except Exception as exc:  # noqa: BLE001 - surface optional export dependency issues safely
                            st.warning(f"{export_format} export is unavailable in this runtime: {exc}")


def _render_dashboard_controls(analytics: dict, sheet_key: str) -> None:
    """Keep non-export dashboard actions available without crowding the header."""

    with st.expander("Dashboard actions"):
        action_columns = st.columns(3)
        with action_columns[0]:
            config_file = st.file_uploader("Load Configuration", type=["json"], key=f"toolbar_config_upload_{sheet_key}")
            if config_file is not None:
                signature = hashlib.sha256(config_file.getvalue()).hexdigest()
                if signature != st.session_state.loaded_config_signature:
                    try:
                        _apply_dashboard_configuration(json.loads(config_file.getvalue().decode("utf-8")))
                        st.session_state.loaded_config_signature = signature
                        st.success("Dashboard configuration loaded.")
                    except Exception as exc:  # noqa: BLE001 - convert malformed config into UI feedback
                        st.error(f"Could not load configuration: {exc}")
        with action_columns[1]:
            if st.button("Reset Dashboard", key=f"toolbar_reset_{sheet_key}", use_container_width=True):
                _reset_dashboard_state(analytics)
                st.rerun()
        with action_columns[2]:
            if st.button("Refresh Analysis", key=f"toolbar_refresh_{sheet_key}", use_container_width=True):
                _refresh_analysis()
                st.rerun()


def _render_quality(analytics: dict) -> None:
    """Render source quality and inferred schema details."""

    quality = analytics["data_quality"]
    st.subheader("Data quality")
    st.write(
        f"Received **{quality['rows_received']:,}** rows and retained **{quality['rows_after_cleaning']:,}** "
        f"after removing **{quality['rows_removed']:,}** duplicate row(s)."
    )
    missing = quality.get("missing_values", {})
    if missing:
        st.warning("Missing values by field: " + ", ".join(f"{key} ({value:,})" for key, value in missing.items()))
    else:
        st.success("No missing values were detected after cleaning.")

    with st.expander("Detected fields and analysis availability"):
        profile = analytics["profile"]
        schema = pd.DataFrame([{"column": column, "detected_type": kind} for column, kind in profile["type_by_column"].items()])
        st.dataframe(THEME_MANAGER.style_dataframe(schema), use_container_width=True, hide_index=True)
        availability = pd.DataFrame(
            [{"analysis": name, "status": "Available", "reason": ""} for name in analytics.get("available_analyses", [])]
            + [{"analysis": name, "status": "Skipped", "reason": reason} for name, reason in analytics.get("unavailable_analyses", {}).items()]
        )
        st.dataframe(THEME_MANAGER.style_dataframe(availability), use_container_width=True, hide_index=True)
    with st.expander("Data preview"):
        st.dataframe(THEME_MANAGER.style_dataframe(st.session_state.cleaned_data.head(100)), use_container_width=True, hide_index=True)


def _render_insights_and_recommendations(analytics: dict) -> None:
    """Render executive findings and cautious next steps."""

    left, right = st.columns(2)
    with left:
        st.subheader("Key insights")
        insights = analytics.get("insights", [])
        if insights:
            for insight in insights:
                st.info(f"**{insight['title']}** — {insight['message']}")
        else:
            st.info("No additional insights were generated from the available fields.")
    with right:
        st.subheader("Recommendations")
        recommendations = analytics.get("recommendations", [])
        for recommendation in recommendations:
            st.info(f"**{recommendation['title']}** — {recommendation['message']}")


def _render_chat() -> None:
    """Render the optional AI copilot at the bottom of the dashboard."""

    st.divider()
    st.subheader("Ask DataWonder")
    st.caption("Optional copilot: answers are grounded in the generated dashboard metadata and verified analyses.")
    question = st.chat_input("Ask about the dashboard, trends, relationships, or data quality")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Reviewing the verified dashboard context..."):
            response = answer_question(question, st.session_state.analytics_context, history=st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    if st.session_state.chat_history:
        st.markdown("#### Conversation history")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def _render_back_to_top() -> None:
    """Render a safe floating anchor with smooth scrolling for long dashboards."""

    st.markdown(
        """
        <style>
        html { scroll-behavior: smooth; }
        #dw-back-to-top {
            position: fixed;
            right: 24px;
            bottom: 24px;
            z-index: 999999;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 10px 14px;
            border: 1px solid rgba(148, 163, 184, .35);
            border-radius: 999px;
            background: rgba(17, 24, 39, .94);
            color: #f8fafc !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, .22);
            font-size: 13px;
            font-weight: 600;
            line-height: 1;
            text-decoration: none !important;
            opacity: .95;
            visibility: visible;
            transform: translateY(0);
            transition: opacity .2s ease, visibility .2s ease, transform .2s ease;
        }
        #dw-back-to-top:hover {
            background: #2563eb;
            color: #ffffff !important;
            transform: translateY(-2px);
        }
        body.dw-at-top #dw-back-to-top {
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            transform: translateY(8px);
        }
        @media (max-width: 640px) {
            #dw-back-to-top {
                right: 14px;
                bottom: 14px;
                padding: 10px 12px;
            }
            #dw-back-to-top .dw-back-to-top-label { display: none; }
        }
        </style>
        <div id="dw-top"></div>
        <a id="dw-back-to-top" href="#dw-top" aria-label="Back to top" title="Back to top">
            <span aria-hidden="true">↑</span>
            <span class="dw-back-to-top-label">Back to top</span>
        </a>
        <script>
        (function () {
            const updateBackToTop = function () {
                document.body.classList.toggle('dw-at-top', window.scrollY < 320);
            };
            updateBackToTop();
            window.addEventListener('scroll', updateBackToTop, { passive: true });
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


def _render_customize_dashboard(analytics: dict) -> None:
    """Render the final presentation-oriented dashboard layer."""

    _render_customize_panel(analytics, show_toggle=False)
    _apply_theme(st.session_state.dashboard_theme)

    st.divider()
    st.subheader(st.session_state.dashboard_title or analytics.get("dashboard_metadata", {}).get("title", "DataWonder Dashboard"))
    if st.session_state.dashboard_subtitle:
        st.caption(st.session_state.dashboard_subtitle)
    _render_ordered_dashboard_components(analytics)


def main() -> None:
    """Render DataWonder's three-layer dashboard experience."""

    _initialise_state()
    _render_sidebar()
    if st.session_state.analytics is None:
        st.title("DataWonder")
        st.caption("Upload a CSV to generate an automatic dashboard and optionally create custom visualizations.")
        st.info("Upload any CSV from the sidebar to begin. No predefined columns are required.")
        return

    analytics = st.session_state.analytics
    metadata = analytics.get("dashboard_metadata", {})
    _apply_theme(st.session_state.dashboard_theme)
    _render_back_to_top()
    auto_tab, generator_tab, customize_tab = st.tabs(["Auto Dashboard", "Chart Generator", "Customize Dashboard"])

    with auto_tab:
        _render_toolbar(analytics, "auto", metadata.get("title", "Auto Dashboard"), metadata.get("subtitle", "Automatically generated from the uploaded dataset."))
        _render_dashboard_controls(analytics, "auto")
        _render_kpis(analytics)
        st.caption("Preview only. Choose charts manually in Customize Dashboard when you are ready to build the final dashboard.")
        _render_dataset_summary(analytics, _recommended_dataset_summary_ids(analytics), "Dataset Summary preview")
        _render_charts(analytics)
        _render_auto_chart_editor(analytics)
        _render_quality(analytics)
        _render_insights_and_recommendations(analytics)

    with generator_tab:
        _render_toolbar(analytics, "generator", "Chart Generator", "Create additional visualizations from the uploaded dataset.")
        _render_dashboard_controls(analytics, "generator")
        _render_chart_generator(analytics)

    with customize_tab:
        _render_toolbar(analytics, "customize", "Customize Dashboard", "Select, style, and arrange the final storytelling dashboard.")
        _render_dashboard_controls(analytics, "customize")
        _render_customize_dashboard(analytics)

    _render_chat()


if __name__ == "__main__":
    main()
