# DataWonder

> **Turn any CSV into a presentation-ready dashboard—in one click.**

DataWonder is a dataset-agnostic business analytics dashboard. Upload a CSV and it automatically profiles the data, identifies the available analytical opportunities, generates visual insights, and lets users build a final presentation dashboard without learning Power BI or Tableau.

## Live demo and source code

- **Live demo:** `https://<your-streamlit-subdomain>.streamlit.app`
- **Source code:** `https://github.com/<your-github-username>/datawonder`

Replace these placeholders after publishing the project. The two URLs can then be used in Devpost's **Try it out** section.

## Why DataWonder?

Many teams have CSV data but lack the time, expertise, or BI tooling to turn it into a useful business story. DataWonder removes fixed assumptions such as `Sales`, `Product`, `Customer ID`, or `Date`. It analyzes whatever fields are actually present and clearly skips an analysis when the data does not support it.

## Features

- Upload **any CSV** with no predefined schema or manual column mapping
- Automatic column detection for numerical, categorical, datetime, and text fields
- Adaptive EDA, descriptive statistics, distributions, correlations, trends, anomalies, and data-quality checks
- Graceful availability messages instead of fabricated results
- Automatic Dashboard for immediate insights
- Chart Generator for custom visualizations
- Customize Dashboard for selecting KPIs, charts, tables, summaries, and insights
- Drag-and-drop component layout with user-controlled columns and ordering
- Theme-aware charts, KPI cards, tables, HTML, PNG, and PDF exports
- Optional, server-configured AI copilot grounded in generated dashboard context

## How Codex and GPT-5.6 were used

Codex and GPT-5.6 were used as development collaborators throughout the project lifecycle:

- Translated product requirements into a dataset-agnostic analytics workflow.
- Helped design the three-layer experience: **Auto Dashboard**, **Chart Generator**, and **Customize Dashboard**.
- Implemented and refined Python, Streamlit, Plotly, export, theme, and session-state logic.
- Diagnosed UI issues involving drag-and-drop persistence, responsive multi-column layouts, and HTML export overflow.
- Added validation checks for layout preservation, exports, and dependency behavior.
- Helped write product copy, the elevator pitch, and this project documentation.

GPT-5.6 was used during **development through Codex**; it is not required for DataWonder's core CSV analytics workflow. The application can analyze uploaded data without any user API key.

## Built with

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Kaleido and Pillow
- streamlit-sortables
- Google Gen AI SDK and Gemini API *(optional copilot)*
- HTML, CSS, JavaScript, JSON, and CSV

## Run locally

```bash
cd outputs
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, then upload a CSV from the sidebar.

### Optional AI copilot

The dashboard works without an AI key. To enable the optional Gemini-powered copilot, configure server-side environment variables:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

Never commit real keys, `.env`, or Streamlit secrets to a public repository.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and push this project.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app using your repository and select `outputs/app.py` as the entrypoint.
4. Streamlit installs dependencies from `outputs/requirements.txt`.
5. Optionally add Gemini settings through Streamlit's **Advanced settings / Secrets** interface.
6. Copy the resulting `https://<name>.streamlit.app` URL into this README and Devpost.

## Project structure

```text
.
├── README.md
├── .env.example
└── outputs/
    ├── app.py                 # Streamlit interface and dashboard customization
    ├── analytics_engine.py    # Dataset profiling, cleaning, EDA, and recommendations
    ├── ai_agent.py            # Optional server-side Gemini copilot
    └── requirements.txt
```

## What's next

- More data sources, including Excel and databases
- Forecasting and more advanced anomaly detection
- Shareable dashboards and collaboration
- PowerPoint export for executive presentations
- Additional reusable visualization templates
