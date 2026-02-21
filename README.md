## Live demo
https://mrp-shortage-analyzer-kevjc.streamlit.app

# MRP Shortage Analyzer (Python + Streamlit)

Tool to analyze MRP exports and detect material shortages based on projected stock (`Unrestricted`).

## Features
- Upload CSV/TSV exports (supports `,`, `;` or tab separators)
- Detect shortage events (`Unrestricted < 0`) per component
- Earliest critical time and maximum deficit
- Component detail view with timeline table and chart (Unrestricted vs time)
- Estimates initial stock if `Unrestricted = Stock - AccumulatedNeed` (Stock ≈ Unrestricted + AccumulatedNeed)

## Tech Stack
- Python, pandas
- Streamlit
- Matplotlib

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py