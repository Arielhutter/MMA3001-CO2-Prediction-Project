# predicting indoor CO2 from environmental sensors (MMA3001)

The study: can CO2 be estimated from temperature, humidity, pressure,
VOC, particulates and hour of day? Main method: Support Vector Regression;
alternative: Linear Regression. Data: Monash Smart Infrastructure sensors.

## Structure
- `project.ipynb` – full workflow (audit, cleaning, models, tuning, performance)
- `src/co2_model.py` – tested helper functions
- `tests/` – pytest tests; report in `docs/test_report.html`
- `docs/html/` – generated API documentation (open `docs/html/src/co2_model.html`)
- `docs/Predicting Indoor CO2 form Environmental Sensors Report.pdf` - written project report (also submitted via moodle)

## Setup
python -m venv .venv
.venv\Scripts\activate        (Windows)   |   source .venv/bin/activate (macOS)
pip install -r requirements.txt

## Data
Place `5EnvSensor_MayToDec2024_180kRows.csv` in `data/raw/` (not committed).
The file name says 2024 but the records run to April 2026.

## Run
1. Open `project.ipynb`, choose the `.venv` kernel, Restart and Run All.
2. Tests: `python -m pytest -v`
3. Docs: `python -m pdoc src -d numpy -o docs/html`

## Key results
See report Sections 4 to 6. In summary: SVR beats a mean predictor by ~5% in the 
unknown room; no model reliably beats a constant on Upper Compactus.

