# GKI Python Project

This project accompanies the GKI course and contains Python-based data analysis, 
visualization, and classification tasks using Jupyter notebooks and scripts.

---

## 📁 Project Structure

```
GKI-python/
├── data/           # input data: raw and processed CSVs, zips
├── figs/           # output plots (e.g., PNGs for wine, MNIST, etc.)
├── notebooks/      # Jupyter notebooks for interactive exploration
├── scripts/        # standalone .py scripts for plotting, ML, etc.
├── src/            # reusable helper functions (e.g., `example.py`)
├── .venv/          # project virtual environment
├── requirements.txt
├── main.py
└── README.md
```

---

## 🔧 Setup

1. Clone the repo:
   ```bash
   git clone <your_repo_url>
   cd GKI-python
   ```

2. Create and activate the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

- Run scripts:
  ```bash
  python scripts/03_01_weinplots.py
  ```

- Start notebooks:
  ```bash
  jupyter lab
  # or open notebooks/ in VSCode
  ```

---

## 📌 Notes

- **Excel file:** `GKIS_PSO_Implementierung_in_Excel.xlsx` is used for a specific
  task and kept in the project root for visibility.
- Data sources are stored in `data/` and must remain locally available.
- Figures generated are saved to `figs/`.

---

## ✅ Status

- Scripts and notebooks are functional
- Folder structure follows modern data science conventions
- Works on Linux/macOS with Python 3.8+

---

## 📬 Contact

For questions or contributions, contact the course instructor or open an issue.
