## Setup

### 1) Clone the repo
git clone https://github.com/Diennhutvo/transit-delay-prediction
cd transit-delay-prediction

### 2) Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

### 3) Install dependencies
python -m pip install -r requirements.txt

### 4) Add API key
api: mban2026
pass: Mban2026!
Open API Key is: e25jm8i2vh3ZHGgLRndd


### 5) Run scripts
python scripts/test_gtfs_rt_trip_updates.py
python scripts/parse_trip_updates_to_csv.py


