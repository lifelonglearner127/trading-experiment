import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ROOT_PATH = Path(__file__).parent.resolve()
CSV_OUTPUT_DIR_PATH = os.path.join(ROOT_PATH, "out")
SYMBOL = "BTCUSDT"


class TimeInterval(Enum):
    # INTERVAL_1D = "1d"
    # INTERVAL_12H = "12h"
    # INTERVAL_8H = "8h"
    INTERVAL_4H = "4h"
    # INTERVAL_1H = "1h"
    # INTERVAL_15M = "15m"
    # INTERVAL_5M = "5m"
