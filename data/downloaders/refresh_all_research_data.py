from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.config.data_sources import PROJECT_ROOT


downloaders = [
    "data/downloaders/download_cryptocompare_daily_candles.py",
    "data/downloaders/download_fred_macro_data.py",
    "data/downloaders/download_yfinance_market_data.py",
    "data/downloaders/download_binance_btc_funding_rate.py",
    "data/downloaders/download_deribit_btc_dvol.py",
    "data/downloaders/build_data_inventory.py",
]

for downloader in downloaders:
    script_path = PROJECT_ROOT / Path(downloader)
    print(f"\nRunning {downloader}...")
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)

print("\nFinished refreshing research data.")
