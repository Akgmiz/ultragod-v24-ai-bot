from flask import Flask
import threading, time
from ultragod_v24_complete_production import CompleteTradingSystem

app = Flask(__name__)
bot = CompleteTradingSystem()

def run_bot():
    while True:
        bot.run_cycle()
        time.sleep(10)

threading.Thread(target=run_bot, daemon=True).start()

@app.route('/')
def home():
    return "✅ ULTRA-GOD v24 is running on Render (Free Cloud Version)"

@app.route('/status')
def status():
    return {
        "balance": bot.balance,
        "positions": bot.positions,
        "trading_enabled": bot.trading_enabled
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
