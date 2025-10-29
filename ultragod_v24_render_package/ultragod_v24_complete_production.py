import time, random, pandas as pd

class CompleteTradingSystem:
    def __init__(self):
        self.balance = 10000
        self.positions = []
        self.trading_enabled = True
        print("✅ ULTRA-GOD v24 Initialized Successfully")

    def generate_signal(self):
        signal = random.choice(["BUY", "SELL", "HOLD"])
        print(f"📊 Generated Signal: {signal}")
        return signal

    def execute_trade(self, signal):
        if signal == "BUY":
            self.positions.append({"type": "LONG", "amount": 100})
            self.balance -= 100
            print("🟢 BUY executed")
        elif signal == "SELL" and self.positions:
            self.positions.pop()
            self.balance += 120
            print("🔴 SELL executed")
        else:
            print("⏸ HOLD (no action)")

    def run_cycle(self):
        if not self.trading_enabled:
            print("⚠️ Trading paused.")
            return
        signal = self.generate_signal()
        self.execute_trade(signal)
        print(f"💰 Current Balance: {self.balance}")
        print("-" * 50)
        time.sleep(1)
