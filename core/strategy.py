# core/strategy.py

import backtrader as bt
import numpy as np

class AceStrategy(bt.Strategy):
    params = dict(
        model=None,
        features=[],
        confidence_thresh=0.6,
        risk_per_trade=0.02,
        reward_risk=1.5,
    )

    def __init__(self):
        self.model = self.p.model
        self.order = None
        self.order_pending = False

        # Initialize indicators for each selected feature (e.g., MA, RSI)
        for feat in self.p.features:
            setattr(self, feat, bt.indicators.SimpleMovingAverage(self.data.close, period=1))

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.datetime()
        print(f"{dt.isoformat()} — {txt}")

    def next(self):
        if self.order_pending:
            return

        # Build feature vector for model prediction
        row = [getattr(self.data, col)[0] for col in self.p.features]
        confs = self.model.predict_proba([row])[0]
        pred = np.argmax(confs)
        conf = np.max(confs)

        # Trade setup
        price = self.data.close[0]
        atr = bt.ind.ATR(self.data, period=14)[0]
        stop_dist = 2 * atr
        tp_dist = self.p.reward_risk * stop_dist

        if not self.position and conf > self.p.confidence_thresh:
            size = (self.broker.get_cash() * self.p.risk_per_trade) / stop_dist
            if pred == 1:
                self.order = self.buy_bracket(
                    size=size,
                    price=price,
                    stopprice=price - stop_dist,
                    limitprice=price + tp_dist
                )
            else:
                self.order = self.sell_bracket(
                    size=size,
                    price=price,
                    stopprice=price + stop_dist,
                    limitprice=price - tp_dist
                )
            self.order_pending = True
            self.log(f"ORDER CREATED: type={'BUY' if pred==1 else 'SELL'}, size={size:.2f}, conf={conf:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(f"{side} EXECUTED at price {order.executed.price:.5f}, "
                     f"Cost {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order_pending = False
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE CLOSED: Gross PnL {trade.pnl:.2f}, Net PnL {trade.pnlcomm:.2f}")

    def start(self):
        self.log("Strategy started")
        self.order_pending = False

    def stop(self):
        self.log("Strategy completed")
