from localbitmex.bitmex_websocket import BitMEXWebsocket
import logging
import json
from time import sleep
from datetime import datetime, timedelta, timezone
from dateutil import parser as datetime_parser
import pandas as pd


# Basic use of websocket.
def run():
    logger = setup_logger()

    # Instantiating the WS will make it connect. Be sure to add your api_key/api_secret.
    ws = BitMEXWebsocket(endpoint="https://www.bitmex.com/api/v1", symbol="XBTUSD",
                         api_key=None, api_secret=None)
    # api_key=api_key, api_secret=api_secret)

    logger.info("Instrument data: %s" % ws.get_instrument())

    # Run forever
    while ws.ws.sock.connected:
        # logger.info("Ticker: %s" % ws.get_ticker())
        # logger.info("Data: %s" % ws.recent_trades())
        # if ws.api_key:
        #     logger.info("Funds: %s" % ws.funds())
        # logger.info("Market Depth: %s" % ws.market_depth())
        # logger.info("Recent Trades: %s\n\n" % ws.recent_trades())
        sleep(10)
        candles = ws.candles
        # candles_data = pd.DataFrame(candles)
        print('Candles: ', len(candles))
        # order_book = ws.market_depth()
        # trades = ws.recent_trades()
        # candle_trades = []
        # order_book_buy = []
        # order_book_sell = []
        # for trade in trades:
        #     if datetime.now(timezone.utc) - datetime_parser.parse(trade['timestamp']) > candle_time:
        #         continue
        #     # print(json.dumps(trade))
        #     candle_trades.append(trade)
        # for order in order_book:
        #     if order['side'] == 'Buy':
        #         order_book_buy.append(order)
        #     else:
        #         order_book_sell.append(order)
        # data_order_book_sell = pd.DataFrame(order_book_sell)
        # data_order_book_buy = pd.DataFrame(order_book_buy)
        # data_order_book_buy.sort_values('price', ascending=False, inplace=True)
        # data_order_book_sell.sort_values('price', ascending=True, inplace=True)
        # print(data_order_book_sell.head())
        # print(data_order_book_buy.head())
        # data = pd.DataFrame(candle_trades)
        # data.drop(['trdMatchID', 'foreignNotional', 'grossValue', 'homeNotional', 'symbol', 'tickDirection'], axis=1,
        #           inplace=True)
        # print(data)
        # print(json.dumps(ws.market_depth()[int(len(ws.market_depth()) / 2)]))
        # print(json.dumps(ws.recent_trades()[0:5]))


def setup_logger():
    # Prints logger info to terminal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    run()
