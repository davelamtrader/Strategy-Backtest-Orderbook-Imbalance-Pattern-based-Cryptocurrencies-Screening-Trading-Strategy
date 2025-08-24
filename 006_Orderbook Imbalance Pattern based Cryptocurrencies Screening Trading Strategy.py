import pandas as pd
import asyncio
from tardis_client import TardisClient, Channel
import numpy as np
import vectorbt as vbt
import quantstats as qs
from typing import List, Dict

async def fetch_tardis_data(api_key, exchange, symbols, start_date, end_date):
    """
    Asynchronously fetches trades and order book data from Tardis.dev API.
    """
    tardis_client = TardisClient(api_key=api_key)
    
    # Define the data types (channels) to download.
    # 'trades' for OHLCV and 'orderBookL2' for the pressure calculation.
    filters = [
        Channel(name="trades", symbols=symbols),
        Channel(name="orderBookL2", symbols=symbols)
    ]

    print(f"Fetching data for {symbols} from {start_date} to {end_date}...")
    
    # The replay method returns an async generator
    messages_generator = tardis_client.replay(
        exchange=exchange,
        from_date=start_date,
        to_date=end_date,
        filters=filters,
    )

    # Process the streamed messages into lists
    data_rows = []
    async for local_timestamp, message in messages_generator:
        # The message format is exchange-native. We extract relevant fields.
        # For this example, we'll focus on a simplified format.
        msg_type = message.get('table') 
        
        if msg_type == 'trade':
            for trade in message['data']:
                data_rows.append({
                    'timestamp': pd.to_datetime(trade['timestamp']),
                    'symbol': trade['symbol'],
                    'type': 'trade',
                    'side': trade['side'].lower(),
                    'price': trade['price'],
                    'amount': trade['size']
                })
        elif msg_type == 'orderBookL2':
            for book_update in message['data']:
                # For updates, amount represents the new total size at that price level.
                # A size of 0 means the level is removed.
                data_rows.append({
                    'timestamp': pd.to_datetime(local_timestamp),
                    'symbol': book_update['symbol'],
                    'type': 'book',
                    'side': book_update['side'].lower() + 's', # 'bids' or 'asks'
                    'price': book_update.get('price', None), 
                    'amount': book_update.get('size', 0)
                })

    print("Data fetching complete.")
    return pd.DataFrame(data_rows)


class OrderBookProcessor:
    """Processes raw Tardis.dev data into extended order book format per report."""

    def process_raw_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # Filter trades and book updates 
        trades = raw_df[raw_df['type'] == 'trade'].copy()
        book = raw_df[raw_df['type'] == 'book'].copy()

        # Generate daily OHLCV from trades 
        trades['price'] = pd.to_numeric(trades['price'])
        trades['amount'] = pd.to_numeric(trades['amount'])
        daily_ohlcv = trades['price'].resample('1D').ohlc()
        daily_ohlcv['volume'] = trades['amount'].resample('1D').sum()

        # Last hour book for extended order book 
        book['timestamp'] = pd.to_datetime(book['timestamp'])
        book.set_index('timestamp', inplace=True)
        book['price'] = pd.to_numeric(book['price'], errors='coerce')
        book['amount'] = pd.to_numeric(book['amount'])

        # Aggregate unique price levels 
        extended_book = book.dropna(subset=['price']).groupby(['side', 'price'])['amount'].sum().reset_index()
        return {'ohlcv': daily_ohlcv, 'extended_book': extended_book}


class ImbalanceSignalGenerator:
    """Generates order book imbalance signals per report."""

    def generate_signals(self, ohlcv: pd.DataFrame, extended_book: pd.DataFrame) -> pd.DataFrame:
        signals = []
        for day in ohlcv.index:
            close = ohlcv.loc[day, 'close']
            if pd.isna(close):
                continue
            # Separate bids and asks 
            bids = extended_book[extended_book['side'] == 'bids']
            asks = extended_book[extended_book['side'] == 'asks']

            # Calculate weights and pressures 
            epsilon = 1e-9
            bid_weights = abs(close / (bids['price'] - close + epsilon))
            p_buy = (bids['amount'] * bid_weights).sum()
            ask_weights = abs(close / (asks['price'] - close + epsilon))
            p_sell = (asks['amount'] * ask_weights).sum()
            pressure_ratio = np.log(p_buy + epsilon) - np.log(p_sell + epsilon)

            signals.append({'timestamp': day, 'pressure_ratio': pressure_ratio})

        signals_df = pd.DataFrame(signals).set_index('timestamp')
        ohlcv = ohlcv.join(signals_df)
        ohlcv['returns'] = ohlcv['close'].pct_change()
        return ohlcv


class Backtester:
    """Handles backtesting with trading rules and holdings limits."""

    def __init__(self, holdings_limit: int):
        self.holdings_limit = holdings_limit

    def run_backtest(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, vbt.Portfolio]:
        portfolios = {}
        for symbol, df in data_dict.items():
            if df.empty:
                continue
            df['pressure_mean'] = df['pressure_ratio'].rolling(20).mean()
            df['pressure_std'] = df['pressure_ratio'].rolling(20).std()
            buy_pressure_dominant = df['pressure_ratio'] > (df['pressure_mean'] + 1.96 * df['pressure_std'])
            day_was_down = df['returns'] < -0.02  # Per report threshold 
            df['entry'] = buy_pressure_dominant & day_was_down
            df['exit'] = False  # Time-based exit 

            # Simulate selection: limit to top holdings_limit by signal strength)
            # (In full multi-symbol backtest, rank and select top N)
            price = df['close']
            portfolios[symbol] = vbt.Portfolio.from_signals(
                close=price,
                entries=df['entry'],
                exits=df['exit'],
                exit_after=10,
                freq='1D',
                init_cash=100000,
                fees=0.001,
                sl_stop=0.10,
                direction='longonly'
            )
        return portfolios  # Aggregate or evaluate across symbols as needed


class PerformanceEvaluator:

    def evaluate(self, portfolios: Dict[str, vbt.Portfolio], benchmark: pd.Series) -> Dict:
        results = {}
        for symbol, pf in portfolios.items():
            returns = pf.returns()
            benchmark_reindexed = benchmark.reindex(returns.index).fillna(0)
            metrics = {
                'CAGR': qs.stats.cagr(returns),
                'Sharpe': qs.stats.sharpe(returns),
                'Sortino': qs.stats.sortino(returns),
                'Calmar': qs.stats.calmar(returns),
                'Max Drawdown': qs.stats.max_drawdown(returns),
                'Volatility': qs.stats.volatility(returns),
                'Alpha': qs.stats.alpha(returns, benchmark_reindexed),
                'Beta': qs.stats.beta(returns, benchmark_reindexed),
            }
            qs.reports.html(returns, benchmark_reindexed, output=f'{symbol}_report.html')
            results[symbol] = metrics
        return results


# Main execution
def main(symbols: List[str], full_data_dict: Dict[str, pd.DataFrame], benchmark: pd.Series):
    processor = OrderBookProcessor()
    signal_gen = ImbalanceSignalGenerator()

    processed_data = {}
    for symbol in symbols:
        raw_df = full_data_dict.get(symbol, pd.DataFrame())
        processed = processor.process_raw_data(raw_df)
        signals_df = signal_gen.generate_signals(processed['ohlcv'], processed['extended_book'])
        processed_data[symbol] = signals_df

    holdings_limits = [10, 20, 30]
    all_portfolios = {}
    for limit in holdings_limits:
        backtester = Backtester(limit)
        portfolios = backtester.run_backtest(processed_data)
        all_portfolios[limit] = portfolios

    evaluator = PerformanceEvaluator()
    for limit, ports in all_portfolios.items():
        results = evaluator.evaluate(ports, benchmark)
        print(f"Results for {limit} holdings: {results}")


if __name__ == '__main__':

# --- Step 1: Configuration for Data Fetching ---
API_KEY = None 
EXCHANGE = 'binance'
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BNBUSDT', 'SOLUSDT', 'TRXUSDT', 'DOGEUSDT', 'ADAUSDT', 'HYPEUSDT', 'SUIUSDT', 
    'XLMUSDT', 'LINKUSDT', 'BCHUSDT', 'HBARUSDT', 'AVAXUSDT', 'TONUSDT', 'LTCUSDT', 'LEOUSDT', 'SHIBUSDT', 'DOTUSDT', 
    'UNIUSDT', 'XMRUSDT', 'BGBUSDT', 'PEPEUSDT', 'CROUSDT', 'AAVEUSDT', 'ENAUSDT', 'TAOUSDT', 'ETCUSDT', 'NEARUSDT', 
    'ONDOUSDT', 'APTUSDT', 'PIUSDT', 'OKBUSDT', 'ICPUSDT', 'MNTUSDT', 'KASUSDT', 'PENGUUSDT', 'POLUSDT', 'GTUSDT', 
    'ALGOUSDT', 'BONKUSDT', 'VETUSDT', 'ARBUSDT', 'IPUSDT', 'RENDERUSDT', 'WLDUSDT', 'TRUMPUSDT', 'SKYUSDT', 'ATOMUSDT', 
    'SEIUSDT', 'FLRUSDT', 'FILUSDT', 'XDCUSDT', 'FETUSDT', 'SPXUSDT', 'FORMUSDT', 'JUPUSDT', 'KCSUSDT', 'QNTUSDT', 
    'INJUSDT', 'CRVUSDT', 'STXUSDT', 'TIAUSDT', 'OPUSDT', 'CFXUSDT', 'PUMPUSDT', 'PYUSDUSDT', 'FLOKIUSDT', 'PAXGUSDT', 
    'GRTUSDT', 'FARTCOINUSDT', 'IMXUSDT', 'ENSUSDT', 'CAKEUSDT', 'WIFUSDT', 'SUSDT', 'NEXOUSDT', 'KAIAUSDT', 'LDOUSDT', 
    'XTZUSDT', 'VIRTUALUSDT', 'AUSDT', 'THETAUSDT', 'JASMYUSDT', 'IOTAUSDT', 'MUSDT', 'RAYUSDT', 'GALAUSDT', 'SANDUSDT', 
    'AEROUSDT', 'PENDLEUSDT', 'BTTUSDT', 'PYTHUSDT', 'JTOUSDT', 'FLOWUSDT', 'MANAUSDT', 'BRETTUSDT', 'DYDXUSDT', 'ZKUSDT']

START_DATE = '2023-71-01'
END_DATE = '2025-06-30' # A few days for demonstration

# Run the async data fetching function
full_data_df = await fetch_tardis_data(API_KEY, EXCHANGE, SYMBOLS, START_DATE, END_DATE)
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

full_data_df = loop.run_until_complete(fetch_tardis_data(API_KEY, EXCHANGE, SYMBOLS, START_DATE, END_DATE))
print("\n--- Sample of Fetched Data ---")
print(full_data_df.head())

# Disable vectorbt warnings for cleaner output
vbt.settings.warnings['silenced'] = True

# Example usage (assuming symbols list and full_data_dict provided)
symbols = SYMBOLS
full_data_dict = {}  # Assume pre-fetched using Tardis.dev client
benchmark = pd.Series()  # Assume BTC or similar
main(symbols, full_data_dict, benchmark)



