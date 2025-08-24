## Overview

This repository showcases a quantitative trading strategy designed to extract alpha by leveraging high-frequency market microstructure data, specifically order book information, to generate actionable low-frequency trading signals. The strategy focuses on identifying periods of significant order book imbalance and correlating them with price movements to predict potential market direction, particularly within the cryptocurrency markets.

## Strategy Concept

The core innovation lies in bridging the gap between high-frequency trading data and low-frequency investment strategies. The approach recognizes that while pure high-frequency data might not directly guide longer-term investment decisions, its unique information can be transformed into valuable signals for lower-frequency analysis. This involves using high-frequency price changes as a "probe" to gather information about intraday market dynamics.

### Harnessing High-Frequency Data for Low-Frequency Signals

The strategy processes raw tick-level data, including trades and order book updates, to construct a richer dataset. By analyzing the "long order data" derived from aggregated order book states, the strategy aims to achieve a more comprehensive view of market sentiment than traditional daily data provides.

### Order Book Dynamics and Market Pressure

The state of the order book, characterized by the distribution of buy (bid) and sell (ask) orders at various price levels, provides crucial insights into participant expectations and potential future price movements. A substantial volume of buy orders below the current price indicates strong support, while a large number of sell orders above the current price signifies selling pressure.

The strategy quantifies this pressure by calculating a "pressure ratio" derived from the weighted buy and sell orders. This ratio, calculated as the logarithmic difference between buy pressure and sell pressure, serves as a primary signal indicator.

## Data Acquisition and Processing Pipeline

### Data Source

Data is sourced from the Tardis.dev API, providing access to historical cryptocurrency market data, including trades and orderBookL2 (Level 2 order book) information.

### Data Fetching

An asynchronous Python script utilizes the tardis_client library to efficiently retrieve data for specified cryptocurrency symbols and date ranges. The fetched data includes timestamps, symbols, trade/order book types, sides (bids/asks), prices, and amounts.

### Data Processing

The raw data is processed through a pipeline that includes:

1.  **Order Book Processing**: The OrderBookProcessor class transforms raw data into:
    -   Daily OHLCV (Open, High, Low, Close, Volume) data derived from trade records.
    -   An "extended order book" representation, which aggregates buy and sell order amounts at unique price levels.
2.  **Signal Generation**: The ImbalanceSignalGenerator calculates the key "pressure ratio" using the processed order book data. This ratio is then combined with daily returns to identify trading opportunities.

## Strategy Implementation and Backtesting

### Signal Generation Logic

The strategy generates trading signals based on the calculated pressure_ratio and daily price movements:

-   **Buy Pressure Dominance**: A signal is triggered when the pressure_ratio indicates significantly more buying pressure than selling pressure, typically identified by exceeding a threshold derived from its rolling mean and standard deviation (e.g., pressure_ratio > (pressure_mean + 1.96 * pressure_std)).
-   **Down Day Confirmation**: The strategy confirms buy signals on days that experienced a significant price decline (e.g., returns < -2%).
-   **Entry Condition**: A buy entry is initiated when both buy pressure dominance and the down day condition are met.

### Backtesting Framework

The strategy is rigorously backtested using the vectorbt library. The Backtester class manages the simulation, incorporating parameters such as:

-   **Holdings Limit**: The maximum number of concurrent positions (tested with 10, 20, 30).
-   **Exit Strategy**: Positions are exited based on a time-based rule (exit_after=10 days) and potential stop-loss mechanisms.
-   **Transaction Costs**: Includes fees (fees=0.001) and a stop-loss (sl_stop=0.10).
-   **Direction**: The strategy is designed for longonly positions.

### Performance Evaluation

Portfolio performance is evaluated using the quantstats library, providing key metrics such as:

-   Compound Annual Growth Rate (CAGR)
-   Sharpe Ratio
-   Sortino Ratio
-   Calmar Ratio
-   Maximum Drawdown
-   Volatility
-   Alpha and Beta (relative to a benchmark)  
    Detailed HTML reports are generated for each evaluated portfolio.

## Code Structure

The project is organized into modular Python classes:

-   fetch_tardis_data: Handles asynchronous data fetching.
-   OrderBookProcessor: Preprocesses raw trade and order book data.
-   ImbalanceSignalGenerator: Generates trading signals from processed data.
-   Backtester: Executes the backtesting simulation.
-   PerformanceEvaluator: Analyzes and reports on strategy performance.

The main function orchestrates these components, allowing for configuration of symbols, date ranges, and testing parameters.

## Configuration

The system is configured to fetch data for a comprehensive list of cryptocurrency pairs from the Binance exchange. The default date range for data analysis spans from July 1, 2023, to June 30, 2025.  
_Note: The specified START_DATE in the provided code ('2023-71-01') appears to contain a typo and should likely be '2023-07-01' for accurate date parsing._

## Conclusion

This strategy represents a sophisticated application of market microstructure analysis to quantitative trading. By transforming high-frequency order book dynamics into interpretable signals, it aims to capture market inefficiencies and deliver robust, risk-adjusted returns. The systematic approach to data processing, signal generation, and backtesting ensures a thorough evaluation of the strategy's efficacy and potential for live deployment.

## Further Enhancements

-   **Parameter Optimization**: Systematically optimize rolling window sizes, Z-score thresholds, and exit parameters.
-   **Multi-Asset Integration**: Extend the strategy to a wider range of assets or incorporate cross-asset correlations.
-   **Machine Learning Integration**: Utilize machine learning models for enhanced feature engineering or direct signal prediction from order book data.
-   **Live Trading**: Adapt the system for real-time data feeds and automated order execution.
