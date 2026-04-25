# Portfolio_rebalancer

A Python-based software prototype for stock portfolio rebalancing based on technical indicators, genetic algorithms, multi-objective optimization, backtesting, and risk-aware performance evaluation.

## Description

Portfolio_Rebalancer is a Python-based system for analyzing financial time series and building a dynamic stock portfolio rebalancing strategy.

The project focuses on creating a full portfolio management pipeline: loading OHLCV market data, calculating and normalizing technical indicators, estimating asset attractiveness, optimizing portfolio weights, performing scheduled rebalancing, accounting for transaction costs, and evaluating the final strategy using financial and risk metrics.

The system is designed as a reproducible research prototype with a modular architecture and a graphical user interface.

## Source Code

- [`NIR7_V2.py`](NIR7_V2.py) — main Python implementation of the portfolio rebalancing system

The source code includes the complete application logic: OHLCV data loading, local caching, technical indicator calculation, asset-level genetic optimization, portfolio-level multi-objective optimization, backtesting, metric calculation, visualization, and GUI interaction.

## Key Features

- OHLCV data loading and preprocessing
- Local data and ticker caching
- Technical indicator calculation and normalization
- Asset scoring based on indicator signals
- Portfolio rebalancing with user-defined constraints
- Multi-objective portfolio optimization
- Genetic algorithm-based optimization
- Lightweight NSGA-II-style portfolio optimization
- Transaction cost and turnover accounting
- Historical backtesting
- Risk and return metric calculation
- GUI for experiment setup and result visualization
- Export of calculated results

## Optimization Approach

The project uses a two-level optimization approach:

1. **Asset-level optimization**  
   Technical indicator weights are calibrated for each asset to estimate its attractiveness score.

2. **Portfolio-level optimization**  
   Portfolio weights are selected during rebalancing using multi-objective optimization.

The optimization process considers several conflicting goals:

- expected return;
- portfolio risk;
- diversification;
- portfolio turnover;
- transaction costs;
- position limits.

## Metrics

The system evaluates portfolio performance using metrics such as:

- Total Return
- CAGR
- Volatility
- Sharpe Ratio
- Calmar Ratio
- Maximum Drawdown
- Turnover
- Transaction Costs
- Portfolio Concentration

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PyTorch
- Tkinter / ttkbootstrap
- Genetic Algorithms
- Technical Analysis
- Backtesting
- Multi-objective Optimization
- NSGA-II-style optimization

## Project Materials

- [Source Code](NIR7_V2.py)
- [Research Report DOCX](КовальскийИВ%20ПЗ%20НИР%207%20семестр.docx)
- [Alternative Report DOCX](Ковальский%20ПЗ%20НИР%207%20семестр.docx)
- [Presentation PDF](КовальскийИВ_Б22-514_Презентация_УИР%207%20семестр.pdf)

## Research Topic

**Development of a software system for stock portfolio rebalancing based on technical indicators**

## Results

The developed prototype was tested on historical stock market data.  
The system demonstrated improved risk control compared to a basic buy-and-hold strategy, including a lower maximum drawdown and a better risk-adjusted performance profile.

## Repository Purpose

This repository contains materials and implementation related to a student research project in financial time series analysis, algorithmic trading, portfolio optimization, and software system design.
