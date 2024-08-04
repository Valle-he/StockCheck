import streamlit as st

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from fredapi import Fred
import plotly.express as px
from scipy.optimize import minimize

# Alpha Vantage API Key for News Sentiment
alpha_vantage_api_key = '7ULSSVM1DNTM3E4C'

# Funktion zur Berechnung der aktuellen Dividendenrendite
def get_dividend_yield(ticker):
    try:
        stock = yf.Ticker(ticker)
        dividend_yield = stock.info.get('dividendYield', None)
        if dividend_yield is None:
            st.warning(f"No dividend yield available for {ticker}")
            return None
        return dividend_yield
    except Exception as e:
        st.error(f"Error fetching dividend yield for {ticker}: {str(e)}")
        return None


# Funktion zur Berechnung des Peter Lynch Valuation Scores
def calculate_peter_lynch_score(ticker, growth_rate):
    dividend_yield = get_dividend_yield(ticker)
    if dividend_yield is None or dividend_yield <= 0:
        st.warning(f"Invalid dividend yield for {ticker}")
        return None

    stock = yf.Ticker(ticker)
    pe_ratio = stock.info.get('trailingPE', None)
    if pe_ratio is None:
        st.warning(f"P/E ratio not available for {ticker}")
        return None

    try:
        peter_lynch_score = (growth_rate * 100) / (pe_ratio * dividend_yield * 100)
    except ZeroDivisionError:
        st.error("Division by zero encountered in Peter Lynch score calculation")
        return None

    return peter_lynch_score

# Funktion zur Berechnung des Fair Value nach Graham
def calculate_graham_valuation(ticker, growth_rate):
    # EPS abrufen
    stock = yf.Ticker(ticker)
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None  # Wenn EPS nicht verfügbar, kein Fair Value berechnen
    
    # Risk-Free Rate über FRED API abrufen
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    
    # Fair Value nach Graham Formel berechnen
    graham_valuation = (eps * (8.5 + (2 * growth_rate) * 100) * 4.4) / (risk_free_rate * 100)
    
    return graham_valuation

# Funktion zur Berechnung des Fair Value nach Formel
def calculate_formula_valuation(ticker, growth_rate):
    # Forward P/E Ratio abrufen
    stock = yf.Ticker(ticker)
    forward_pe_ratio = stock.info.get('forwardPE', None)
    
    if forward_pe_ratio is None:
        return None  # Wenn Forward P/E Ratio nicht verfügbar, kein Fair Value berechnen
    
    # EPS abrufen
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None  # Wenn EPS nicht verfügbar, kein Fair Value berechnen
    
    # Durchschnittlicher Markt Return
    sp500 = yf.Ticker('^GSPC').history(period='5y')
    average_market_return = sp500['Close'].pct_change().mean() * 252
    
    # Fair Value nach Formel berechnen
    formula_valuation = (forward_pe_ratio * eps * ((1 + growth_rate) ** 5)) / ((1 + average_market_return) ** 5)
    
    return formula_valuation

# Funktion zur Berechnung des Expected Return (fundamental)
def calculate_expected_return(ticker, growth_rate):
    # EPS abrufen
    stock = yf.Ticker(ticker)
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None  # Wenn EPS nicht verfügbar, kein Expected Return berechnen
    
    # Gewinn in 5 Jahren berechnen (Extrapolationszeitraum festgelegt auf 5 Jahre)
    future_eps = eps * ((1 + growth_rate) ** 5)
    
    # Programmierter Preis der Aktie in 5 Jahren (prog Kgv = Forward P/E Ratio)
    forward_pe_ratio = stock.info.get('forwardPE', None)
    
    if forward_pe_ratio is None:
        return None  # Wenn Forward P/E Ratio nicht verfügbar, kein Expected Return berechnen
    
    future_stock_price = forward_pe_ratio * future_eps
    
    # Aktueller Preis der Aktie
    current_stock_price = stock.history(period='1d')['Close'].iloc[-1]
    
    # Expected Return berechnen
    expected_return = ((future_stock_price / current_stock_price) ** (1 / 5) - 1) 
    
    return expected_return

# Funktion zur Berechnung des Expected Return (historical)
def calculate_expected_return_historical(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    
    # Daten von Yahoo Finance abrufen
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Log-Renditen berechnen
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    # Historischen Expected Return berechnen
    historical_return = log_returns.mean() * 252 # in Prozent umrechnen
    
    return historical_return

# Function to fetch news data using Alpha Vantage News API
def get_news_data(ticker):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_vantage_api_key}'
    response = requests.get(url)
    try:
        response.raise_for_status()
        news_data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news data: {str(e)}")
        return []
    articles = news_data.get('feed', [])
    news_list = []
    for article in articles:
        published_at = article['time_published']
        title = article['title']
        description = article['summary']
        news_list.append([published_at, title, description])
    return news_list

# Function to analyze sentiment using TextBlob
def analyze_sentiment(news_data):
    sentiments = []
    for entry in news_data:
        title = entry[1]
        sentiment_score = TextBlob(title).sentiment.polarity
        sentiments.append(sentiment_score)
    return sentiments


# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y')
    return hist

# Function to analyze stock based on ticker symbol
def analyze_stock(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = fetch_stock_data(ticker)
    
    # Calculate volatility
    hist['Log Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = hist['Log Return'].std() * np.sqrt(252)
    
    # Calculate Max Drawdown
    hist['Cumulative Return'] = (1 + hist['Log Return']).cumprod()
    hist['Cumulative Max'] = hist['Cumulative Return'].cummax()
    hist['Drawdown'] = hist['Cumulative Return'] / hist['Cumulative Max'] - 1
    max_drawdown = hist['Drawdown'].min()
    
    # Calculate Beta (using S&P 500 as a benchmark)
    sp500 = yf.Ticker('^GSPC').history(period='5y')
    sp500['Log Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
    # Example alignment (adjust according to your data)
    common_index = hist['Log Return'].index.intersection(sp500['Log Return'].index)
    hist_aligned = hist.loc[common_index, 'Log Return'].dropna()
    sp500_aligned = sp500.loc[common_index, 'Log Return'].dropna()

# Calculate covariance
    if len(hist_aligned) != len(sp500_aligned):
        return {"Error": "Data lengths do not match"}

        # Calculate covariance and beta
    try:
        covariance = np.cov(hist_aligned, sp500_aligned)[0, 1]
        beta = covariance / sp500_aligned.var()
    except Exception as e:
        covariance = "N/A"
        beta = "N/A"
    
    # Calculate correlation with market (S&P 500)
    correlation = hist['Log Return'].corr(sp500['Log Return'])
    
    # Function to calculate Cost of Equity
    def calculate_cost_of_equity(risk_free_rate, beta, average_market_return):
        cost_of_equity = risk_free_rate + beta * (average_market_return - risk_free_rate)
        return cost_of_equity

    # Use FRED API to get current 10-year Treasury rate
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]

    # Calculate average market return (you may need to adjust this calculation based on your data)
    # Example: Using S&P 500 index return as average market return
    average_market_return = sp500['Log Return'].mean() * 252

    # Calculate Cost of Equity
    cost_of_equity = calculate_cost_of_equity(risk_free_rate, beta, average_market_return)
    
    # Calculate valuation metrics
    growth_rate = 0.10
    peter_lynch_score = calculate_peter_lynch_score(ticker, growth_rate)
    graham_valuation = calculate_graham_valuation(ticker, growth_rate)
    formula_valuation = calculate_formula_valuation(ticker, growth_rate)
    expected_return = calculate_expected_return(ticker, growth_rate)
    historical_expected_return = calculate_expected_return_historical(ticker)
    
    analysis = {
        'Ticker': ticker,
        'P/E Ratio': info.get('trailingPE'),
        'Forward P/E': info.get('forwardPE'),
        'P/S Ratio': info.get('priceToSalesTrailing12Months'),
        'P/B Ratio': info.get('priceToBook'),
        'Dividend Yield': info.get('dividendYield'),
        'Trailing Eps': info.get('trailingEps'),
        'Target Price': info.get('targetMeanPrice'),
        'Sector': info.get('sector'),
        'Industry': info.get('industry'),
        'Full Time Employees': info.get('fullTimeEmployees'),
        'City': info.get('city'),
        'State': info.get('state'),
        'Country': info.get('country'),
        'Website': info.get('website'),
        'Market Cap (Billion $)': info.get('marketCap') / 1e9 if info.get('marketCap') else None,
        'Enterprise Value (Billion $)': info.get('enterpriseValue') / 1e9 if info.get('enterpriseValue') else None,
        'Enterprise to Revenue': info.get('enterpriseToRevenue'),
        'Enterprise to EBITDA': info.get('enterpriseToEbitda'),
        'Profit Margins': info.get('profitMargins'),
        'Gross Margins': info.get('grossMargins'),
        'EBITDA Margins': info.get('ebitdaMargins'),
        'Operating Margins': info.get('operatingMargins'),
        'Return on Assets (ROA)': info.get('returnOnAssets'),
        'Return on Equity (ROE)': info.get('returnOnEquity'),
        'Revenue Growth': info.get('revenueGrowth'),
        'Payout Ratio': info.get('payoutRatio'),
        'Total Cash (Million $)': info.get('totalCash') / 1e6 if info.get('totalCash') else None,
        'Total Debt (Million $)': info.get('totalDebt') / 1e6 if info.get('totalDebt') else None,
        'Total Revenue (Million $)': info.get('totalRevenue') / 1e6 if info.get('totalRevenue') else None,
        'Gross Profits': info.get('grossProfits'),
        'Total Revenue per Share': info.get('totalRevenuePerShare'),
        'Debt to Equity Ratio': info.get('debtToEquity'),
        'Current Ratio': info.get('currentRatio'),
        'Operating Cashflow (Million $)': info.get('operatingCashflow') / 1e6 if info.get('operatingCashflow') else None,
        'Levered Free Cashflow (Million $)': info.get('leveredFreeCashflow') / 1e6 if info.get('leveredFreeCashflow') else None,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Beta': beta,
        'Market Correlation': correlation,
        'Cost of Equity': cost_of_equity,
        'Peter Lynch Score': peter_lynch_score,
        'Graham Valuation': graham_valuation,
        'Formula Valuation': formula_valuation,
        'Expected Return (Fundamental)': expected_return,
        'Historical Expected Return': historical_expected_return,
        'Historical Prices': hist,
        'Quick Ratio': info.get('quickRatio'),
        'Total Revenue per Share': info.get('totalRevenuePerShare'),
        'Total Cash Per Share': info.get('totalCashPerShare'),
        'Free Cashflow': info.get('freeCashflow'),
        
    
'Audit Risk': info.get('auditRisk'),
'Board Risk': info.get('boardRisk'),
'Compensation Risk': info.get('compensationRisk'),
'Shareholder Rights Risk': info.get('shareHolderRightsRisk'),
'Overall Risk': info.get('overallRisk'),

'Price Hint': info.get('priceHint'),
'Previous Close': info.get('previousClose'),
'Open': info.get('open'),
'Day Low': info.get('dayLow'),
'Day High': info.get('dayHigh'),
'Regular Market Previous Close': info.get('regularMarketPreviousClose'),
'Regular Market Open': info.get('regularMarketOpen'),
'Regular Market Day Low': info.get('regularMarketDayLow'),
'Regular Market Day High': info.get('regularMarketDayHigh'),



'Ex-Dividend Date': info.get('exDividendDate'),
'Five Year Average Dividend Yield': info.get('fiveYearAvgDividendYield'),

'Volume': info.get('volume'),
'Regular Market Volume': info.get('regularMarketVolume'),
'Average Volume': info.get('averageVolume'),
'Average Volume 10 Days': info.get('averageVolume10days'),
'Average Daily Volume 10 Day': info.get('averageDailyVolume10Day'),
'Bid': info.get('bid'),
'Ask': info.get('ask'),
'Bid Size': info.get('bidSize'),
'Ask Size': info.get('askSize'),

'Fifty-Two Week Low': info.get('fiftyTwoWeekLow'),
'Fifty-Two Week High': info.get('fiftyTwoWeekHigh'),
'P/S Ratio Trailing 12 Months': info.get('priceToSalesTrailing12Months'),
'Fifty Day Average': info.get('fiftyDayAverage'),
'Two Hundred Day Average': info.get('twoHundredDayAverage'),
'Currency': info.get('currency'),

'Float Shares': info.get('floatShares'),
'Shares Outstanding': info.get('sharesOutstanding'),
'Shares Short': info.get('sharesShort'),
'Shares Short Prior Month': info.get('sharesShortPriorMonth'),
'Shares Short Previous Month Date': info.get('sharesShortPreviousMonthDate'),
'Date Short Interest': info.get('dateShortInterest'),
'Shares Percent Shares Out': info.get('sharesPercentSharesOut'),
'Held Percent Insiders': info.get('heldPercentInsiders'),
'Held Percent Institutions': info.get('heldPercentInstitutions'),
'Short Ratio': info.get('shortRatio'),
'Short Percent of Float': info.get('shortPercentOfFloat'),
'Implied Shares Outstanding': info.get('impliedSharesOutstanding'),

'Book Value': info.get('bookValue'),


'Trailing EPS': info.get('trailingEps'),
'Forward EPS': info.get('forwardEps'),
'PEG Ratio': info.get('pegRatio'),
'Last Split Factor': info.get('lastSplitFactor'),

'Enterprise to Revenue': info.get('enterpriseToRevenue'),
'Enterprise to EBITDA': info.get('enterpriseToEbitda'),
'52 Week Change': info.get('52WeekChange'),
'S&P 52 Week Change': info.get('SandP52WeekChange'),
'Last Dividend Value': info.get('lastDividendValue'),
'Last Dividend Date': info.get('lastDividendDate'),
'Exchange': info.get('exchange'),


'Current Price': info.get('currentPrice'),
'Target High Price': info.get('targetHighPrice'),
'Target Low Price': info.get('targetLowPrice'),
'Target Mean Price': info.get('targetMeanPrice'),
'Target Median Price': info.get('targetMedianPrice'),
'Recommendation Mean': info.get('recommendationMean'),
'Recommendation Key': info.get('recommendationKey'),
'Number of Analyst Opinions': info.get('numberOfAnalystOpinions'),

'Total Cash Per Share': info.get('totalCashPerShare'),
'EBITDA': info.get('ebitda'),



'Total Revenue': info.get('totalRevenue'),
'Free Cashflow': info.get('freeCashflow'),

'EBITDA Margins': info.get('ebitdaMargins'),
'Operating Margins': info.get('operatingMargins'),


    }

    
    return analysis

# Function for portfolio optimization
def optimize_portfolio(tickers, min_weight, max_weight):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)

    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            adj_close_df[ticker] = data['Adj Close']
        except Exception as e:
            st.error(f"Error downloading data for {ticker}: {str(e)}")
            return None, None, None, None, None

    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    try:
        fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
        risk_free_rate = ten_year_treasury_rate.iloc[-1]
    except Exception as e:
        st.error(f"Error fetching risk-free rate: {str(e)}")
        return None, None, None, None, None

    num_assets = len(tickers)
    results = np.zeros((3, 10000))
    weight_array = np.zeros((10000, num_assets))

    def objective(weights):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(min_weight / 100, max_weight / 100)] * num_assets

    try:
        optimized = minimize(objective, num_assets * [1. / num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized['x']
        optimal_portfolio_return = expected_return(optimal_weights, log_returns)
        optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
        optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
    except Exception as e:
        st.error(f"Error optimizing portfolio: {str(e)}")
        return None, None, None, None, None

    return optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio, adj_close_df


# Streamlit App
st.title('Stock and Portfolio Analysis')

# Sidebar for Stock Analysis Input
st.sidebar.title('Stock and Portfolio Analysis')

st.sidebar.header('Stock Analysis Input')
ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
if not ticker.isalpha():
    st.error("Invalid ticker symbol. Please enter a valid stock ticker.")


if st.sidebar.button("Analyze Stock"):
    if ticker:
        try:
            result = analyze_stock(ticker)
            if 'Error' in result:
                st.error(result['Error'])
            else:
                st.header(f'Stock Analysis Results for {ticker}')
                st.subheader("Summary")
                st.markdown("---")
                # Sort and group ratios by type, display analysis...
        except Exception as e:
            st.error(f"Error analyzing stock: {str(e)}")
        
        # Sort and group ratios by type
        grouped_ratios = {
            
            'Company Information':['Sector','Industry','Full Time Employees','City','State','Country','Website'],
            'Valuation Ratios': ['P/E Ratio', 'Forward P/E', 'P/S Ratio','P/S Ratio Trailing 12 Months', 'P/B Ratio','Book Value','PEG Ratio'],
            'Financial Ratios': [ 'Trailing Eps','Forward EPS', 'Payout Ratio'],
            'Profitability Margins': ['Profit Margins', 'Gross Margins', 'EBITDA Margins', 'Operating Margins'],
            'Financial Metrics': ['Return on Assets (ROA)', 'Return on Equity (ROE)'],
            'Revenue Metrics': ['Revenue Growth', 'Total Revenue (Million $)', 'Total Revenue per Share','EBITDA'],
            'Financial Health': ['Debt to Equity Ratio', 'Current Ratio','Quick Ratio'],
            'Cashflow Metrics': ['Total Cash (Million $)','Total Cash per Share', 'Operating Cashflow (Million $)', 'Levered Free Cashflow (Million $)','Free Cash Flow'],
            'Dividend Ratios': ['Dividend Yield', 'Five Year Average Dividend Yield', 'Last Dividend Value'],
            'Share Metrics': ['Float Shares', 'Shares Outstanding', 'Shares Short', 'Shares Short Prior Month', 'Shares Short Previous Month Date', 'Date Short Interest', 'Shares Percent Shares Out', 'Held Percent Insiders', 'Held Percent Institutions', 'Short Ratio', 'Short Percent of Float', 'Implied Shares Outstanding'],
            'Market Metrics':['Market Cap (Billion $)','Enterprise Value (Billion $)','Enterprise to Revenue','Enterprise to EBITDA','Cost of Equity'],          
            'Fraud Risk Metrics': ['Audit Risk', 'Board Risk', 'Compensation Risk', 'Shareholder Rights Risk', 'Overall Risk'],
            'Company Risk Management Metrics':['Volatility','Max Drawdown','Beta','Market Correlation'],
            'Fair Value Metrics':['Peter Lynch Score','Graham Valuation','Formula Valuation','Expected Return (Fundamental)','Historical Expected Return'],
            'Price Forecasts': ['Current Price', 'Target High Price', 'Target Low Price', 'Target Mean Price', 'Target Median Price', 'Recommendation Mean', 'Recommendation Key', 'Number of Analyst Opinions'],
            'Price Analyses': ['Price Hint', 'Previous Close', 'Open', 'Day Low', 'Day High', 'Regular Market Previous Close', 'Regular Market Open', 'Regular Market Day Low', 'Regular Market Day High'],
            'Trading and Technical Analyses': ['Volume', 'Regular Market Volume', 'Average Volume', 'Average Volume 10 Days', 'Average Daily Volume 10 Day', 'Bid', 'Ask', 'Bid Size', 'Ask Size', 'Fifty-Two Week Low', 'Fifty-Two Week High', 'Fifty Day Average', 'Two Hundred Day Average', 'Currency'],
           
            
           }  


        
        for group_name, ratios in grouped_ratios.items():
            st.subheader(group_name)
            for ratio in ratios:
                if ratio in result and result[ratio] is not None:
                    st.write(f"**{ratio}**: {result[ratio]}")
                else:
                    st.write(f"**{ratio}**: N/A")
            st.write("---")
        
        
        # Display current and historical closing prices
        st.subheader(f'Current and Historical Closing Prices for {ticker}')
        if not result['Historical Prices']['Close'].empty:
            st.write(f"**Current Price**: {result['Historical Prices']['Close'][-1]}")
        else:
           st.write("**Current Price**: Data not available")
        st.line_chart(result['Historical Prices']['Close'])

        # Calculate news sentiment
        try:
            news_data = get_news_data(ticker)
            # Analyze sentiment
            sentiments = analyze_sentiment(news_data)
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments)

            st.subheader('News Sentiment Analysis')
            st.write(f"Average Sentiment for {ticker}: {avg_sentiment:.2f}")

            # Display news articles
            st.subheader('Latest News Articles')
            for article in news_data[:5]:  # Displaying only the first 5 articles
                st.write(f"**Published At**: {article[0]}")
                st.write(f"**Title**: {article[1]}")
                st.write(f"**Summary**: {article[2]}")
                st.write('---')

        except Exception as e:
            st.error(f"Error fetching news data: {str(e)}")

# Sidebar for Portfolio Optimization Input
# Sidebar for Portfolio Optimization Input
st.sidebar.header('Portfolio Optimization Input')
tickers_input = st.sidebar.text_input("Enter the stock tickers separated by commas (e.g., AAPL,GME,SAP,TSLA):", "AAPL,GME,SAP,TSLA")
tickers = [ticker.strip() for ticker in tickers_input.split(',')]

min_weight = st.sidebar.slider('Minimum Weight (%)', min_value=0, max_value=100, value=5)
max_weight = st.sidebar.slider('Maximum Weight (%)', min_value=0, max_value=100, value=30)

if st.sidebar.button("Optimize Portfolio"):
    if not tickers:
        st.error("Please enter at least one valid stock ticker.")
    elif min_weight > max_weight:
        st.error("Minimum weight should be less than or equal to maximum weight.")
    elif min_weight > (100 / len(tickers)):
        st.error(f"Minimum weight should be less than or equal to {100 / len(tickers):.2f}% (1/number of stocks).")
    else:
        try:
            optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio, adj_close_df = optimize_portfolio(tickers, min_weight, max_weight)
            if optimal_weights is None:
                st.error("Error optimizing portfolio.")
            else:
                st.header("Optimal Portfolio Metrics")
                st.subheader("Summary")
                st.markdown("---")
                st.write(f"Expected Annual Return: {optimal_portfolio_return*100:.2f}%")
                st.write(f"Expected Portfolio Volatility: {optimal_portfolio_volatility*100:.2f}%")
                st.write(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

                st.subheader("Optimal Weights:")
                optimal_weights_df = pd.DataFrame(optimal_weights, index=tickers, columns=["Weight"])
                st.write(optimal_weights_df)

                st.subheader('Current and Historical Closing Prices for Optimized Portfolio')
                optimized_portfolio_prices = (adj_close_df * optimal_weights).sum(axis=1)
                st.line_chart(optimized_portfolio_prices)

                fig = px.pie(optimal_weights_df, values='Weight', names=optimal_weights_df.index, title='Asset Allocation')
                st.plotly_chart(fig)

                
        except Exception as e:
            st.error(f"Error optimizing portfolio: {str(e)}")

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, date
from scipy.optimize import minimize
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go

# Funktionen für die Streamlit-App

# Portfolio-Daten erfassen (in Seitenleiste)
def get_portfolio_data():
    portfolio = []
    
    num_assets = st.sidebar.number_input("Wie viele Aktien möchten Sie hinzufügen?", min_value=1, max_value=10, value=1)
    
    for i in range(num_assets):
        ticker = st.sidebar.text_input(f"Bitte geben Sie das Aktien-Ticker-Symbol für Aktie {i+1} ein:")
        date_str = st.sidebar.date_input(f"Bitte geben Sie das Datum ein (YYYY-MM-DD), seit dem Sie investiert sind für Aktie {i+1}:", value=datetime.today() - timedelta(days=365))
        investment_amount = st.sidebar.number_input(f"Bitte geben Sie die Investmentsumme für Aktie {i+1} ein:", min_value=0.0, value=1000.0)
        
        portfolio.append({
            "ticker": ticker,
            "investment_date": date_str,
            "investment_amount": investment_amount
        })
    
    if st.sidebar.button("Track Portfolio"):
        portfolio = fetch_historical_data(portfolio)
        total_value, portfolio_return, total_unrealized, current_volatility, average_volatility, sharpe_ratio, portfolio_expected_return, avg_annual_return, portfolio_values, total_investment = calculate_portfolio_metrics(portfolio)

        # Ergebnisse auf der Hauptseite anzeigen
        st.header("Portfolio Performance Metrics")
        st.subheader("Summary")
        st.markdown("---")
        st.write(f"**Total Portfolio Value:** ${total_value:,.2f}")
        st.write(f"**Unrealized Gains/Losses:** ${total_unrealized:,.2f}")
        st.markdown(" ")
        st.write(f"**Total Portfolio Return:** {portfolio_return * 100:.2f}%")
        st.write(f"**Average Portfolio Return (p.a.):** {avg_annual_return * 100:.2f}%")
        st.write(f"**Expected Portfolio Return (p.a.):** {portfolio_expected_return * 100:.2f}%")
        st.markdown(" ")
        st.write(f"**Current Portfolio Volatility:** {current_volatility * 100:.2f}%")
        st.write(f"**Average Portfolio Volatility:** {average_volatility * 100:.2f}%")
        st.markdown(" ")
        st.write(f"**Current Portfolio Sharpe Ratio:** {sharpe_ratio:.2f}")

        # Grafiken anzeigen
        st.header("Portfolio Performance Charts")
        plot_portfolio_performance(portfolio_values, total_investment)
        plot_asset_allocation(portfolio)

# Historische Daten abrufen
def fetch_historical_data(portfolio):
    end_date = datetime.today().strftime('%Y-%m-%d')
    for stock in portfolio:
        data = yf.download(stock['ticker'], start=stock['investment_date'].strftime('%Y-%m-%d'), end=end_date)
        stock['data'] = data
    return portfolio

# Portfolio-Metriken berechnen
def calculate_portfolio_metrics(portfolio):
    total_investment = 0
    total_value = 0

    portfolio_values = pd.DataFrame()

    start_date = min(stock['investment_date'] for stock in portfolio)
    end_date = datetime.today().strftime('%Y-%m-%d')

    for stock in portfolio:
        initial_price = stock['data']['Adj Close'].iloc[0]
        quantity = stock['investment_amount'] / initial_price

        stock['data']['Position Value'] = stock['data']['Adj Close'] * quantity

        if portfolio_values.empty:
            portfolio_values = stock['data'][['Position Value']].copy()
            portfolio_values.rename(columns={'Position Value': stock['ticker']}, inplace=True)
        else:
            portfolio_values = portfolio_values.join(stock['data'][['Position Value']].rename(columns={'Position Value': stock['ticker']}), how='outer')

        total_investment += stock['investment_amount']
        current_price = stock['data']['Adj Close'].iloc[-1]
        current_value = quantity * current_price
        stock['current_value'] = current_value
        total_value += current_value

    portfolio_values.fillna(0, inplace=True)
    portfolio_values['Total'] = portfolio_values.sum(axis=1)

    total_unrealized = total_value - total_investment
    portfolio_return = (total_value - total_investment) / total_investment
    daily_returns = portfolio_values['Total'].pct_change().dropna()
    current_volatility = np.std(daily_returns) * np.sqrt(252)
    average_volatility = np.mean(np.std(daily_returns) * np.sqrt(252))

    days_held = (datetime.today().date() - start_date).days
    years_held = days_held / 365.25
    avg_annual_return = ((total_value / total_investment) ** (1 / years_held)) - 1

    tickers = [stock['ticker'] for stock in portfolio]
    start_date = min(stock['investment_date'] for stock in portfolio)

    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date)
        adj_close_df[ticker] = data['Adj Close']

    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    try:
        fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
        risk_free_rate = ten_year_treasury_rate.iloc[-1]
    except Exception as e:
        st.sidebar.write(f"Error fetching risk-free rate: {str(e)}")
        return None

    num_assets = len(tickers)
    weights = np.array([stock['current_value'] for stock in portfolio]) / total_value

    portfolio_expected_return = expected_return(weights, log_returns)
    portfolio_sharpe_ratio = sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    return total_value, portfolio_return, total_unrealized, current_volatility, average_volatility, portfolio_sharpe_ratio, portfolio_expected_return, avg_annual_return, portfolio_values, total_investment

# Grafische Darstellung der Portfolio-Performance
def plot_portfolio_performance(portfolio_values, total_investment):
    fig = px.line(portfolio_values, y='Total', title='Kumulative Portfolio-Performance')
    fig.update_layout(xaxis_title='Datum', yaxis_title='Gesamtwert')
    st.plotly_chart(fig)

# Grafische Darstellung der Asset-Allokation
def plot_asset_allocation(portfolio):
    labels = [stock['ticker'] for stock in portfolio]
    sizes = [stock['current_value'] for stock in portfolio]

    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3)])
    fig.update_layout(title_text='Asset Allocation')
    st.plotly_chart(fig)

# Streamlit App

# Seitenleiste für die Eingabe der Portfolio-Daten und "Berechnen" Button
st.sidebar.header("Portfolio Tracker Input")
get_portfolio_data()
