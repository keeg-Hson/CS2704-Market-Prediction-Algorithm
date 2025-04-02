# 📈 CS2704-Market-Prediction-Algorithm
-----
👨‍💻 Author:
Keegan Hutchinson
------
Welcome!

The following project was both designed and implementtd to pose as a machine learning algorithm that seeks to predict both past, present, and future stock market behaviours (such as market spikes and crashes) wish the utilization of both built in technical indicators (such as RSI, moving average, and volatility calculations), as well as a Random Forest machine learning classifier model.
-----
##Purpose: 
This project was built based off of  live dataset valuations as extracted from the [Alpha Vantage API](https://www.alphavantage.co/documentation/), with the algorithmic function aiming to perform daily market predictions with accompanying confidence score valuations, log results, as well as associated trend valuations over time.
-----
##Key Features:

-**Real Time Data Ingestion** -
-**Technical Indicator Calculations** -
-**Event Detection** - 
-**Random Forest ML Model** - 
-**Human Readable Forecasting** - 
-**Daily Scheduling** - 
-**Confidence Trend Graph** - 
-**Prediction Logging** - 
-----
##Quick Start:
1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/market-prediction-algorithm.git
   cd market-prediction-algorithm

2. **Set up enviornment**
    **A) Install dependencies**
        pip install -r requirements.txt
    
    **B) Create .env file, add [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) here**
    
        ALPHA_VANTAGE_KEY=your_api_key_here

3. **Run program maually**
    python main.py

4. **Run with daily scheduler**
    python main.py  #Scheduler initializes automatically at 6:00 PM

-----
##Example Output:
**Graphs:** Saved to /graphs/
**Model:** Stored in Market_Crash_Model.pkl
**Logs:** Confidence prediction valuations stored in /prediction_log.txt
-----




