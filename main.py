#-----------MARKET PREDICTION ALGORITHM-----------
#ADDITIONAL: INSTALL LIBRARIES ON LOCAL MACHINE (OR FROM TERMINAL TO REPO DIRECTLY? LOOK INTO THIS)

#___________USE OF ALPHA VANTAGE API TO PULL LIVE STOCK VALUATION FIGURES, TRAIN MODEL OFF OF THESE VALUATIONS (RANDOM FOREST MODEL)________
#ADDITIONALLY: BUILD IN FUCNTIONALITY THAT CPMPARES PREDICTED VALUATIONS WITH REAL TIME ONES/FUTURE ONES. THIS COULD BE ACHEVED WITH A GRAPHICAL VISUALIZATION OF PREDICTED VS. REAL TIME VALUATIONS

#LIBRARIES REQD'
import requests #PULLS STOCK MARKET DATA
import pandas as pd #ENHANCED DATA MANIPULATION LAYER
import time #PAUSE/TIMIING PROTOCOL: NECESSARY FOR LIVE VALUATIONS
time.sleep(12)  # wait 12 seconds between requests


#ADDITIONAL 
import numpy as np #ENHANCED NUMERICAL HANDLING
import matplotlib.pyplot as plt #GRAPH STATS
import sklearn.ensemble #ML MODEL TRAINING EVALUATING ACCURACY
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier #ML MODEL (ADDITIONAL)
import seaborn as sns #DATA VISUALIZATION
import joblib #SAVE/LOAD MODEL, GIVE USER CAPABILITY TO RUN ACROSS VARIOUS SESSIONS USING PRESET METRICS
import os #FILE MANAGEMENT
from dotenv import load_dotenv #DEALS WITH API KEY
print("CWD:", os.getcwd())



#Alpha Vantage API key configuration
load_dotenv(dotenv_path="/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/.env") #"/Users/keeganhutchinson/CS2704-Market-Prediction-Algorithm/AV-API-key.env" 
api_key = os.getenv("ALPHA_VANTAGE_KEY")
print(f"DEBUG: Loaded API key: {api_key}")

#DEBUG: check if API key is loaded
if not api_key:
    #raise ValueError("ERROR: API key not found. is it in your .env file?")
    print("DEBUG: API key not found! check .env file or file path")
else:
    print ("DEBUG: API Key loaded successfully!")


#-----------GENERAL PSEUDOCODE/HIERARCHICAL LAYOUT-----------


#CRITERIA/FUNCTIONAL COMPONENTS

#1. DATA INGESTION
#-DOWNLOAD DAILY/LIVE STOCK VALUATION FIGURES. TO BE ACCOMLISHED VIA. USE OF DAILY OHLCV FROM ALPHA VANTAGE
#--SCHEDULE DAILY JOB (VIA SCHEDULE)
#--FETCHING OF LATEST SPY DATA (VIA ALPHA VANTAGE API): *COMPLETED*
#---THIS IS TO FETCH CURRENT MARKET VALUATION VARIABLES, DAILY ADJUSTED OHLCV VALUATIONS 
#----VIA USE OF ***TIME_SERIES_DAILY_ADJUSTED*** ENDPOINT, RETURNING OHLCV VALUATIONS FROM AV API
def fetch_ohlcv(symbol="SPY", interval='1min', outputsize='full', api_key=None):
    
    #Fetch daily OHLCV data from Alpha Vantage API
    print('Fetching OHLCV data valuations...')
    url=f"https://www.alphavantage.co/query" #THIS LINK MIGHT BE BROKEN

    params = {
        "function": "TIME_SERIES_INTRADAY", #ONLY USE TIME_SERIES_INTRADAY FOR PER MINUTE DATA, BUT THISLL DO FOR THE ASSIGNMENT OBJECTIVE ATM #TIME_SERIES_DAILY_ADJUSTED IS APPARENTLY A PREMIUM ENDPOINT??
        "symbol": symbol,
        "interval": "5min",
        "apikey": api_key,
        "outputsize": "compact", #much smaller data set, may be good for avoiding overwhelming AI
        "datatype": "json"
    }

#    params = {
#        "function": "TIME_SERIES_INTRADAY", #ONLY USE TIME_SERIES_INTRADAY FOR PER MINUTE DATA, BUT THISLL DO FOR THE ASSIGNMENT OBJECTIVE ATM #TIME_SERIES_DAILY_ADJUSTED IS APPARENTLY A PREMIUM ENDPOINT??
#        "symbol": "SPY",
#        "interval": "1min",
#        "apikey": api_key,
#        "outputsize": "compact",
#        "datatype": "json"
#    }
    print(f"DEBUG: API request params: {params}")
    print(f"DEBUG: API request URL: {url}?{requests.compat.urlencode(params)}")

    # Make the API request
    response = requests.get(url, params=params)
    print(f"DEBUG: API response status code: {response.status_code}")
    #-------
    response=requests.get(url, params=params)

    #parse .json response
    data=response.json()
    print(f"DEBUG: API response: {data}")


    #DEBUG: check if API response was successful
    if response.status_code != 200:
        print(f"ERROR: API request failed with status code {response.status_code}")
        return None
    
    #check rate limit/invalid response time
    if "Note" in data:
        print("ERROR: API rate limit exceeded. Try again later.")
        return None
    #CHECK FOR INVALID/UNEXPECTED RESPONSE FORMAT
    time_series_key=[k for k in data.keys() if 'Time Series' in k]
    if not time_series_key:
        print("ERROR: Time Series data not found in API response")
        return None
    
    key = time_series_key[0]
    raw_df = pd.DataFrame.from_dict(data[key], orient='index')
    raw_df=raw_df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",        
        "5. volume": "Volume",
    })
    # Convert "sting" valuations to "floats"
    raw_df = raw_df.astype(float)

    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()
    print('DATA PARSED/EXTRACTED SUCCESSFULLY!')
    return raw_df

    # DEBUG: print first couple rows of DataFrame
    print(f"DEBUG: Parsed DataFrame head:\n{raw_df.head()}")

    raw_df.index =pd.to_datetime(raw_df.index)
    raw_df=raw_df.sort_index()
    print('DATA PARSED/EXTRACTED SUCCESSFULLY!')
    return raw_df

#new: test call
if __name__ == "__main__":
    print("DEBUG: Starting program...")
    df = fetch_ohlcv(symbol="SPY", api_key=api_key)
    if df is not None:
        print("DEBUG: Data fetched successfully!")
        print(df.head())
    else:
        print("ERROR: Failed to fetch data.")

#2. FEATURE ENGINEERING
#-STUCTURES ACTUAL SET FUNCTIONALITY OF ALGORITHMS TECHNICAL FEATURES
#--RSI, MACID, MOVING AVERAGES, VOLATILITY, RETURNS (OR OTHER RELEVANT INDICATORS)
#---ONLY MOST RECENT ROW NEEDS COMPUTATION: REDUCES REDUNCANCY + MITIGATES POTENTIAL MODEL OVERFITTING
#---***RECALCULATION OF FEATURE COLUMNS + FEATURE CONSISTENCY *CRITICAL* FOR VALID MODEL OUTPUT, AS IS TO BE SOLE BASIS OF OUR BELOW RENDITIONS!****
def calculate_technical_indicators(df):
    #this function intends to deal with adding our extracted moving averages, returns, RSI, etc. etc. etc.: effectively all the above metrics
    df["Return"] = df['Close'].pct_change()
    df["MA_20"] = df['Close'].rolling(window=20).mean()
    df["Volatility"] = df['Return'].rolling(window=20).std()

    #RSI calc (14 day interval)
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0)
    loss = delta.where(delta<0,0)

    avg_gain=gain.rolling(window=14).mean()
    avg_loss=loss.rolling(window=14).mean()

    #'relative strength' calculation
    rs = avg_gain/avg_loss 
    df['RSI'] = 100 - (100 / (1+rs)) #RS *INDICATIOR* VALUATION

    return df.dropna()


#3. "CRASH LABELING" LOGIC
#BINARY CLASSIFICATION ON BASIS OF (PREDICTED) FUTURE RETURNS
#--Each row labeled as followed: (0==NORMAL (ELSE), 1==CRASH (means next day return <-3%)) 
# + CONFIDENCE PROBABLITY VALUATION (LOOK INTO A LIL BIT)
def label_crashes(df, threshold=-0.03): #labels crash if next day return <-3%
    df=df.copy()
    df["Future_Close"]=df['Close'].shift(-1)
    df["Future_Return"]=(df["Future_Close"]-df['Close'])/df['Close']
    df.dropna(subset=["Future_Return"])
    df["Crash"]=(df["Future_Return"]<threshold).astype(int)

    return df


#4. ML MODEL ARCHETECHURE (BASED ON RANDOM FOREST MODEL)
#-RECURSIVE SELF TRAINING OF ML MODEL
#-- WILL UTILIZE "RANDOM FOREST" STYLED ML MODEL, BASED OFF OF THESE EXTRACTED VALUATIONS/EVERCHANGING DATASET VALUATIONS
#---RANDOM FOREST MODEL: USED FOR INTERPRETABILITLY/ROBUSTNESS OF OVERALL ML ALGORITHM AND ARCHITECHTURE
def train_model(df, features=["RSI", "MA_20", "Volatility", "Return"], target="Crash"): #in theory, trains our model on above extractions
    
    #selection of feature and target (X and y variables respectively) from DataFrame
    X=df[features] #features inputted to be used to train our model below
    y=df[target] #deals w/ output labels (crash/no crash)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting of dat ainto training/test subsets (80/20 split in this case)
    model=RandomForestClassifier(n_estimators=100, random_state=42) #imitilaization of RF classifier, in this case utilizing 100 trees.
    model.fit(X_train, y_train) #train model w/ training data

    y_pred=model.predict(X_test) #prediction crash labels on test set

    #display performance metrics
    print('\nModel Performance Metric Valuations:')
    print('Accuracy')
    print('Classification Report:\n', classification_report(y_test, y_pred))

    #saving of trained model above for future use.
    joblib.dump(model, "market_crash_model.pkl") #Saves our pre trained model (ideally)
    return model

#5. LIVE PREDICTION PIPELINE
#5.1: 
#-WILL INCLUDE ACCOMPANYING CUMULATIVE CONFIDENCE SCORES, AS DISTILLED FROM ABOVE PROCESSES
#--MAITENCNCE OF A LIVE CONSISTENT LOG IMPORTANT HERE, AS EVEN IF NO CRASH IS PREDICTED IS STILL CRITICAL COMPONTNET OF GENERATIING A CUMULATIVE DAILY CONFIDENCE TREND VISUALIZATION
def live_predict(df, model_path="market_crash_model.pkl"):
    if not os.path.exists(model_path):
        print('Model file not found')
        return None
    
    model = joblib.load(model_path)

    latest_row=df.iloc[-1:]
    features=['RSI', 'MA_20', 'Volatility', 'Return']
    prediction=model.predict(latest_row[features])[0]
    prob=model.predict_proba(latest_row[features])[0][1] #DEALS W/ CRASH CLASS PROBABILITY

    print(f'Live Prediction: {"CRASH" if prediction == 1 else "NORMAL"} | Confidence: {prob:.2f}')

#LOG PREDICTIONS
    log_entry=f'{pd.Timestamp.now()}, Prediction: {prediction}, Confidence: {prob:.4f}\n'
    with open('prediction_log.txt', "a") as f: 
        f.write(log_entry)


    return prediction, prob #BUGGING


#5.2: (OPTIONAL, FOR ACCURACY SAKE)
# RETRAIN ML MODEL MONTHLY WITH UPDATED DATASET VALUATIONS 
# --THIS IN THEORY WILL HELP FOR OUR ML MODEL TO ADAPT TO EVER CHANGING MARKET BEHAVIOUR + MAINTAIN A LAYER OF PREDICTION ACCURACY

def retrain_model_monthly(df, features=['RSI', 'MA_20', "Volatility", "Return"], target='Crash'): 
    print("Retraining model with updated data figures...")
    model = train_model(df, features, target)
    print("Model retraining successful!")
    return model

#6.
if __name__ == '__main__':
    print ("DEBUG: starting program...")
    df=fetch_ohlcv(symbol="SPY", api_key = api_key)

    if df is not None:
        df = calculate_technical_indicators(df)
        df=label_crashes(df)
        model=train_model(df)
        live_predict(df)
    else:
        print("ERROR: Failed to fetch data")

#7. (TBD) DATA VISULAIZATION