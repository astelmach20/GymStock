# GymStock - a simple OpenAI gym environment for stock/crypto trading

 creates a new stock environment with the given dataframe
  - dataframe must already be pre-processed with date data structure as the index
    - i.e. using pd.to_datetime
    - dates must start at earliest time (so row 0 is the earliest time)

  - this environment allows us to buy, sell, or hold a stock every day with a discrete action space
    - 0 = buy
    - 1 = sell
    - 2 = hold

  - hyperparameters also include start_index and window_size
    - start_index is the time that we start our data at
    - window_size is the period of time that we want to be looking at previously
      - i.e. 30 days behind 
