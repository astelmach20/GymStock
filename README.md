# GymStock - a simple OpenAI gym environment for stock/crypto trading

 creates a new stock environment with the given dataframe
  - dataframe must already be pre-processed with date data structure as the index
    - i.e. using pd.to_datetime
    - dates must start at earliest time (so row 0 is the earliest time)

  - this environment allows us to buy, sell, or hold a stock every day with a discrete action space
    - 0 = hold
    - 1 = buy 1 share
    - 2 = buy 25% of your money to invest in shares
    - 3 = buy 50% of your money to invest
    - 4 = buy 75% of your money to invest
    - 5 = buy all of your money to invest
    - 6 = sell 1 share
    - 7 = sell 25% of your shares
    - 8 = sell 50% of your shares
    - 9 = sell 75% of your shares
    - 10 = sell all shares

  - hyperparameters also include start_index and window_size
    - start_index is the time that we start our data at
    - window_size is the period of time that we want to be looking at previously
      - i.e. 30 days behind 
