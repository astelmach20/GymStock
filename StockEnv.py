class StockEnv(Env):
  ''' creates a new stock environment with the given dataframe
  - dataframe must already be pre-processed with date data structure as the index
    - i.e. using pd.to_datetime
    - dates must start at earliest time (so row 0 is the earliest time)
  - dataframe must contain price of the stock in a column named 'value'

  - this environment allows us to buy, sell, or hold a stock every day with a discrete action space
    - 0 = buy
    - 1 = sell
    - 2 = hold

  - hyperparameters also include start_index and window_size
    - start_index is the time that we start our data at
    - window_size is the period of time that we want to be looking at previously
      - i.e. 30 days behind 

  '''

  def __init__(self, df, start_index=0, end_index=None, window_size = 30):

    self.df = df
    self.window_size = window_size
    self.end_index = len(self.df.index) if end_index == None else end_index # where we end the data

    self.action_space = Discrete(3)

    stock_prices = Box(low=np.min(self.df)[0], high = np.max(self.df)[0], shape=(self.window_size, ), dtype=np.float32)
    account_amount = Box(low = 0, high=np.inf, shape=(1, ), dtype=np.float32)
    
    self.observation_space = Tuple((stock_prices, account_amount))

    self.starting_index = start_index
    self.index = start_index # used for getting the prices
    self.current_time = self.index + self.window_size # the current day we are looking at

    prices = self._get_prices()
    self.shares = 0
    self.state = (prices, 0)
    self.prev_money = 0
    # a state is a tuple of the current prices we are looking at in the window along with the money we have

    self.money = []
    self.first_rendering = True



  # gets the prices and increments the index
  def _get_prices(self):
    # if we are at the end of the window
    if self.window_size + self.index > len(self.df.index):
      values = self.df.iloc[self.index:, 0]
      return values.reindex(range(self.window_size), fill_value=0).tolist()

    values = self.df.iloc[self.index:self.window_size + self.index, 0].tolist()

    self.index += 1 # advance to the next timestep
    self.current_time = self.index + self.window_size

    return values

  def _get_today_price(self):
      return self.df['value'][self.current_time-1]



  def step(self, action):

    # calculate state

    if action == 0: # buy
      self.shares += 1

    elif action == 1: # sell
      self.shares = self.shares - 1 if self.shares > 0 else 0

    elif action == 2: # hold
      pass # do nothing

    else:
      raise Exception("Invalid action")

    self.state = (self._get_prices(), self.shares * self._get_today_price())

    # calculate reward - how much money we made each timestep/trade
    reward = self.state[1] - self.prev_money
    self.prev_money = self.state[1]

    # calculate done
    done = self.current_time == self.end_index

    # calculate info
    info = dict()
    info['Current Money'] = self.state[1]
    info['Shares'] = self.shares
    info['Reward'] = reward
    info['Current Timestep'] = self.current_time

    self.money.append(self.shares * self._get_today_price())
    return self.state, reward, done, info


  def render(self):
    if self.first_rendering:
        plt.figure()
        self.first_rendering = False
    plt.clf()
    plt.plot(self.money)
    plt.show()
      

  def reset(self):
    self.index = self.starting_index
    self.current_time = self.index + self.window_size

    prices = self._get_prices()
    self.shares = 0
    self.state = (prices, 0)
    self.prev_money = 0

    self.money = []
    self.first_rendering = True


    return self.state

  def close(self):
        plt.close()