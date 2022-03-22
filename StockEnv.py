class StockEnv(Env):
  ''' creates a new stock environment with the given dataframe
  - dataframe must already be pre-processed with date data structure as the index
    - i.e. using pd.to_datetime
    - dates must start at earliest time (so row 0 is the earliest time)
  - dataframe must contain price of the stock in a column named 'Close'

  - this environment allows us to buy, sell, or hold a stock every day with a discrete action space
    - 0 = buy
    - 1 = hold
    - 2 = sell 1 share
    - 3 = sell 25% of your shares
    - 4 = sell 50% of your shares
    - 5 = sell 75% of your shares
    - 6 = sell all shares

  - hyperparameters also include start_index and window_size
    - start_index is the time that we start our data at
    - window_size is the period of time that we want to be looking at previously
      - i.e. 30 days behind 

  '''

  def __init__(self, df, investment=1000, start_index=0, end_index=None, window_size = 30):

    self.df = df
    self.window_size = window_size
    self.end_index = len(self.df.index) if end_index == None else end_index # where we end the data

    self.money_left_to_invest = investment
    self.action_space = Discrete(6)

    spaces = self.setup_state()
    self.observation_space = gym.spaces.Dict(spaces)

    self.starting_index = start_index
    self.starting_investment = investment
    self.index = start_index # used for getting the prices
    self.current_time = self.index + self.window_size # the current day we are looking at

    prices = self._get_prices()
    self.shares = 0
    self.state = self.get_current_state()
    self.prev_money = 0

    self.money = []
    self.prev_render = []


  def setup_state(self):
    state = dict()
    shape = (self.window_size, )

    mins = np.min(self.df)
    maxes = np.max(self.df)
    
    for col in self.df.columns:
      state[col] = Box(low=mins[col], high=maxes[col], shape=shape, dtype=np.float32)

    state['money_left_to_invest'] = Box(low = 0, high=np.inf, shape=(1, ), dtype=np.float32)
    state['account_amount'] =  Box(low = 0, high=np.inf, shape=(1, ), dtype=np.float32)

    return state

  # gets the prices
  def _get_prices(self):
    # if we are at the end of the window
    if self.window_size + self.index > len(self.df.index):
      values = self.df.iloc[self.index:, :]['Close']
      return values.reindex(range(self.window_size), fill_value=np.mean(values)).tolist()

    values = self.df.iloc[self.index:self.window_size + self.index, :]['Close'].tolist()

    return values

  def get_total_money(self):
    return self.shares * self._get_today_price() + self.money_left_to_invest

  def _get_today_price(self):
      return self.df['Close'][self.current_time-1]

  def get_current_state(self):
    state = dict()
    state['account_amount'] =  self.shares * self._get_today_price()
    state['money_left_to_invest'] = self.money_left_to_invest

    if self.window_size + self.index > len(self.df.index):
      values = self.df.iloc[self.index:, :]
      values = values.reindex(range(self.window_size), fill_value=np.mean(values))

    else:
      values = self.df.iloc[self.index:self.window_size + self.index, :]

    for col in self.df.columns:
     state[col] = values[col].tolist()

    return state

  # sell n shares
  def sell_n_shares(self, n):
    n = int(n)
    if self.shares <= 0:
      return

    self.shares -= n
    self.money_left_to_invest += self._get_today_price() * n

  def step(self, action):

    # calculate state

    if action == 0: # buy
      cur_price = self._get_today_price()

      # if we have enough money left in our bank account to buy
      if self.money_left_to_invest >= cur_price:
        self.shares += 1
        self.money_left_to_invest -= cur_price
      
    elif action == 1: # hold
      pass # do nothing

    elif action == 2: # sell 1 share
      self.sell_n_shares(1)

    elif action == 3: # sell 25% of shares
      self.sell_n_shares( .25 * self.shares)
    
    elif action == 4: # sell 50% of shares
      self.sell_n_shares( .5 * self.shares)

    elif action == 5: # sell 75% of shares
      self.sell_n_shares( .75 * self.shares)

    elif action == 6:
      self.sell_n_shares(self.shares)

    else:
      raise Exception("Invalid action")

    self.state = self.get_current_state()

    # calculate reward - how much money we made each timestep/trade
    reward = self.get_total_money() - self.prev_money
    self.prev_money = self.get_total_money()

    self.money.append(self.get_total_money())

    # calculate done
    done = self.current_time == self.end_index


    # advance to the next timestep
    self.index += 1 
    self.current_time = self.index + self.window_size

    # calculate info
    info = dict()
    info['Current Money'] = self.state['account_amount'] + self.money_left_to_invest
    info['Shares'] = self.shares
    info['Reward'] = reward
    info['Current Timestep'] = self.current_time

    return self.state, reward, done, info


  def render(self, mode='regular'):
    plt.clf()

    y = self.prev_render if mode == 'dummy' else self.money

    plt.plot(self.df.index[self.starting_index:len(y)], y)
    plt.show()
      

  def reset(self, mode=None):
    self.index = self.starting_index
    self.current_time = self.index + self.window_size

    prices = self._get_prices()
    self.shares = 0
    self.prev_money = 0
    self.money_left_to_invest = self.starting_investment

    self.prev_render = self.money
    self.money = []
    self.first_rendering = True

    self.state = self.get_current_state()


    return self.state

  def close(self):
        plt.close()