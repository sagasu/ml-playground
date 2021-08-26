import pandas as pd
import quandl

data_frame = quandl.get('WIKI/GOOGL')

print(data_frame)