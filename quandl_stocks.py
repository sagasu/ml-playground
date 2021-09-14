import pandas as pd
import quandl
import math

data_frame = quandl.get('WIKI/GOOGL')

# limit the columns that we display, and work with from 12 to 6
data_frame = data_frame[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
data_frame['HL_PCT'] = (data_frame['Adj. High'] - data_frame['Adj. Close']) / data_frame['Adj. Close'] * 100.0
data_frame['PCT_change'] = (data_frame['Adj. Close'] - data_frame['Adj. Open']) / data_frame['Adj. Open'] * 100.0

# again limiting number of columns to just a meaningful features, hopefully they are labels.
data_frame = data_frame[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(data_frame)

forecast_col = 'Adj. Close'
data_frame.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(data_frame)))
data_frame['label'] = data_frame[forecast_col].shift(-forecast_out)
data_frame.dropna(inplace=True)
print(data_frame.head())
