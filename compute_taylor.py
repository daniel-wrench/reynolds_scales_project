from utils import *


data = pd.read_pickle('wih2mfi_hr_16')

num_periods = ((data.index[-1] - data.index[0]).days*(24*60*60) +
               (data.index[-1] - data.index[0]).seconds) // (6 * 60*60)


taylor_scales = pd.DataFrame(
    [
        compute_taylor_time_scale(
            data,
            data.index[0] + (i*pd.DateOffset(hours=6)),
            data.index[0] + (i+1)*pd.DateOffset(hours=6),
            0.1,
            num_lags=20,
            show=False
        )
        for i in range(num_periods)
    ],
    columns=['Timestamp', 'taylor_ts', 'sample_freq'])

taylor_scales.set_index('Timestamp', inplace=True)

taylor_scales.to_pickle('taylor_16')
