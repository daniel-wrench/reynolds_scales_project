from utils import *

data = pd.read_pickle('wih2mfi_lr_19')

num_periods = ((data.index[-1] - data.index[0]).days*(24*60*60) +
               (data.index[-1] - data.index[0]).seconds) // (6 * 60*60)


corr_scales = pd.DataFrame(
    [compute_correlation_time_scale(
        data,
        data.index[0] + (i*pd.DateOffset(hours=6)),
        data.index[0] + (i+1)*pd.DateOffset(hours=6),
        0.1,
        num_lags=1500,
        show=False)
     for i in range(num_periods)],
    columns=['Timestamp', 'cor_ts', 'cor_ts_est', 'sample_freq'])

corr_scales.set_index('Timestamp', inplace=True)

corr_scales.to_pickle('corr_19')
