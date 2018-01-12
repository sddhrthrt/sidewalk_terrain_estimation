import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import scipy
from scipy.signal import butter, lfilter


def drawAccelerometerPlots(df, tags, cols):
  n = len(cols)
  print(n)
  fig, axes = plt.subplots(nrows=1, ncols=n)
  if n==1: axes=[axes,]
  for i in range(n):
    ax = df.reset_index().plot(x='tsr', y=cols[i], ax=axes[i], figsize=(18, 21/n), title="Accelerometer readings")
    ax.set_xlabel("Timestamp (ms)")
    ax.set_ylabel("Accelerometer reading")
    ymin, ymax = axes[i].get_ylim()
    axes[i].vlines(x=tags['tsr']*1000, ymin=ymin, ymax=ymax, label="Tags")
    plt.legend()
  plt.show()

def read_files(filelist, display=False):
  dfs = {"df": [], "tags_df": []}
  
  for fn in filelist:
    tags_available = True
    dfi = pd.read_csv(fn,
                 header=None,
                 names=['ts', 'a_x', 'a_y', 'a_z']
                )
    try:
      tags_df = pd.read_csv(fn.replace("accel_raw", "tags"),
                            header=None,
                            names=['ts', 'tag']
                           )
    except FileNotFoundError:
      print("NOT FOUND")
      tags_available = False
    

    dfi['tag'] = ''
    dfi['tag_m'] = 0
    tags = ['TAG_1', 'TAG_2', 'TAG_3', 'TAG_4']
    if tags_available:
      for i, r in tags_df.iterrows():
        range_i = np.abs((dfi['ts']/1000).astype('int')-(r['ts']))<2
        dfi['tag'][range_i] = r['tag']
        dfi['tag_m'][range_i] = tags.index(r['tag'])+1 

    dfi['tsr'] = dfi['ts']-dfi['ts'][0]
    if tags_available:
      tags_df['tsr'] = tags_df['ts']-(dfi['ts']/1000).astype('int')[0]

    dfs["df"].append(dfi)

    if tags_available:
      dfs["tags_df"].append(tags_df)
    
    if tags_available:
      if display:
        for df, tags_df in zip(dfs["df"], dfs["tags_df"]):
          drawAccelerometerPlots(df, tags_df, ['a_y'])
  return dfs


def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype="band")
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y

def preprocess(df, display=False):
  # drop rows, make data more .. periodic ( :( )
  df = df[df['ts'].diff()>10]

  # normalize with zscore 
  # for col in 'a_x', 'a_y', 'a_z':
    # df[col+'_norm'] = scipy.stats.zscore(df[col])
  
  # # subtract rolling mean
  # for col in 'a_x_norm', 'a_y_norm', 'a_z_norm':
    # df[col+'_subtracted'] = df[col]-df[col].rolling(window=5, center=True).mean()

  # # Trim after rolling mean:
  # df = df.iloc[3:-3]
  
  # Savgol filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
  # for col in 'a_x', 'a_y', 'a_z':
    # df[col+'_sg'] = scipy.signal.savgol_filter(df[col+'_norm_subtracted'], 21, 2, deriv=1)
  
  fs = 1000/df['ts'].diff().mean()
  print("FREQUENCY: ", fs)
  # butterworth filter
  try: 
    for col in 'a_x', 'a_y', 'a_z':
      df[col+'_bw'] = butter_bandpass_filter(df[col], 0.1, 4, fs, order=6)
  except ValueError:
    return None, None, None

  # root of squared components 
  # s = df['a_x_sg']**2
  # s = s + df['a_y_sg']**2
  # s = s + df['a_z_sg']**2
  # s = np.sqrt(s)
  # df['a_sg_sq'] = s

  # root of squared components 
  s = df['a_x_bw']**2
  s = s + df['a_y_bw']**2
  s = s + df['a_z_bw']**2
  s = np.sqrt(s)
  df['a_bw_sq'] = s
  
  # preparing X
  # features = ['a_x_norm', 'a_y_norm', 'a_z_norm',
              # 'a_x_norm_subtracted', 'a_y_norm_subtracted', 'a_z_norm_subtracted',
              # 'a_x_sg', 'a_y_sg', 'a_z_sg', 'a_sg_sq', 
              # 'a_x_bw', 'a_y_bw', 'a_z_bw', 'a_bw_sq'] 
  features = ['a_x_bw', 'a_y_bw', 'a_z_bw', 'a_bw_sq'] 

  ncols = len(features)
  X = np.zeros((df.shape[0], ncols))

  # copy df columns into X
  for i in range(len(features)):
    X[:,i] = df[features[i]]
  
  y = np.array(df['tag_m'])  

  window=16
  if X.shape[0]%window:
    X_p = X[:-(X.shape[0]%window)]
    y_p = y[:-(X.shape[0]%window)].astype('int')
    ts = np.array(df['ts'])[:-(X.shape[0]%window)]
  else:
    X_p = X
    y_p = y
    ts = np.array(df['ts'])
  ws = int(X_p.shape[0]/window)
  X_w = np.zeros([62, ws])
  y_w = np.zeros([1, ws])

  for w in range(int(X_p.shape[0]/window)):
    nmax = np.max(X_p[w*window:(w+2)*window,3])
    nmin = np.min(X_p[w*window:(w+2)*window,3])
    nstd = np.std(X_p[w*window:(w+2)*window,3])
    nmean = np.mean(X_p[w*window:(w+2)*window,3])
    X_w[0,w] = nmax
    X_w[1,w] = nstd
    X_w[2,w] = nmean

  f, t, Zxx = signal.stft(X_p[:,0], nperseg=window*2)
  # take only first 10 rows
  X_w[3:13,:] = Zxx[:10,:ws]

  # f_bw, t_bw, Zxx_bw = signal.stft(X[:,13], nperseg=64)
  # # first 10 rows
  # X_w[10:20,:] = Zxx_bw[:10,:ws]

  f_x, t_x, Zxx_x = signal.stft(X[:,0], nperseg=window*2)
  f_y, t_y, Zxx_y = signal.stft(X[:,1], nperseg=window*2)
  f_z, t_z, Zxx_z = signal.stft(X[:,2], nperseg=window*2)
  # first 10 rows
  X_w[13:23,:] = Zxx_x[:10,:ws]
  X_w[23:33,:] = Zxx_y[:10,:ws]
  X_w[33:43,:] = Zxx_z[:10,:ws]

  for w in range(int(X_p.shape[0]/window)):
    pgram = signal.lombscargle(ts[w*window:(w+2)*window], X_p[w*window:(w+2)*window,0], f[1:11]*fs, normalize=True)
    f_w, Pxx = signal.welch(X_p[w*window:(w+1)*window:,0], axis=0) # take 10
    X_w[43:53,w] = pgram[:10]
    X_w[53:62,w] = Pxx[:10]

  ### Normalization
  X_w = (X_w-X_w.mean(axis=0))/X_w.std(axis=0)

  for w in range(int(y_p.shape[0]/window)):
    y_w[0,w] = np.bincount(y_p[w*window:(w+2)*window]).argmax()

  y_w[y_w>0] = 1

  return df, X_w, y_w

