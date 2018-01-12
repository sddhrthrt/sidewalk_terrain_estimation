import pandas as pd
import numpy as np
import os

def get_gps_data(filename=None, last_n=1):
  # files: just one file used
  # 
  # returns 
  # df: ["ts", "provider", "lat", "lng", "haccuracy", "bearing", "speed", "tsr"]
  # dots: [ [ lat, lng],
  #         [ lat, lng], ]
  if filename is None:
    for root, dirs, files in os.walk("files/"):
      filelist = [os.path.join(root, f) for f in files if f.startswith("gps")]

    filename = filelist[-last_n]

  df = pd.read_csv(filename, header=None, names=["ts", "provider", "lat", "lng", "haccuracy", "bearing", "speed"])

  df['tsr'] = df['ts']-df['ts'][0]

  # plt.plot(df.index, (df['tsr'].shift(-1)-df['tsr'])/1000)
  # plt.show()

  # plt.scatter(df['lat'], df['lng'], s=df['haccuracy']/df['haccuracy'].mean()*10)
  # plt.show()

  dots = np.array((df['lat'], df['lng'])).T
  return df, dots
