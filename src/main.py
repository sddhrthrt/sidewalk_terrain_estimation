import numpy as np
import pandas as pd
import os

from src.gps import get_gps_data
from src.shapes import get_shapefile_segments
from src.geom import great_circle_dist_from_dots
from src.utils import hashize
from src.signal import read_files, preprocess
from src.learn import *


def process_signals(last_n=1, n=1, fitfunc=nn_fit, testfunc=nn_test, display=False):
  filelist = []
  for root, dirs, files in os.walk("files/olddata/"):
    filelist = [os.path.join(root, f) for f in files if f.startswith("accel")]

  filelist = filelist[-last_n:][:n]

  df_all = read_files(filelist, display=display)

  df_all["X"], df_all["y"], df_all["df_full"] = [], [], []
  
  for df, tags_df in zip(df_all["df"], df_all["tags_df"]):
    df, X, y = preprocess(df, display=display)
    if X is not None:
      df_all["df_full"].append(df)
      df_all["X"].append(X)
      df_all["y"].append(y)
  
  print([x.shape for x in df_all["X"]])
  X = np.hstack(df_all["X"])
  y = np.hstack(df_all["y"])
  
  c, model = fit_validate(X, y, fitfunc, testfunc, display=display)
  return c, model


def generate_heatmap():
  # first, take locations of data
  files = [
      "files/gps_raw_1510789098_walktogym.csv",
      ]
  shapefile_name = 'sidewalk_inventory_wgs84/Sidewalk_Inventory_wgs84/Sidewalk_Inventory_wgs84.shp'
  dots_df, dots = get_gps_data(files)
  
  # load sidewalk maps
  P1, P2, sidewalk_df = get_shapefile_segments(shapefile_name)

  # Prepare for HMM:
  # need three things:
  # 1. compute map-matching candidates for each GPS position
  # 2. compute distances between GPS positions and map-matching candidates
  # 3. compute shortest routes between subsequent map-matching candidates
  
  # 1. compute map-matching candidates for each GPS position
  dist, distd = great_circle_dist_from_dots(P1, P2, dots)

  candidates = {}
  hashable_points = list(map(tuple, hashize(dots)))

  for p in range(len(hashable_points)):
    candidate_list = []
    for s in (-dist)[p,:].argsort()[:20]:
      candidate_list.append(dist[p,s])
    candidates[hashable_points[p]] = candidate_list
  return dist, dots, candidate_list 

if __name__=="__main__":
  pass
