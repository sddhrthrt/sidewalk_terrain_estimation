import shapefile
import pandas as pd
import numpy as np

def get_shapefile_segments(filename):
  sf = shapefile.Reader(filename)

  nshapes = len(sf.shapes())
  PL1, PL2 = [], []

  for s in sf.shapes():
    PL1.append([s.bbox[1], s.bbox[0]])
    PL2.append([s.bbox[3], s.bbox[2]])

  P1 = np.array(PL1)
  P2 = np.array(PL2)

  names = [[f[0], f[1]] for f in sf.records()]

  names = list(zip(*names))
  names = list(zip(*names))
  namelists = list(zip(*names))

  sidewalk_df = pd.DataFrame({"index": range(len(P1)), "name1": namelists[0], "name2": namelists[1], "lat1": P1[:,0], "lng1": P1[:,1], "lat2": P2[:,0], "lng2": P2[:,1]}) 
  return P1, P2, sidewalk_df


