import numpy as np


def great_circle_dist(P1, P2):#, dot):
  P1 = np.radians(P1)
  P2 = np.radians(P2)
  # (lat, lon) (43, -79
  lat1 = P1[:,0]
  lat2 = P2[:,0]
  dlat = P2[:,0] - P1[:,0] 
  dlon = P2[:,1] - P1[:,1]
  
  a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
  
  km = 6371 * c
  
  return np.abs(km)

def great_circle_dist_from_dots(P1, P2, P3):
  # P1, P2: segments P1[n]-P2[n]
  # P3: points
  # returns:
  #   dxt: abs distance from each P3 to P1-P2 segments
  #        [          ... points ... ]
  #        [ ...                     ]
  #        [ segment  distance(km)   ]
  #        [ ...                     ]
  #   dxtd:  distance from P1 to closest point on P1-P2 to P3
  #          TODO: working?

  P1 = np.radians(P1)
  P2 = np.radians(P2)
  P3 = np.radians(P3)
  # (lat, lon) (43, -79)
  # phi is latitude, lambda longitude
  # http://www.movable-type.co.uk/scripts/latlong.html
  lat1 = P1[:,[0]]
  lat2 = P2[:,[0]]
  lat3 = P3[:,[0]].T
  
  dlat31 = P3[:,[0]].T - P1[:,[0]]
  dlon21 = P2[:,[1]] - P1[:,[1]]
  dlon31 = P3[:,[1]].T - P1[:,[1]]
  
  R = 6371 
  
  a31 = np.sin(dlat31/2)**2 + np.cos(lat1)*np.cos(lat3)*(np.sin(dlon31/2)**2)
  c31 = 2 * np.arctan2(np.sqrt(a31), np.sqrt(1-a31))
  km31 = R * c31
  
  # for bearings
  y12 = np.sin(dlon21)*np.cos(lat2)
  x12 = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon21)
  b12 = np.arctan2(y12, x12)
  
  y13 = np.sin(dlon31)*np.cos(lat3)
  x13 = np.cos(lat1)*np.sin(lat3) - np.sin(lat1)*np.cos(lat3)*np.cos(dlon31)
  b13 = np.arctan2(y13, x13)
  
  # cross track
  dxt = np.arcsin(np.sin(km31/R)*np.sin(b13-b12))*R
  
  # distance from first point to cross point
  dxtd = np.arccos(np.cos(km31/R)/np.cos(dxt/R))*R
  
  return np.abs(dxt), dxtd
