import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model

from src.main import process_signals
from src.learn import nn_test
from src.osm import  gpx_track_from_dots, build_GPX_from_dots
from src.signal import read_files, preprocess
from src.gps import get_gps_data


app = Flask(__name__)

try:
  model = load_model("model.h5")
except OSError:
  c, model = process_signals(-1)
  model.save("model.h5")


@app.route("/ping")
def ping():
  return jsonify({"ping": "pong"})

@app.route("/upload", methods=["POST", ])
def upload_sensor_data():
  if request.method == "POST":
    data = request.get_json()
    app.logger.debug(data)
    saved_name = os.path.join('files/', data['filename'])  
    with open(saved_name, 'w') as f:
      f.write(data['contents'])
    if data['filename'].startswith('gps'):
      df, dots = get_gps_data(saved_name)
      gpxname = os.path.join("files/", data['filename'].replace('gps', 'gpx').replace('csv', 'xml'))
      gpxname = build_GPX_from_dots(dots, gpxname)
    return jsonify({"filename": saved_name})


@app.route("/classify", methods=["POST", ])
def classify():
  if request.method == "POST":
    data  = request.get_json()
    print(data)
    points = np.array(data["points"])
    filename = os.path.join("files/", data["filename"])
    accel = filename.replace("gpx", "accel").replace("xml", "csv")

    dfa = read_files([accel,]) 
    df, X, y = preprocess(dfa["df"][0]) 
    if X is None:
      return [0,]*len(points)
    
    scores = nn_test(model, X.T).astype('int')
    
    _, _, gpx_segment = gpx_track_from_dots(points)

    ratios = gpx_segment._get_interval_distances_ratios(gpx_segment.points, gpx_segment.points[0], gpx_segment.points[-1])
    colors = list(map(lambda r: int(scores[int((len(scores)-1)*r)]), ratios))
    print("Colors: ", colors)
    return jsonify(colors)

@app.route("/app/<path:path>")
def serve_app(path):
  return send_from_directory('app', path)
