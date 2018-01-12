from osmread import parse_file, Way
import gpxpy
import gpxpy.gpx


def get_footways():
  footways = []
  for entity in parse_file("map.osm"):
    if isinstance(entity, Way):
      if 'footway' in entity.tags:
        footways.append(entity)
  return footways

def gpx_track_from_dots(dots):
  gpx = gpxpy.gpx.GPX()

  gpx_track = gpxpy.gpx.GPXTrack()
  gpx.tracks.append(gpx_track)

  gpx_segment = gpxpy.gpx.GPXTrackSegment()
  gpx_track.segments.append(gpx_segment)

  for node in dots:
    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(node[0], node[1]))

  return gpx, gpx_track, gpx_segment


def build_GPX_from_dots(dots, filename):
  # dots = [ [lat, lng],
  #          ...
  #          ]
  
  gpx, gpx_track, gpx_segment = gpx_track_from_dots(dots)
  gpx_segment.simplify(max_distance=10)

  gpxfile = gpx.to_xml()

  with open(filename, "w") as f:
    f.write(gpxfile)

  return filename 

  # import requests

  # headers = {"Content-Type": "application/gpx+xml"}
  # files = {"gpx.gpx": gpxfile}
  # response = requests.post("http://localhost:8989/match?vehicle=foot&type=json",  headers=headers, files=files)
  # print(response.json())

# !curl -XPOST -H "Content-Type: application/gpx+xml" -d @./path.gpx "localhost:8989/match?vehicle=foot&type=json"
