# General utilities
import warnings
from typing import Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import json


# Plotting
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Sentinel Hub services
from sentinelhub import (
    CRS,
    DataCollection,
    Geometry,
    MimeType,
    SentinelHubRequest,
    SHConfig,
)
from shapely.geometry import shape

warnings.filterwarnings("ignore")
# %matplotlib inline

##
def plot_image(image: np.ndarray, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any) -> None:
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

##
# API CONFIG
# Only run this cell if you have not created a configuration.
config = SHConfig()
config.sh_client_id = "sh-9787af24-7c93-48da-b065-5458580c87ab"
config.sh_client_secret = "ABAh3asffXfc7quHhL3bD1qN4xopMf3k"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.save("cassini")

##
# Possible process for app
# 1) get area of interest by creating a polygon input 2)
area_of_interest = """
{
  "type": "Polygon",
  "coordinates": [
    [
      [
        20,
        34
      ],
      [
        29,
        34
      ],
      [
        29,
        42
      ],
      [
        20,
        42
      ],
      [
        20,
        34
      ]
    ]
  ]
}
"""
aoi = gpd.read_file(area_of_interest)
aoi["geometry"] = aoi
aoi["area"] = aoi.area
aoi.explore("area", color="Green", legend=False)

full_geometry = Geometry(aoi.to_crs(32630).geometry.values[0], crs=CRS.UTM_30N)

##
# Evalscripts
# use satellite data to process certain indices

TIME_INTERVAL = ("2023-07-20", "2023-07-25")
OTHER_ARGS = {"dataFilter": {"mosaickingOrder": "mostRecent"}}



##
evalscript_fire_detection = """
//high accuracy Detect active fire points 
//Sentinel-3 SLSTR
//by Tiznger startup co
//www.tiznegar.com

var SAHM= ((S6 - S5) / (S6 + S5));

if(SAHM>.05 && S1<.23){
  return[5*S3, 1*S2, 1*S1]
}

else {
 return [S6,S3,S2]
}

//Red color indicates active fire areas and points
"""
#for t in ["2023-07-20", "2023-07-21", "2023-07-22", "2023-07-23", "2023-07-24", "2023-07-25"]:
#    TIME_INTERVAL = (t, t)
request_fire_detection = SentinelHubRequest(
    evalscript=evalscript_fire_detection,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL3_SLSTR.define_from(
                name="s3", service_url="https://sh.dataspace.copernicus.eu"
            ),
            time_interval=TIME_INTERVAL,
            other_args=OTHER_ARGS,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    geometry=full_geometry,
    size=[1000, 920],
    config=config,
)

true_color_imgs = request_fire_detection.get_data()
image = true_color_imgs[0]

print(f"Image type: {image.dtype}")
# plot function
plot_image(image, factor=1.5 / 255, clip_range=(0, 1))
plt.title(str(TIME_INTERVAL))
plt.show()





evalscript_ndvi = """
//VERSION=3
//NDVI index

function setup() {
  return {
    input: ["B03", "B04", "B08", "dataMask"],
    output: [
      { id: "default", bands: 4 },
      { id: "index", bands: 1, sampleType: "FLOAT32" },
      { id: "eobrowserStats", bands: 2, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 },
    ],
  };
}

function evaluatePixel(samples) {
  let val = index(samples.B08, samples.B04);
  let imgVals = null;
  // The library for tiffs works well only if there is only one channel returned.
  // So we encode the "no data" as NaN here and ignore NaNs on frontend.
  const indexVal = samples.dataMask === 1 ? val : NaN;

  if (val < -0.5) imgVals = [0.05, 0.05, 0.05, samples.dataMask];
  else if (val < -0.2) imgVals = [0.75, 0.75, 0.75, samples.dataMask];
  else if (val < -0.1) imgVals = [0.86, 0.86, 0.86, samples.dataMask];
  else if (val < 0) imgVals = [0.92, 0.92, 0.92, samples.dataMask];
  else if (val < 0.025) imgVals = [1, 0.98, 0.8, samples.dataMask];
  else if (val < 0.05) imgVals = [0.93, 0.91, 0.71, samples.dataMask];
  else if (val < 0.075) imgVals = [0.87, 0.85, 0.61, samples.dataMask];
  else if (val < 0.1) imgVals = [0.8, 0.78, 0.51, samples.dataMask];
  else if (val < 0.125) imgVals = [0.74, 0.72, 0.42, samples.dataMask];
  else if (val < 0.15) imgVals = [0.69, 0.76, 0.38, samples.dataMask];
  else if (val < 0.175) imgVals = [0.64, 0.8, 0.35, samples.dataMask];
  else if (val < 0.2) imgVals = [0.57, 0.75, 0.32, samples.dataMask];
  else if (val < 0.25) imgVals = [0.5, 0.7, 0.28, samples.dataMask];
  else if (val < 0.3) imgVals = [0.44, 0.64, 0.25, samples.dataMask];
  else if (val < 0.35) imgVals = [0.38, 0.59, 0.21, samples.dataMask];
  else if (val < 0.4) imgVals = [0.31, 0.54, 0.18, samples.dataMask];
  else if (val < 0.45) imgVals = [0.25, 0.49, 0.14, samples.dataMask];
  else if (val < 0.5) imgVals = [0.19, 0.43, 0.11, samples.dataMask];
  else if (val < 0.55) imgVals = [0.13, 0.38, 0.07, samples.dataMask];
  else if (val < 0.6) imgVals = [0.06, 0.33, 0.04, samples.dataMask];
  else imgVals = [0, 0.27, 0, samples.dataMask];

  return {
    default: imgVals,
    index: [indexVal],
    eobrowserStats: [val, isCloud(samples) ? 1 : 0],
    dataMask: [samples.dataMask],
  };
}

function isCloud(samples) {
  const NGDR = index(samples.B03, samples.B04);
  const bRatio = (samples.B03 - 0.175) / (0.39 - 0.175);
  return bRatio > 1 || (bRatio > 0 && NGDR > 0);
}
"""
request_ndvi = SentinelHubRequest(
    evalscript=evalscript_ndvi,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A.define_from(
                name="s2", service_url="https://sh.dataspace.copernicus.eu"
            ),
            time_interval=TIME_INTERVAL,
            other_args=OTHER_ARGS,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    geometry=full_geometry,
    size=[1000, 920],
    config=config,
)

ndvi_imgs = request_ndvi.get_data()
image = ndvi_imgs[0]

print(f"Image type: {image.dtype}")
# plot function
plot_image(image, factor=1.5 / 255, clip_range=(0, 1))
plt.title(str(TIME_INTERVAL))
plt.show()




evalscript_ndmi = """
//VERSION=3
const moistureRamps = [
    [-0.8, 0x800000],
    [-0.24, 0xff0000],
    [-0.032, 0xffff00],
    [0.032, 0x00ffff],
    [0.24, 0x0000ff],
    [0.8, 0x000080]
  ];

const viz = new ColorRampVisualizer(moistureRamps);

function setup() {
  return {
    input: ["B8A", "B11", "SCL", "dataMask"],
    output: [
      { id: "default", bands: 4 },
      { id: "index", bands: 1, sampleType: "FLOAT32" },
      { id: "eobrowserStats", bands: 2, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 },
    ],
  };
}

function evaluatePixel(samples) {
  let val = index(samples.B8A, samples.B11);
  // The library for tiffs works well only if there is only one channel returned.
  // So we encode the "no data" as NaN here and ignore NaNs on frontend.
  const indexVal = samples.dataMask === 1 ? val : NaN;
  return {
    default: [...viz.process(val), samples.dataMask],
    index: [indexVal],
    eobrowserStats: [val, isCloud(samples.SCL) ? 1 : 0],
    dataMask: [samples.dataMask],
  };
}

function isCloud(scl) {
  if (scl == 3) {
    // SC_CLOUD_SHADOW
    return false;
  } else if (scl == 9) {
    // SC_CLOUD_HIGH_PROBA
    return true;
  } else if (scl == 8) {
    // SC_CLOUD_MEDIUM_PROBA
    return true;
  } else if (scl == 7) {
    // SC_CLOUD_LOW_PROBA
    return false;
  } else if (scl == 10) {
    // SC_THIN_CIRRUS
    return true;
  } else if (scl == 11) {
    // SC_SNOW_ICE
    return false;
  } else if (scl == 1) {
    // SC_SATURATED_DEFECTIVE
    return false;
  } else if (scl == 2) {
    // SC_DARK_FEATURE_SHADOW
    return false;
  }
  return false;
}
"""
request_ndmi = SentinelHubRequest(
    evalscript=evalscript_ndmi,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A.define_from(
                name="s2", service_url="https://sh.dataspace.copernicus.eu"
            ),
            time_interval=TIME_INTERVAL,
            other_args=OTHER_ARGS,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    geometry=full_geometry,
    size=[1000, 920],
    config=config,
)

ndmi_imgs = request_ndmi.get_data()
image_new = ndmi_imgs[0]
print(f"Image type: {image_new.dtype}")

# plot function
plot_image(image_new, factor=1.5 / 255, clip_range=(0, 1))
plt.title(str(TIME_INTERVAL))
plt.show()
print(np.array(image_new).shape)

