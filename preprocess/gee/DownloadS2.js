// ----------------------------------- //
// ------------ Variables ------------ //
// ----------------------------------- //
var year_int = 2022;
var year = year_int.toString();
var field_name = 'Adams';
var bound = Adams;

var bandnames = ['QA60', 'aerosol', 'blue', 'green', 'red','red1','red2','red3','nir','red4','h2o', 'swir1', 'swir2'];
var bandnames_ex = ['blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'swir1', 'swir2']

var rgbVis = {
    min: 0,
    max: 0.3000,
    bands: ['B4', 'B3', 'B2'],
};

var num_dict_2019 = {
 Adams: 106,
 Haskell: 115,
 Randolph: 80
};

var num_dict_2020 = {
 Adams: 119,
 Haskell: 120,
 Randolph: 90
};

var num_dict_2021 = {
 Adams: 115,
 Haskell: 125,
 Randolph: 83
};


var num_dict_2022 = {
 Adams: 112,
 Haskell: 118,
 Randolph: 87
};

// ----------------------------------- //
// ------------ Functions ------------ //
// ----------------------------------- //
var Extract = require('users/YXXX/Sentinel:ExtractFunc')
var addImageDate = Extract.addImageDate;
var dailyMosaics = Extract.dailyMosaics;
// var maskS2clouds = Extract.maskS2clouds;
var toa_2AtoInt = Extract.toa_2AtoInt;
var zonalStats = Extract.zonalStats;
var S2_maskCloud = Extract.S2_maskCloud;

// ----------------------------------- //
// --------------- Main -------------- //
// ----------------------------------- //
var startDay = ee.Date.fromYMD(year_int, 1, 1)
var endDay = ee.Date.fromYMD(year_int, 12, 31)

var s2s = ee.ImageCollection("COPERNICUS/S2_SR")
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .filterBounds(bound)
  .filterDate(startDay, endDay)


// rename s2 bandnames
s2s = s2s
  .map(function(img){
    var t = img.select([ 'B1','B2','B3','B4','B5','B6','B7','B8','B8A', 'B9','B11','B12']).divide(10000);//Rescale to 0-1
    t = t.addBands(img.select(['QA60']));
    var out = t.copyProperties(img).copyProperties(img,['system:time_start']);
    return out;
  })
  .select(['QA60', 'B1','B2','B3','B4','B5','B6','B7','B8','B8A', 'B9','B11','B12'], bandnames)
  .map(function(img){
    var doy = ee.Date(img.get('system:time_start')).getRelative('day','year');
    return img
      .addBands(ee.Image.constant(doy).rename('doy').float())
      .set('doy', doy);
  })
  .map(function(img){
    return img.clip(bound)
  });

// daily mosaic
s2s = dailyMosaics(s2s);

// mask cloud
s2s = S2_maskCloud(s2s);

// var s2 = s2s.filterDate('2022-03-20', '2022-05-31').first()
// print(s2)
// Map.centerObject(Randolph, 9);
// var visParams = {bands: ['red', 'green', 'blue'], max: 0.3};
// Map.addLayer(s2, visParams, 'true-color composite');
// Map.addLayer(Randolph)


// download image ---------------------------------
var data = s2s.map(function(img){
  return img
    .select(bandnames_ex).multiply(10000)
    .addBands(img.select(['doy']))
    .set('system:index', img.get('system:index'));
});

var i_size = data.size()
var data = data.toList(i_size)
// print(i_size)
for(var i=0;i<num_dict_2022[field_name];i++){ // i_size.getInfo()
  worker(i)
}

function worker(i){
  var s2_format_name = field_name + "_" + year + "_" + (i+1);

  i = ee.Number(i);
  var img = ee.Image(data.get(i));

  Export.image.toDrive({
      image:img.toUint16(),
      description:s2_format_name,
      region:bound,
      scale:30,
      folder:field_name + "_S2_" + year,
      crs:"EPSG:4326",
      fileDimensions:512,
      maxPixels: 1e13,
      fileFormat:'GeoTIFF'
  })
}
