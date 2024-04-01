// ----------------------------------- //
// ------------ Variables ------------ //
// ----------------------------------- //
var year_int = 2022;
var year = year_int.toString();
var field_name = 'Adam';
var bound = bound;  // add your own area of interest

var startDay = year_int.toString() + "-01-01"
var endDay = year_int.toString() + "-12-31"

// ----------------------------------- //
// --------------- Main -------------- //
// ----------------------------------- //

var CDL = ee.ImageCollection("USDA/NASS/CDL")
  .filterBounds(bound)
  .filterDate(startDay, endDay)

var LIC = CDL.toList(CDL.size());
var i = ee.Number(0);
var cdl = ee.Image(LIC.get(i)).clip(bound)

// download image ---------------------------------
var cdl_format_name = field_name + "_CDL_" + year_int
Export.image.toDrive({
  image:cdl.toUint16(),
  description:cdl_format_name,
  region:bound,
  scale:30,
  folder:field_name + "_CDL_" + year,
  crs:"EPSG:4326",
  fileDimensions:512,
  maxPixels: 1e13,
  fileFormat:'GeoTIFF'
})
