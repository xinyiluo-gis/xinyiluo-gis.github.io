// Compact Sentinel-2 random-forest land-cover workflow for Google Earth Engine.
// Replace the AOI and imported training collections with your own assets.

var aoi = ee.FeatureCollection('users/your_username/your_aoi');

function maskSentinel2(image) {
  var scl = image.select('SCL');
  var clear = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));
  return image.updateMask(clear).divide(10000);
}

var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
var composite = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2025-06-01', '2025-08-31')
  .filterBounds(aoi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskSentinel2)
  .median()
  .clip(aoi);

// Import FeatureCollections named water, forest, city, cropland, grassland,
// shrubland, and bareland. Each feature needs an integer `landcover` property.
var samples = water
  .merge(forest)
  .merge(city)
  .merge(cropland)
  .merge(grassland)
  .merge(shrubland)
  .merge(bareland);

var sampled = composite.select(bands).sampleRegions({
  collection: samples,
  properties: ['landcover'],
  scale: 10
});

var split = sampled.randomColumn('random', 42);
var training = split.filter(ee.Filter.lt('random', 0.7));
var testing = split.filter(ee.Filter.gte('random', 0.7));

var classifier = ee.Classifier.smileRandomForest({numberOfTrees: 200}).train({
  features: training,
  classProperty: 'landcover',
  inputProperties: bands
});

var classified = composite.select(bands).classify(classifier);
var validation = testing.classify(classifier);
var matrix = validation.errorMatrix('landcover', 'classification');

print('Confusion matrix', matrix);
print('Overall accuracy', matrix.accuracy());
print('Kappa', matrix.kappa());

Map.centerObject(aoi, 11);
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'RGB');
Map.addLayer(classified, {min: 1, max: 7}, 'Land-cover classification');


