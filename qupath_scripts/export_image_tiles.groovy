// Get the current image (supports 'Run for project')
def imageData = getCurrentImageData()

// Define output path
def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
// def pathOutput = buildFilePath(/PATH_TO_OUTPUT_DIRECTORY/, name)
def pathOutput = buildFilePath("C:/Users/kwanw/PycharmProjects/dl_histopathology/datasets/Tau/images/BG", name)
mkdirs(pathOutput)

// Define output resolution in calibrated units (e.g. µm if available)
double requestedPixelSize = 0.25  // reduce this for better resolution

// Convert output resolution to a downsample factor
double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

// Create an exporter that requests corresponding tiles from the original & labelled image servers
new TileExporter(imageData)
    .downsample(downsample)   // Define export resolution
    .imageExtension('.png')   // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(512)            // Define size of each tile, in pixels
    .annotatedTilesOnly(true) // If true, only export tiles if there is a (classified) annotation present
    .overlap(32)              // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)   // Write tiles to the specified directory

print 'Done!'

