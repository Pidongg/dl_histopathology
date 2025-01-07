import javafx.application.Platform
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.RegionRequest
Platform.setImplicitExit(false)

def imageData = getCurrentImageData()
def server = imageData.getServer()

// First, ensure all detections have valid classifications
def validClasses = ["TA", "CB", "NFT", "Others", "coiled", "tau_fragments"]
def backgroundClass = PathClass.fromString("Background")

print("Checking and fixing classifications...")
getDetectionObjects().each { detection ->
    def className = detection.getPathClass()?.getName()
    if (className == null || !validClasses.contains(className)) {
        detection.setPathClass(backgroundClass)
        print("Fixed null/invalid class for detection at: " + detection.ROI.centroidX + ", " + detection.ROI.centroidY)
    }
}

// Define output path
def name = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath('M:/Unused/TauCellDL/test_seg_masks_test', name)
mkdirs(pathOutput)

// Define output resolution
double requestedPixelSize = 0.25
double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

print("Creating label server...")
def labelServer = new LabeledImageServer.Builder(imageData)
    .useDetections()
    .grayscale()
    .backgroundLabel(0)
    
    // Add labels with explicit PathClass objects
    // Using values between 0-255 for 8-bit grayscale
    .addLabel(PathClass.fromString("Background"), 0)      // Darkest
    .addLabel(PathClass.fromString("TA"), 1)             // Very dark
    .addLabel(PathClass.fromString("CB"), 128)           // Medium gray
    .addLabel(PathClass.fromString("coiled"), 128)       // Same as CB
    .addLabel(PathClass.fromString("NFT"), 192)          // Light gray
    .addLabel(PathClass.fromString("tau_fragments"), 255) // Brightest
    .addLabel(PathClass.fromString("Others"), 255)       // Same as tau_fragments
    
    .downsample(downsample)
    .multichannelOutput(false)
    .build()

print("Creating region request...")
def region = RegionRequest.createInstance(
    server.getPath(),
    downsample,
    0, 0, server.getWidth(), server.getHeight()
)

def cleanName = name.replaceAll('[\\\\/:*?"<>|]', '_')
def outputPath = buildFilePath(pathOutput, cleanName + '_full_mask.ome.tif')

// Add after creating labelServer but before writing
print("\nVerifying label server configuration:")
print("Number of objects to export: ${getDetectionObjects().size()}")
print("Objects by class:")
getDetectionObjects()
    .collect { it.getPathClass()?.getName() }
    .groupBy { it }
    .each { className, list ->
        print("  ${className}: ${list.size()} objects")
    }

// Add debug visualization before export
def debugRegion = RegionRequest.createInstance(
    labelServer.getPath(),
    1.0,
    0, 0,
    1000, 1000  // Check a larger area
)
def debugImg = labelServer.readRegion(debugRegion)
def nonZeroCount = 0
def totalPixels = 1000 * 1000

// Count non-background pixels
for (int x = 0; x < 1000; x++) {
    for (int y = 0; y < 1000; y++) {
        if ((debugImg.getRGB(x, y) & 0xff) != 0) {
            nonZeroCount++
        }
    }
}
print("\nDebug region statistics:")
print("Total pixels checked: ${totalPixels}")
print("Non-background pixels: ${nonZeroCount}")
print("Percentage non-background: ${(nonZeroCount * 100.0 / totalPixels).round(2)}%")

try {
    print("Writing mask to: " + outputPath)
    writeImageRegion(labelServer, region, outputPath)
    print("Successfully exported mask!")
    
    // Verify the classes used
    print("\nVerifying classifications used:")
    def classesUsed = getDetectionObjects()
        .collect { it.getPathClass()?.getName() }
        .unique()
        .findAll { it != null }
    print("Classes found in detections: " + classesUsed)
    
} catch (Exception e) {
    print("Error writing mask: ${e.getMessage()}")
}