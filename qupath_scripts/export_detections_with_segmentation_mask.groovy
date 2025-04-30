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
def pathOutput = buildFilePath('M:/Unused/TauCellDL/test_seg_masks', name)
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
    .addLabel(PathClass.fromString("Background"), 0)
    .addLabel(PathClass.fromString("TA"), 1)
    .addLabel(PathClass.fromString("CB"), 2)
    .addLabel(PathClass.fromString("coiled"), 2)
    .addLabel(PathClass.fromString("NFT"), 3)
    .addLabel(PathClass.fromString("tau_fragments"), 4)
    .addLabel(PathClass.fromString("Others"), 4)
    .downsample(downsample)
    .tileSize(512)
    .multichannelOutput(false)
    .build()

int i = 0
for (annotation in getAnnotationObjects()) {
    def region = RegionRequest.createInstance(
        labelServer.getPath(), downsample, annotation.getROI())
    i++
    def outputPath = buildFilePath(pathOutput, 'Region ' + i + '.png')
    writeImageRegion(labelServer, region, outputPath)
}