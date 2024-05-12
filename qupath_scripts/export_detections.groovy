// Get the current image
def imageData = getCurrentImageData()

def server = getCurrentServer()

def imageName = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
// comment out the following line and replace the placeholder
// def outDir = /PATH_TO_OUTPUT_DIR/
mkdirs(outDir)

def filePath = buildFilePath(outDir, String.format('%s_detections.txt', imageName))

File f = new File(filePath)

f.newWriter().withWriter {
    f << server.getWidth() + ' x ' + server.getHeight() << '\n'

    for (obj in getDetectionObjects()) {
        f << obj.getClassifications() << ' : ' << obj.getROI() << '\n'
    }
}