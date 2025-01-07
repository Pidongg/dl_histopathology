// Get the current image
def imageData = getCurrentImageData()

def server = getCurrentServer()

def imageName = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
// uncomment the following line and replace the placeholder
def outDir = 'M:/Unused/TauCellDL/test_labels'
mkdirs(outDir)

def filePath = buildFilePath(outDir, String.format('%s_detections.txt', imageName))

File f = new File(filePath)

f.newWriter().withWriter {
    f << server.getWidth() + ' x ' + server.getHeight() << '\n'

    for (obj in getDetectionObjects()) {
        f << obj.getClassifications() << ' : ' << obj.getROI() << '\n'
    }
}