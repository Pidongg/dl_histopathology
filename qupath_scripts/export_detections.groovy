// Get the current image
def imageData = getCurrentImageData()

def server = getCurrentServer()

def imageName = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName())
// def outDir = /PATH_TO_OUTPUT_DIR/
def outDir = "C:/Users/kwanw/PycharmProjects/dl_histopathology/datasets/Tau/labels/Cortical"
mkdirs(outDir)

def filePath = buildFilePath(outDir, String.format('%s_detections.txt', imageName))

File f = new File(filePath)

f.newWriter().withWriter {
    f << server.getWidth() + ' x ' + server.getHeight() << '\n'

    for (obj in getDetectionObjects()) {
        f << obj.getClassifications() << ' : ' << obj.getROI() << '\n'
    }
}