import qupath.ext.djl.DjlObjectDetector
import ai.djl.modality.cv.transform.*
import ai.djl.modality.cv.translator.*
import ai.djl.translate.Pipeline
import static qupath.lib.scripting.QP.*

// Setup translator with pipeline
def translator = new YoloV8Translator.Builder()
        .setPipeline(new Pipeline().add(new ToTensor()))
        .optThreshold(0.01f)
        .build()

// Create detector with model settings
def detector = new DjlObjectDetector(
        "PyTorch", 
        new File("C:/Users/peiya/Desktop/train16/weights/best.torchscript").toURI(), 
        translator, 
        640, 0.0, 0.45, 0.01)

// Set detection thresholds
detector.setClassThreshold("TA", 0.06)
detector.setClassThreshold("NFT", 0.06)

try {
    clearDetections()
    
    def selectedAnnotations = getSelectedObjects()
    
    // Run detection
    def detections = detector.detect(getCurrentImageData(), selectedAnnotations)
    print(detections.isPresent() 
            ? "Detection completed with ${detections.get().size()} objects found" 
            : "Detection was interrupted")
    
} catch (Exception e) {
    print("Error during detection: " + e.getMessage())
    e.printStackTrace()
}