import qupath.ext.djl.*
import static qupath.lib.scripting.QP.*
import ai.djl.modality.cv.*
import ai.djl.repository.zoo.*
import ai.djl.translate.*
import ai.djl.modality.cv.translator.*
import ai.djl.modality.cv.transform.*
import java.nio.file.Paths
import ai.djl.modality.cv.output.DetectedObjects
import javax.imageio.ImageIO
import ai.djl.modality.cv.Image
import ai.djl.inference.Predictor
import qupath.lib.regions.RegionRequest
import java.awt.image.BufferedImage
import ai.djl.onnxruntime.engine.OrtEngine
import ai.djl.Device
import java.nio.file.Path

var modelPath = 'C:/Users/peiya/Downloads/train26/weights/best.onnx'
def model = null

try {
    // Disable all other engines and downloads
    System.setProperty("ai.djl.repository.disable.download", "true")
    System.setProperty("ai.djl.pytorch.disable_native_library", "true")
    System.setProperty("ai.djl.mxnet.disable_native_library", "true")
    System.setProperty("ai.djl.tensorflow.disable_native_library", "true")
    System.setProperty("ai.djl.use.native.engine", "false")
    
    // Configure ONNX Runtime
    def userHome = System.getProperty("user.home")
    def onnxRuntimePath = new File(userHome, ".djl.ai/onnxruntime")
    System.setProperty("ai.djl.onnxruntime.lib_path", onnxRuntimePath.absolutePath)
    System.setProperty("ai.djl.default_engine", "OnnxRuntime")
    
    // Initialize engine
    def engine = OrtEngine.getInstance()

    // YOLOv8 specific pipeline with default input size
    Pipeline pipeline = new Pipeline()
            .add(new Resize(640, 640))
            .add(new ToTensor())
            .add(new Normalize(
                new float[] {0, 0, 0},
                new float[] {255.0f, 255.0f, 255.0f}
            ))
            
    println "Setting up translator..."
    translator = new YoloV8Translator.Builder()
            .setPipeline(pipeline)
            .optThreshold(0.0f)
            .build()
            
    println "Creating criteria..."
    Criteria<Image, DetectedObjects> criteria = Criteria.builder()
        .setTypes(Image.class, DetectedObjects.class)
        .optTranslator(translator)
        .optModelPath(Paths.get(modelPath))
        .optDevice(Device.cpu())
        .optEngine("OnnxRuntime")
        .build()

    // Verify model file exists
    def modelFile = new File(modelPath)
    if (!modelFile.exists()) {
        throw new RuntimeException("Model file not found at: ${modelPath}")
    }
    println "Model file exists and size: ${modelFile.length()} bytes"

    println "Loading model..."
    model = criteria.loadModel()
    println "Model loaded successfully!"
    
    // Get current image
    var imageData = getCurrentImageData()
    var server = imageData.getServer()
    
    // Create proper region request
    def request = RegionRequest.createInstance(
        server,
        1.0
    )
    
    var img = server.readBufferedImage(request)
    
    // Resize image to 640x640
    var resized = new BufferedImage(640, 640, BufferedImage.TYPE_INT_RGB)
    var g = resized.createGraphics()
    g.drawImage(img, 0, 0, 640, 640, null)
    g.dispose()
    
    // Save resized image to verify
    def desktop = System.getProperty("user.home") + File.separator + "Desktop"
    def outputFile = new File(desktop, "test_image_resized.png")
    ImageIO.write(resized, "PNG", outputFile)
    println "Resized image saved to: ${outputFile.absolutePath}"
    
    // Convert to DJL Image
    println "Converting to DJL Image..."
    var picture = ImageFactory.getInstance().fromImage(resized)
    println "Image size: ${picture.getWidth()} x ${picture.getHeight()}"
    
    // Create predictor and run detection
    println "Creating predictor..."
    def predictor = model.newPredictor()
    println "Running prediction..."
    var detections = predictor.predict(picture)
    
    // Print detailed detection information
    if (detections != null) {
        println "Number of detections: ${detections.getNumberOfObjects()}"
        detections.items().each { detection ->
            println """
                Class: ${detection.className}
                Probability: ${detection.probability}
                Bounds: ${detection.boundingBox}
            """
        }
    } else {
        println "No detections returned (null)"
    }
    
} catch (Exception e) {
    println "Error: ${e.getMessage()}"
    println "Stack trace:"
    e.printStackTrace()
    if (e.getCause() != null) {
        println "Caused by: ${e.getCause()}"
        e.getCause().printStackTrace()
    }
} finally {
    if (model != null) {
        model.close()
    }
}