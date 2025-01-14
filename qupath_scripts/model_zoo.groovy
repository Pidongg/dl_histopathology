import qupath.ext.djl.*

// Allow model to be downloaded if it's not already
boolean allowDownsamples = true

// Get an object detection model from the zoo
var artifacts = DjlZoo.listObjectDetectionModels()
var artifact = artifacts[0]

// Load the model
var criteria = DjlZoo.loadModel(artifact, allowDownsamples)

// Apply the detection to the current image
var imageData = getCurrentImageData()
var detected = DjlZoo.detect(criteria, imageData)
println "Detected objects: ${detected.orElse([])}"