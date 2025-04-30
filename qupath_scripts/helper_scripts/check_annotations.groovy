// Get the selected annotation
def annotations = getSelectedObjects()

if (annotations == null || annotations.isEmpty()) {
    print "Please select an annotation"
    return
}

annotations.eachWithIndex { annotation, i ->
// Get the ROI (Region of Interest)
def roi = annotation.getROI()

def boundX = roi.getBoundsX()
def boundY = roi.getBoundsY()
def width = roi.getBoundsWidth()
def height = roi.getBoundsHeight()

print "Bounding Box:"
print "X: " + boundX
print "Y: " + boundY
print "Width: " + width
print "Height: " + height
}