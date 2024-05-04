from ultralytics import YOLO
import matplotlib.pyplot as plt


def main():
    # Load a pretrained YOLOv8n model
    model = YOLO('../models/YOLOv8/Tau/0_baseline/weights/best.pt')
    model.to('cuda')

    # Run inference on an image
    results = model("..\\prepared_data\\Tau\\images\\valid\\747297 [d=0.98892,x=94937,y=79747,w=506,h=506].png")  # list of 1 Results object

    print(results[0].boxes)
    # plt.imshow(results[0].plot())
    # plt.show()
    # print(results[0].tojson())


if __name__ == "__main__":
    main()
