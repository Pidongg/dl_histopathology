from ultralytics import YOLO


def main():
    model = YOLO("C:\\Users\\kwanw\\PycharmProjects\\dl_histopathology\\runs\detect\\train\\weights\\last.pt")
    model.to('cuda')

    results = model.train(resume=True)


if __name__ == "__main__":
    main()
