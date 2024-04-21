from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO(".\\models\\exclude_zero_scores_w_rotate_30_epochs\\weights\\best.pt")
    model.to('cuda')

    print("Device:", model.device.type)

    # Validate the model
    metrics = model.val(data="./yolov8_Amgad2019_eval.yaml", classes=[1, 2, 3, 4, 6, 9, 10, 11, 14, 16, 17, 18, 19, 20])


if __name__ == "__main__":
    main()

