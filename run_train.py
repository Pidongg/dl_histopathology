# referenced the docs: https://docs.ultralytics.com/usage/python/

from ultralytics import YOLO


def main():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    model.to('cuda')

    print("Device:", model.device.type)

    # Train the model
    results = model.train(data="./yolov8_DeepLesion_train.yaml",
                          epochs=50,
                          patience=10,
                          save_period=10,
                          degrees=180,
                          flipud=0.5,
                          plots=True,
                          dropout=0.2
                          )

    # Evaluate the model's performance on the validation set
    # results = model.val()

    # Perform object detection on an image using the model
    # image_list = glob.glob(os.path.join(ROOT_DIR, "images/test") + "/*.png")
    # example_test_img = image_list[0]
    # results = model(example_test_img)


if __name__ == "__main__":
    main()
