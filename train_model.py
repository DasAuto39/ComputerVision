from ultralytics import YOLO

def main():
    # Load dataset
    data_path = "data.yaml"

    # Load YOLOv5s model
    model = YOLO("runs\detect\sign_detector\weights\last.pt")

    # Train configuration
    model.train(
        data=data_path,
        imgsz=640,
        epochs=50,
        batch=8,
        name="sign_detector",
    )

if __name__ == "__main__":
    # Required on Windows to avoid multiprocessing error
    main()
