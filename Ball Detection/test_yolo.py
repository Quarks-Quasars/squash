from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\light\Downloads\Ball Detection\runs\detect\runs\fine_tune\weights\best.pt")

    test_metrics = model.val(
        data=r"C:\Users\light\Downloads\Squash Ball Detection.v4i.yolo26\data.yaml",
        split="test",
        plots=True,
        save_json=True,
        name="test_eval",
    )

    print("Test mAP50:", test_metrics.box.map50)
    print("Test mAP50-95:", test_metrics.box.map)
    print("Test precision:", test_metrics.box.mp)
    print("Test recall:", test_metrics.box.mr)

if __name__ == "__main__":
    main()