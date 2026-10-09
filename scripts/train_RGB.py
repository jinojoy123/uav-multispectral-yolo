from ultralytics import YOLO

model = YOLO('/yolomodel.pt')


results = model.train(data = 'dataset.yaml', epochs=100,deterministic = False, plots=True, device=0, batch=16, imgsz=640, workers=4, seed=0,
    task='segment', save=True,
    save_period=10, val=True, amp=True)


metrics = model.val()
print("\nValidation metrics:")
print(metrics)
print("\nTraining completed successfully.")