import cv2
from PIL import Image

from .inference import predict


def predict_video(input_path, output_path, model, threshold=0.7):
    capture = cv2.VideoCapture(str(input_path))
    writer = None

    try:
        if not capture.isOpened():
            raise ValueError("Could not open the input video")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Could not read the video frame rate")

        while True:
            success, frame = capture.read()
            if not success:
                break

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise ValueError("Could not create the output video")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)

            result = predict(image, model, threshold=threshold)

            for detection in result["predictions"]:
                x1, y1, x2, y2 = map(int, detection["box"])
                score = detection["score"]

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    color=(0, 0, 255), thickness=2,
                )
                cv2.putText(
                    frame,
                    f"Drone: {score:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            writer.write(frame)

        if writer is None:
            raise ValueError("The video contains no readable frames")

    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return output_path