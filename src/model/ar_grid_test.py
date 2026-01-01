import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # Draw rectangle
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

            # Compute center of rectangle
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = approx[0][0]

            # Overlay digit "9"
            cv2.putText(
                frame,
                "9",
                (cx - 10, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3
            )
            break

    cv2.imshow("Rectangle with 9 Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
