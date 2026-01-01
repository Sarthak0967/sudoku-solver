import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    # 🔑 CONNECT EDGES
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Prefer true rectangles
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
        else:
            # fallback: bounding box
            x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)

        if 0.6 < aspect_ratio < 1.4:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Draw 9x9 grid
            cell_w = w // 9
            cell_h = h // 9

            for i in range(1, 9):
                cv2.line(frame, (x + i * cell_w, y), (x + i * cell_w, y + h), (255, 0, 0), 1)
                cv2.line(frame, (x, y + i * cell_h), (x + w, y + i * cell_h), (255, 0, 0), 1)

            # Put random digits
            for r in range(9):
                for c in range(9):
                    num = random.randint(1, 9)
                    cx = x + c * cell_w + cell_w // 3
                    cy = y + r * cell_h + 2 * cell_h // 3

                    cv2.putText(
                        frame,
                        str(num),
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            detected = True
            break

    # DEBUG VISUALS
    cv2.imshow("Edges", edges)
    cv2.imshow("AR Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
