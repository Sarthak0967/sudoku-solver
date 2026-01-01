import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=150,
        minLineLength=150,
        maxLineGap=20
    )

    h_lines = []
    v_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if abs(y1 - y2) < 10:
                h_lines.append(line[0])
            elif abs(x1 - x2) < 10:
                v_lines.append(line[0])

    # Draw detected lines
    for x1, y1, x2, y2 in h_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for x1, y1, x2, y2 in v_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sudoku heuristic
    if len(h_lines) >= 8 and len(v_lines) >= 8:
        cv2.putText(
            frame,
            "SUDOKU GRID DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

    cv2.imshow("Sudoku Detection", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
