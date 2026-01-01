import cv2
import numpy as np
import random

# ---------- Helper Functions ----------

def reorder(points):
    points = points.reshape((4, 2))
    new = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(1)
    diff = np.diff(points, axis=1)

    new[0] = points[np.argmin(s)]      # top-left
    new[2] = points[np.argmax(s)]      # bottom-right
    new[1] = points[np.argmin(diff)]   # top-right
    new[3] = points[np.argmax(diff)]   # bottom-left
    return new

def draw_numbers(img, size):
    cell = size // 9
    for i in range(9):
        for j in range(9):
            num = random.randint(1, 9)
            x = j * cell + cell // 3
            y = i * cell + 2 * cell // 3
            cv2.putText(img, str(num), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

# ---------- Main ----------

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    edges = cv2.Canny(thresh, 50, 150)

    # ----- Hough Line Detection -----
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=150,
        minLineLength=150,
        maxLineGap=20
    )

    h_lines, v_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:
                h_lines.append(line[0])
            elif abs(x1 - x2) < 10:
                v_lines.append(line[0])

    sudoku_detected = len(h_lines) >= 8 and len(v_lines) >= 8

    # ----- Find Outer Grid -----
    if sudoku_detected:
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        sudoku_cnt = None

        for cnt in contours:
            if cv2.contourArea(cnt) < 5000:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                sudoku_cnt = approx
                break

        if sudoku_cnt is not None:
            pts = reorder(sudoku_cnt)

            size = 450
            dst = np.array([
                [0, 0],
                [size, 0],
                [size, size],
                [0, size]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(original, matrix, (size, size))

            # ----- Overlay Grid -----
            cell = size // 9
            for i in range(1, 9):
                cv2.line(warped, (0, i * cell), (size, i * cell), (255, 0, 0), 1)
                cv2.line(warped, (i * cell, 0), (i * cell, size), (255, 0, 0), 1)

            # ----- Overlay Random Numbers -----
            draw_numbers(warped, size)

            # ----- Warp Back to Camera View (AR) -----
            inv_matrix = cv2.getPerspectiveTransform(dst, pts)
            overlay = cv2.warpPerspective(warped, inv_matrix,
                                          (frame.shape[1], frame.shape[0]))

            mask = np.zeros_like(gray)
            cv2.fillConvexPoly(mask, pts.astype(int), 255)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            overlay_fg = cv2.bitwise_and(overlay, overlay, mask=mask)

            frame = cv2.add(frame_bg, overlay_fg)

            cv2.drawContours(frame, [sudoku_cnt], -1, (0, 255, 0), 3)

    cv2.imshow("AR Sudoku Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
