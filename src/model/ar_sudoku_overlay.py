import cv2
import numpy as np

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


def extract_matrix(warped, size):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cell = size // 9

    matrix = [[0]*9 for _ in range(9)]

    for i in range(9):
        for j in range(9):
            x, y = j * cell, i * cell
            cell_img = gray[y:y+cell, x:x+cell]

            cell_img = cv2.resize(cell_img, (28, 28))
            _, thresh = cv2.threshold(
                cell_img, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Ignore borders
            margin = 4
            roi = thresh[margin:-margin, margin:-margin]

            if cv2.countNonZero(roi) > 50:
                matrix[i][j] = 1  # digit present

    return matrix


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
        edges, 1, np.pi / 180,
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

            matrix_p = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(original, matrix_p, (size, size))

            sudoku_matrix = extract_matrix(warped, size)

            # ---- Print matrix ----
            print("\nDetected Sudoku Matrix:")
            for row in sudoku_matrix:
                print(row)

            cv2.drawContours(frame, [sudoku_cnt], -1, (0, 255, 0), 3)

            cv2.imshow("Warped Sudoku", warped)

    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
