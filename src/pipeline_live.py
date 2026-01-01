import cv2
import torch
from pathlib import Path

from .model.model import ConvNet
from .model.solver import Sudoku as solve_sudoku_algorithm
from .preprocess.build_features import process_sudoku_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_grid(model, cell_images):
    model.eval()
    grid = [[0] * 9 for _ in range(9)]

    with torch.no_grad():
        for i in range(9):
            for j in range(9):
                img = cell_images[i * 9 + j]
                tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
                tensor = tensor.repeat(1, 3, 1, 1).to(device)

                output = model(tensor)
                grid[i][j] = torch.argmax(output, dim=1).item()

    return grid


def overlay_digits(image, grid, cell_coords, color=(0, 255, 0)):
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                x, y, w, h = cell_coords[i * 9 + j]
                cv2.putText(
                    image,
                    str(grid[i][j]),
                    (x + w // 3, y + 2 * h // 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
    return image


def main():
    model_path = Path("models/50epochs_convnet_sudoku_only.pkl")

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    cap = cv2.VideoCapture(0)

    solved_grid = None
    cell_coords = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # --- Solve only once ---
        if solved_grid is None:
            cells, coords, warped = process_sudoku_image(frame)

            if cells is not None:
                grid = predict_grid(model, cells)
                solution = [row[:] for row in grid]

                if solve_sudoku_algorithm(solution, 0, 0):
                    solved_grid = solution
                    cell_coords = coords
                    print("Sudoku solved and locked")

        # --- Overlay every frame ---
        if solved_grid is not None and cell_coords is not None:
            display = overlay_digits(display, solved_grid, cell_coords)

        cv2.imshow("Live AR Sudoku Solver", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
