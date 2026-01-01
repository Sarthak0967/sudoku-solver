import cv2
import numpy as np

from ..common import tools
from ..data import dataio

def perspective_transform(image, corners):
    def order_corner_points(corners):
        corners = [(corner[0], corner[1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = (
            corners[3],
            corners[0],
            corners[1],
            corners[2],
        )
        return (top_l, top_r, bottom_r, bottom_l)
    
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners
    
    width_A = np.sqrt(
        ((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2)
    )
    width_B = np.sqrt(
        ((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2)
    )
    width = max(int(width_A), int(width_B))
    
    height_A = np.sqrt(
        ((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2)
    )
    
    height_B = np.sqrt(
        ((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2)
    )
    
    height = max(int(height_A), int(height_B))
    
    dimensions = np.array(
        [[0,0], [width-1,0], [width-1,height-1], [0,height-1]],
        dtype="float32"
    )
    
    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def extract_cells_with_coords_from_warped_image(image):
    h, w = image.shape[:2]
    cell_h, cell_w = h // 9, w // 9

    cells = []
    for i in range(9):
        for j in range(9):
            x, y = j * cell_w, i * cell_h
            cells.append(
                {
                    "image": image[y : y + cell_h, x : x + cell_w],
                    "coords": (x, y, cell_w, cell_h),
                }
            )
    return cells


def finding_sudoku_mask(image):
    sudoku_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # BLur
    sudoku_blur = cv2.GaussianBlur(sudoku_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        sudoku_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )
    dilate = cv2.dilate(thresh, kernel=np.ones((3, 3), np.uint8), iterations=1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, np.ones((3, 3)))

    return closing


def extract_sudoku_grid(image, mask):
    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find Contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw Contour on the image
    cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)

    peri = cv2.arcLength(largest_contour, True)  # Get the perimeter
    approx = cv2.approxPolyDP(
        largest_contour, 0.02 * peri, True
    )  
    
    if len(approx) != 4:
        rect = cv2.minAreaRect(
            largest_contour
        )  
        box = cv2.boxPoints(
            rect
        )  
        approx = np.array(box)

    corners = approx.reshape(4, 2)  # Reshape to 4x2 array

    return corners


def process_sudoku_image(image, invert_for_mnist_compatibility=True):
    try:
        # Get sudoku box
        mask = finding_sudoku_mask(image.copy())
        corners = extract_sudoku_grid(image.copy(), mask)
        warped = perspective_transform(image, corners)

        # Process cells
        cells_data = extract_cells_with_coords_from_warped_image(warped)
        processed_cells = []
        coords = []

        for cell in cells_data:
            # Convert to grayscale
            gray = cv2.cvtColor(cell["image"], cv2.COLOR_BGR2GRAY)

            # Apply Otsu's thresholding - FIX: Use THRESH_BINARY (not INV) to match MNIST
            if invert_for_mnist_compatibility:
                # MNIST format: black background, white digits
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                # Original format: white background, black digits
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

            # Resize to 28x28 and normalize to [0,1] range
            processed = cv2.resize(binary, (28, 28)) / 255.0

            processed_cells.append(processed)
            coords.append(cell["coords"])

        return processed_cells, coords, warped
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


if __name__ == "__main__":
    config = tools.load_config()
    base = config["base"]
    image_path = base + "sudoku/mixed 2/mixed 2/image2.jpg"

    
    image = cv2.imread(image_path)
    cv2.imshow("Oryginalny obraz", image)

    
    mask = finding_sudoku_mask(image)
    cv2.imshow("Maska", mask)

    
    contour = extract_sudoku_grid(image, mask)
  
    warped = perspective_transform(image, contour)
    cv2.imshow("Wyciete sudoku", warped)

    # Example of using the main processing function
    processed_cells, coords_on_warped, warped_display = process_sudoku_image(
        image, invert_for_mnist_compatibility=True
    )
    if processed_cells:
        print(f"Successfully processed {len(processed_cells)} cells.")
        cv2.imshow("Warped Sudoku for Display", warped_display)
        # You can now use coords_on_warped to draw on warped_display

    print("Naciśnij dowolny klawisz, aby zakończyć...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
