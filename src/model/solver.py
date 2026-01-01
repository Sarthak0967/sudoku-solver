N = 9

def puzzle(a):
    for i in range(N):
        for j in range(N):
            print(a[i][j], end=" ")
            print()
            
def solve(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
        
    for x in range(9):
        if grid[x][col] == num:
            return False
        
    startRow = row - row % 3
    startCol = col - col % 3
    
    for i in range(3):
        for j in range(3):
            if grid[i+startRow][j+startCol] == num:
                return False
    return True

def Sudoku(grid, row, col):
    """
    Solve Sudoku puzzle using backtracking algorithm.
    
    Args:
    grid (list of list of int): 9x9 Sudoku grid with 0s for empty cells.    
    row (int): Current row index.   
    col (int): Current column index.
    
    Returns:
    bool: True if the puzzle is solved, False otherwise.
    """
    
    if row == N-1 and col == N:
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Sudoku(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if solve(grid, row, col, num):
            grid[row][col] = num
            if Sudoku(grid, row, col + 1):
                return True
            grid[row][col] = 0
    return False