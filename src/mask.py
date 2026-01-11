from PIL import Image
import numpy as np
from collections import deque

def image_to_binary_matrix(image_path):
    img = Image.open(image_path).convert("L")  # גרסה בגווני אפור
    img = img.point(lambda x: 0 if x > 170 else 1, mode='1')  # בינארי לפי סף
    return np.array(img, dtype=np.uint8)

def binary_matrix_to_image(matrix, output_path):
    img = Image.fromarray((1 - matrix) * 255).convert("L")  # 1 שחור, 0 לבן
    img.save(output_path)

def flood_fill_from_corners_2(matrix):
    rows, cols = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def bfs(r, c):
        queue = deque()
        if matrix[r][c] == 1:
            queue.append((r, c))
            matrix[r][c] = 0
            visited[r][c] = True

        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and not visited[nr][nc] and matrix[nr][nc] == 1:
                    matrix[nr][nc] = 0
                    visited[nr][nc] = True
                    queue.append((nr, nc))

    # פינות
    corners = [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)]
    for r, c in corners:
        bfs(r, c)

    return matrix


import os
import numpy as np
import cv2
from PIL import Image
from collections import deque

def image_to_binary_matrix(image_path, threshold=235):
    """המרת תמונה לגווני אפור ובינארית לפי סף"""
    img = Image.open(image_path).convert("L")
    img = img.point(lambda x: 0 if x > threshold else 1, mode='1')
    return np.array(img, dtype=np.uint8)

def binary_matrix_to_image(matrix, output_path):
    """המרת מטריצה בינארית לתמונה ושמירה"""
    img = Image.fromarray((1 - matrix) * 255).convert("L")  # שחור = 1
    img.save(output_path)

def flood_fill_from_corners(binary):
    """יוצר מסכה של כל מה שמחובר לפינות השחורות בלבד"""
    rows, cols = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    output_mask = np.zeros_like(binary, dtype=np.uint8)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def bfs(r, c):
        queue = deque()
        queue.append((r, c))
        visited[r, c] = True
        output_mask[r, c] = 1

        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and not visited[nr, nc] and binary[nr, nc] == 1:
                    visited[nr, nc] = True
                    output_mask[nr, nc] = 1
                    queue.append((nr, nc))

    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    for r, c in corners:
        if binary[r, c] == 1 and not visited[r, c]:
            bfs(r, c)

    return output_mask  # פיקסלים שמחוברים לפינה

def apply_mask_to_image(binary, mask):
    """הפיכת כל מה שמחובר לפינה ללבן"""
    result = binary.copy()
    result[mask == 1] = 0  # הפוך ללבן (0)
    return result
