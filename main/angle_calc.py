import math

import numpy

# test values
x = -6
y = 7
x_mid = 5
y_mid = 10
x2 = 2
y2 = 12
mat1 = [[3], [-1]]
mat2 = [[-4], [7]]


# vector addition
def vec_add(vec1, vec2):
    new_vec = [[], []]
    for row in range(2):
        new_vec[row].append(vec1[row][0] + vec2[row][0])
    return new_vec


# print(mat_add(mat1, mat2))


# line direction vector creation
def dir_vec_create(x_pos, y_pos, mid_x_pos, mid_y_pos):
    pos_vec1 = [[x_pos], [y_pos]]
    neg_pos_vec2 = [[-mid_x_pos], [-mid_y_pos]]
    dir_vec = vec_add(pos_vec1, neg_pos_vec2)
    return dir_vec


# print(dir_vec_create(x, y, x_mid, y_mid))


# vector line length calculation
def line_length(x_pos, y_pos, mid_x_pos, mid_y_pos):
    x_length = mid_x_pos - x_pos
    y_length = mid_y_pos - y_pos
    line_length = numpy.sqrt((x_length) ** 2 + (y_length) ** 2)
    return line_length


# print(line_length(x, y, x_mid, y_mid))


# vector multiplication (dot product, 2x1 matrix only)
def vec_multiplication(vec1, vec2):
    dot_prod = 0
    for row in range(2):
        dot_prod += vec1[row][0] * vec2[row][0]
    return dot_prod


# print(vec_multiplication(mat1, mat2))


# angle calculator
def angle_calc(x1_pos, y1_pos, x2_pos, y2_pos, mid_x_pos, mid_y_pos):
    dir1 = dir_vec_create(x1_pos, y1_pos, mid_x_pos, mid_y_pos)
    dir2 = dir_vec_create(x2_pos, y2_pos, mid_x_pos, mid_y_pos)
    numerator = vec_multiplication(dir1, dir2)
    line1_length = line_length(x1_pos, y1_pos, mid_x_pos, mid_y_pos)
    line2_length = line_length(x2_pos, y2_pos, mid_x_pos, mid_y_pos)
    denom = line1_length * line2_length
    if numerator >= 0:
        theta = math.acos(numerator / denom)
        angle = math.degrees(theta)
    else:
        theta = math.acos(abs(numerator) / denom)
        angle = 180 - math.degrees(theta)
    return angle


# print(angle_calc(x, y, x2, y2, x_mid, y_mid))
