list_1 = [
    [4, 19], [64, 12], [38, 14], [22, 45], [23, 33], [1, 46], [56, 22], [38, 19], [4, 3], [20, 57],
    [44, 18], [49, 27], [52, 49], [55, 50], [23, 27], [48, 40], [30, 7], [22, 31], [5, 55], [33, 25],
    [41, 3], [45, 61], [44, 41], [48, 57], [30, 56], [51, 29], [10, 20], [59, 63], [54, 67], [32, 19],
    [5, 18], [5, 21], [32, 55], [5, 24], [31, 26], [0, 33], [15, 20], [56, 53], [57, 22], [0, 60],
    [53, 8], [45, 4], [63, 45], [54, 52], [2, 19], [24, 31], [25, 18], [9, 12], [7, 4], [0, 14]
]

list_2 = [
    1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
    1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1
]

group_1 = []
group_0 = []

for i in range(len(list_2)):
    if list_2[i] == 1:
        group_1.append(list_1[i])
    else:
        group_0.append(list_1[i])

combined_result = [group_0, group_1]

print(combined_result)
