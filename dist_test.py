from imagewect.wect import distance_between_wects_unidirectional


w1 = [[0,1], [1, 4], [2,2]]
w2 = [[0,2]]

for i in range(0,250):
    d = distance_between_wects_unidirectional(w1, w2, -1, 3)
    print(d)