import numpy as np

def compute_wect(img_matrix, filtration, fe="MAX"):
    weights = np.concatenate(img_matrix)
    change_points = []
    for s in filtration:
        height = s.data
        weight = __get_simplex_weight(s, weights, fe)
        
        if len(change_points) == 0:
            change_points.append([height, weight])
            print("new change point\ns =",s , "curr =", weight)
            continue

        if height == change_points[-1][0]:
            change_points[-1][1] += (-1)**(s.dimension())*weight
            print("s =", s, "weight +=", (-1)**(s.dimension())*weight, "now", change_points[-1][1])
        else:
            change_points.append([height, weight])
            print("new change point\ns =",s , "curr =", weight)
    
    return change_points

def __contains_zero_weights(s, weights):
    for i in range(0,s.dimension()+1):
        if weights[s[i]] == 0:
            # print(s, "had a zero weight")
            return True
    return False

def __get_maxfe_weight(s, weights):
    if __contains_zero_weights(s, weights):
        return 0

    d = s.dimension()
    if d == 0:
        return weights[s[0]]
    elif d == 1:
        return max([weights[s[0]], weights[s[1]]])
    else:
        return max([weights[s[0]], weights[s[1]], weights[s[2]]])
    
def __get_simplex_weight(s, weights, fe):
    if fe == "MAX":
        return __get_maxfe_weight(s, weights)
    elif fe == "MIN":
        return __get_minfe_weight(s, weights)
    elif fe == "AVG":
        return __get_avgfe_weight(s, weights)
    elif fe == "EC":
        return 1
    else:
        raise NotImplementedError(f"unknown function extension \"{fe}\"")
    

def __get_minfe_weight(s, weights):
    if __contains_zero_weights(s, weights):
        return 0
    
    d = s.dimension()
    if d == 0:
        return weights[s[0]]
    elif d == 1:
        return min([weights[s[0]], weights[s[1]]])
    else:
        return min([weights[s[0]], weights[s[1]], weights[s[2]]])


def __get_avgfe_weight(s, weights):
    if __contains_zero_weights(s, weights):
        return 0
    
    d = s.dimension()
    if d == 0:
        return weights[s[0]]
    elif d == 1:
        return sum([weights[s[0]], weights[s[1]]]) / 2
    else:
        return sum([weights[s[0]], weights[s[1]], weights[s[2]]]) / 3