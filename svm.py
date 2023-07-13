import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def __stack_arrays(shapes1, shapes2, num_images, independent=False):
    assert(len(shapes1) == len(shapes2))
    x0s, y0s, x1s, y1s = [], [], [], []
    for i in range(0,len(shapes1)):
        x0s.append(shapes1[f"{i}"])
        y0s.append(0)

        x1s.append(shapes2[f"{i}"])
        y1s.append(1)

    if independent:
        X = np.array(x0s)
        ys = y0s + y1s
        y = np.random.choice(ys, size=(X.shape[0],1), replace=False)
    else:
        xs = x0s + x1s
        X = np.array(xs)
        ys = y0s + y1s
        y = np.array(ys)

    p = np.random.permutation(X.shape[0])
    X, y = X[p], y[p]
    return X[:num_images][:], y[:num_images][:]

def __test(model, X, y, num_runs=100):
        results = np.zeros((num_runs,))
        for i in range(0, num_runs):
            x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, shuffle=True)
            model.fit(x_train, np.ravel(y_train))
            preds = model.predict(x_test)
            results[i] = accuracy_score(y_test, preds)
        return results

def vs_uniform_classification_with_SVM(num_runs=100, num_images=250, c=20, fe="", directions=8):
    shapes = ["square", "annulus", "circle", "clusters", "square_annulus", "swiss_cheese", "tetris"]
    distributions = ["uniform", "n17", "n25", "n50"]

    for shape in shapes:
            for dist in distributions:
                shapes1 = np.load(f"data/wects_disc/{fe}/{directions}dir/{shape}/uniform.npz")
                shapes2 = np.load(f"data/wects_disc/{fe}/{directions}dir/{shape}/{dist}.npz")

                model = SVC(C=c)
                ind = False
                if dist == "uniform":
                    ind = True
                X, y = __stack_arrays(shapes1, shapes2, num_images=num_images, independent=ind)
                res = __test(model, X, y, num_runs=num_runs)

                with open("data/results_svm/vs_uniform.csv", "a") as outfile:
                    outfile.write(f"{fe},{directions},{shape},{dist},{np.mean(res)},{np.std(res)}\n")
                print("finished", shape, dist)

def shapewise_classification_with_SVM(num_runs=100, num_images=250, c=20, fe="", directions=8):
    shapes = ["square", "annulus", "circle", "clusters", "square_annulus", "swiss_cheese", "tetris"]
    distributions = ["uniform", "n17", "n25", "n50"]

    for shape in shapes:
            for shape2 in shapes:
                shapes1 = np.load(f"data/wects_disc/{fe}/{directions}dir/{shape}/uniform.npz")
                shapes2 = np.load(f"data/wects_disc/{fe}/{directions}dir/{shape2}/uniform.npz")

                model = SVC(C=c)
                ind = False
                if shape == shape2:
                    ind = True
                X, y = __stack_arrays(shapes1, shapes2, num_images=num_images, independent=ind)
                res = __test(model, X, y, num_runs=num_runs)

                with open("data/results_svm/shapewise.csv", "a") as outfile:
                    outfile.write(f"{fe},{directions},{shape},{shape2},{np.mean(res)},{np.std(res)},uniform\n")
                print("finished", shape, shape2)

def compare_all(shapes, distributions, fe, num_directions, num_images, num_runs, c):
     for shape1 in shapes:
        for shape2 in shapes:
            for dist1 in distributions:
                for dist2 in distributions:
                    shapes1 = np.load(f"data/wects_disc/{fe}/{num_directions}dir/{shape1}/{dist1}.npz")
                    shapes2 = np.load(f"data/wects_disc/{fe}/{num_directions}dir/{shape2}/{dist2}.npz")

                    model = SVC(C=c)
                    ind = False
                    if dist1 == dist2:
                        ind = True
                    X, y = __stack_arrays(shapes1, shapes2, num_images=num_images, independent=ind)
                    res = __test(model, X, y, num_runs=num_runs)

                    with open("data/results_svm/everything_pairwise.csv", "a") as outfile:
                        outfile.write(f"{fe},{num_directions},{shape1},{dist1},{fe},{num_directions},{shape2},{dist2},{np.mean(res)},{np.std(res)}\n")
                    print("finished", f"{fe},{num_directions},{shape1},{dist1},{fe},{num_directions},{shape2},{dist2}")
     


def pairwise_classification_with_SVM(num_runs=100, num_images=250, c=20):
    shapes = ["square", "annulus", "circle", "clusters", "square_annulus", "swiss_cheese", "tetris"]
    distributions = ["uniform", "n17", "n25", "n50"]
    fes = ["MAXfe", "AVGfe"]
    directions = [8, 15]

    for fe in fes:
         for direction in directions:
              compare_all(shapes, distributions, fe, direction, num_images, num_runs, c)


def compare_different_directions(num_runs=100, num_images=250, c=20):
     shapes = ["square", "annulus", "circle", "clusters", "square_annulus", "swiss_cheese", "tetris"]
     dist = "n25"
     dir_numbers = [30]
     fe = "AVGfe"
     for shape in shapes:
          for dir_number in dir_numbers:
            shapes1 = np.load(f"data/wects_disc/{fe}/{dir_number}dir/{shape}/uniform.npz")
            shapes2 = np.load(f"data/wects_disc/{fe}/{dir_number}dir/{shape}/{dist}.npz")

            model = SVC(C=c)
            ind = False
            if dist == "uniform":
                ind = True
            X, y = __stack_arrays(shapes1, shapes2, num_images=num_images, independent=ind)
            res = __test(model, X, y, num_runs=num_runs)

            # with open("data/results_svm/change_directions.csv", "a") as outfile:
            #     outfile.write(f"{fe},{dir_number},{shape},uniform,{fe},{dir_number},{shape},{dist},{np.mean(res)},{np.std(res)}\n")
            #     print("finished", f"{fe},{dir_number},{shape},uniform,{fe},{dir_number},{shape},{dist}")
            print("dir", dir_number, shape, "uniform vs.", dist, ". avg:", np.mean(res), ". std:", np.std(res))
          
    
    

compare_different_directions()
