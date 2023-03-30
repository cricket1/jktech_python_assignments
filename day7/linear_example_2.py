import  numpy as np

np.random.seed(1)
labels = ["dog", "cat", "panda"]


image = np.random.rand(5) ## dog
W = np.random.rand(3, 5) + 0.5

scores = W.dot(image)

print(scores)

while True:
    best_label = labels[np.argmax(scores)]
    # add penalty.Line 17 to 20 is loss function
    if best_label != "dog":
        W[1] = W[1]*0.95
        W[2] = W[2]*0.95
        scores = W.dot(image)
        print(scores)
    else:
        break


scores = W.dot(image)
print('prediction is {}'.format(labels[np.argmax(scores)]))
