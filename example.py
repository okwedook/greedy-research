from spaces import Box

box = Box([1, 3], [5, 7])

for _ in range(1000):
    sample = box.genSample()
    print(sample)