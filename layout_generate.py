from random import randint
configs = list()
xMax = 7
yMax = 4
num_layout = 500
while len(configs) < num_layout:
    components = set()
    while len(components) < 3:
        components.add((randint(0, xMax), randint(0, yMax)))
    new_config = tuple(components)
    if new_config not in configs:
        configs.append(new_config)

for con in configs:
    print "initial position:({},{}), true goal:({},{}), dummy goal:({},{})".format(con[0][0],con[0][1], con[1][0],con[1][1], con[2][0],con[2][1])
