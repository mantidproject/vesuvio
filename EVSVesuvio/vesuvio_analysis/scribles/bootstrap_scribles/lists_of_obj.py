
class Fruit:
    def __init__(self, name):
        self.name = name


def makeFruitList(names):
    fruitList = []
    for na in names:
        fruit = Fruit(na)
        fruitList.append(fruit)
    return fruitList

def changeFruits(fruitList, names):
    for fruit, na in zip(fruitList, names):
        fruit.name = na

fruitList = makeFruitList(["APPL", "BANA", "GRAP"])

for fruit in fruitList:
    print(fruit.name)

changeFruits(fruitList, ["PEAR", "MANG"])

for fruit in fruitList:
    print(fruit.name)

print(fruitList)