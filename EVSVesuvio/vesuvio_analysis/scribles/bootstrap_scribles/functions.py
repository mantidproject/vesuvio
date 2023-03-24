

def someFun(a, flag):

    if flag:
        a = 3
    print(a)
    return a


print(someFun(5, False))
print(someFun(5, True))


A = [[1, 1], [2, 3]]

B = [a[0] for a in A]
print(B)

print(~False)

s = "15."
print(s)
print(int(float(s)))


def multipleArgs():
    return 2, 3, 4

res = multipleArgs()
print(res)

someList = [1, 2, 3]
values = []
ICs = []

booleans = [a in someList for a in values]
print(all(booleans))
for v, ic in zip(values, ICs):
    print("Some command!") 



def someFunc():
     return

someFunc(2, 3, 4)