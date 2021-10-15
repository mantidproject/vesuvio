
# This example shows that all of the functions that I have defined interms of the initila]
# conditions will work if I change the conditions later down the line
class Numbers:
    no  = 3

my_number = Numbers()

def addTwoToNumber():
    return my_number.no + 2

print(addTwoToNumber())
my_number.no = 10
print(addTwoToNumber())

# %%
