# numbers1=[2,4,6,8,10]
# numbers2=[1,2,3,4,5,6]

# print(any([num %2==0 for num in numbers2 ]))

# practice
def my_sum(*args):
    if all([type(arg)==int or type(arg)==float for arg in args]):
        total=0
        for num in args:
            total+=num
    return total
print(my_sum(1,2,3,4))
