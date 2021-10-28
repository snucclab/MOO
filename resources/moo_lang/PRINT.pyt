if type({value}) is int or int({value}) == {value}:
    print(int({value}))
else:
    print('%.2f' % {value})
