try:
    if type({value}) is int or int({value}) == {value}:
        print(int({value}))
    elif type({value}) is float:
        print('%.2f' % {value})
except:
    print({value})