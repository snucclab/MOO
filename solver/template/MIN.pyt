if type({LIST}[0]) == tuple:
    min_name = {LIST}[0][0]
    min_val = {LIST}[0][1]
    for i in range(1,len({LIST})):
        curr_name = {LIST}[i][0]
        curr_val = {LIST}[i][1]
        if curr_val < min_val:
            min_val = curr_val
            min_name = curr_name
    {result} = min_name
else:
    {result} = mim({LIST})
