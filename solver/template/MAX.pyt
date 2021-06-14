if type({LIST}[0]) == list:
    max_name = {LIST}[0][0]
    max_val = {LIST}[0][1]
    for i in range(1,len({LIST})):
        curr_name = {LIST}[i][0]
        curr_val = {LIST}[i][1]
        if curr_val > max_val:
            max_val = curr_val
            max_name = curr_name
    {result} = max_name
else:
    {result} = max({LIST})
