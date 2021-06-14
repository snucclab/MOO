result_list = []
for num in {ls1}:
    for div in {ls2}:
        if num // div == 0:
            result_list.append(num)
{result} = len(result_list)
