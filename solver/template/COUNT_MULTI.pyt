result_list = []
for num in {ls1}:
    count = 0
    for div in {ls2}:
        if num % div == 0:
            count += 1
    if count == len({ls2}):
        result_list.append(num)
{result} = len(result_list)