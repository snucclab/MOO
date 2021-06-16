min_seq = []
for idx, value in enumerate({seq}):
    if type(value) is str:
        if (idx > 0) and (type({seq}[idx-1])) is str:
            min_seq = {seq}
            break
        elif idx < len({seq})//2:
            min_seq = {seq}[len({seq})//2:]
        else :
            min_seq = {seq}[:len({seq})//2]
if min_seq == [] :
    min_seq = {seq}[len({seq})//2:]
{result} = (min_seq * (int({index}/len(min_seq)) + 1))[{index}-1]