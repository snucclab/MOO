min_seq = []
for idx, value in enumerate({seq}):
    if type(value) is str:
        if idx < len({seq})/2:
            min_seq = {seq}[int(len({seq})/2):]
        else :
            min_seq = {seq}[:int(len({seq})/2)]

if min_seq == [] :
    min_seq = {seq}[int(len({seq})/2):]

print(min_seq)
print({index})
{result} = (min_seq * (int({index}/len(min_seq)) + 1))[{index}-1]
