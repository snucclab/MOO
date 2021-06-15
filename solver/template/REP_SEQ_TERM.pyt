min_seq = []
if len({seq}) % 2 == 0:
    if {seq}[:len({seq})//2]=={seq}[len({seq})//2:]:
        for idx, value in enumerate({seq}) :
            if type(value) is str:
                if idx < len({{seq}})//2:
                    min_seq = {seq}[int(len({seq})//2):]
                else :
                    min_seq = {seq}[:int(len({seq})//2)]
            else:
                min_seq = {seq}[int(len({seq})//2):]
    else:
        min_seq = {seq}
else:
    min_seq = {seq}
{result} = (min_seq * (int({index}/len(min_seq)) + 1))[{index}-1]