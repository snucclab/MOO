{result} = 0

base_line = 0
int_check = 0
for idx, value in enumerate({seq}):
    if type(value) == int:
        int_check += 1
    else :
        int_check = 0
        base_line = idx + 1
    
    if int_check == 3:
        break

diff_1 = {seq}[base_line+1] - {seq}[base_line]
diff_2 = {seq}[base_line+2] - {seq}[base_line + 1]
diff = diff_2 - diff_1

if {index} != 0:
    if diff != 0: 
        {result} = {seq}[0] + int(diff * {index}*({index} -1) / 2) 
    else:
        {result} = {seq}[0] + diff_1 * ({index} - 1)
else :
     if diff != 0:
        {result} = {seq}[1] - (diff_1 - diff)
    else:
        {result} = {seq}[1] - diff_1

