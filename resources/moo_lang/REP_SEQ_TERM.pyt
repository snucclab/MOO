seq_len = len({seq})
period = {seq}
if seq_len % 2 == 0:
    half = seq_len // 2
    left_half = {seq}[:half]
    right_half = {seq}[half:]

    is_half_period = True
    is_left_chosen = True
    for l, r in zip(left_half, right_half):
        l_upper = type(l) is str and len(l) == 1 and l.isupper()
        r_upper = type(r) is str and len(r) == 1 and r.isupper()
        if l_upper or r_upper:
            if l_upper:
                is_left_chosen = False
            continue
        if l != r:
            is_half_period = False
            break

    if is_half_period:
        period = left_half if is_left_chosen else right_half

{result} = period[({index}-1) % len(period)]