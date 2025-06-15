ch_mult = [1, 2, 2, 2]
ch = 128
chs = [ch]  # initial channel size
now_ch = ch  # current channel size
for i, mult in enumerate(ch_mult):
    out_ch = ch * mult
    print(f"Block {i}: in_ch={now_ch}, out_ch={out_ch}")
    for _ in range(4):  # append 4 ResBlocks
        print(f"  ResBlock: in_ch={now_ch}, out_ch={out_ch}")
        now_ch = out_ch
        chs.append(now_ch)
    if i != len(ch_mult) - 1:  # not the last block
        print(f"  DownSample: in_ch={now_ch}")
        chs.append(now_ch)

print("Final channel sizes:", chs)