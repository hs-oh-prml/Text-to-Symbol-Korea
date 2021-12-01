from jamo import h2j 
from g2pk import G2p
from tqdm import tqdm 
from text import symbol_to_sequence
import numpy as np 
import os 

f = open("./demo.txt", "r")
w = open("./demo_meta.txt", "w")
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
lines = f.readlines()
g2p = G2p()
spk = {}
for idx, line in enumerate(tqdm(lines)):
    s, t = line.split("|")

    if s not in spk:
        spk[s] = 0
    else:
        spk[s] += 1
    sentence = t.replace("\n", "")
    words = sentence.split(" ")
    p = ""
    for word in words:
        g_w = g2p(word)
        p += h2j(g_w)
    ph = "{" + " ".join(p) + "}"
    symbol = symbol_to_sequence(ph)
    symbol = np.array(symbol)
    np.save("./output/{}_{}.npy".format(s, spk[s]), symbol)
    w_line = "{}_{}|{}|{}".format(s, spk[s], ph, t)
    w.write(w_line)

