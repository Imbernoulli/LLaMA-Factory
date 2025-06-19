from jload import jload, jsave
import random

data = jload("/home/test/test12/bohan/ws/LLaMA-Factory/data/alpaca_en_demo.json")

for d in data:
    d["weight"] = random.uniform(1, 10)

jsave(data, "/home/test/test12/bohan/ws/LLaMA-Factory/data/alpaca_en_demo.json")
