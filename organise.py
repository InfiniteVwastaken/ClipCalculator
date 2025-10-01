from tqdm import tqdm
files = ["data.adj", "data.adv", "data.noun", "data.verb"]

words = set()
for i in files:
    with open("raw_data/" + i, "r") as f:
        for j in f.readlines():
            word = j.split(" ")[4]
            words.add(word.replace("_", " ").split("(")[0])


print(len(words))
with open("data", "w") as f:
    for i in tqdm(list(words)):
        f.write(i + "\n")