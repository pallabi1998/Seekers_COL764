import ir_datasets
dataset = ir_datasets.load("trec-cast/v1/2020/judged")
cnt = 0
for query in dataset.queries_iter():
    cnt += 1
    print(query, dir(query))
    break