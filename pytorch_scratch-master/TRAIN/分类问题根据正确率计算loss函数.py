import torch

output = torch.tensor([ [0.1,0.2],
                        [0.3,0.4]
                        ])
# 根据维度，0为列，1为行。返回最大索引
print(output.argmax(dim=1))

targets = torch.tensor([0,1])
preds = output.argmax(dim=1)

print(targets == preds)
# True为1, False为0
diff = (targets == preds).sum()
print(diff)
print("正确率：{}".format(diff/len(targets)))