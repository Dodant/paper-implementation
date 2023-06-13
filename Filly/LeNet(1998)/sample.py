
# def num2base(n, base):
#     answer = []
#     while True:
#         answer.append(str(n % base))
#         n //= base
#         if n == 0:
#             break
#     # answer = int(''.join(answer))
#     result = 0
#     for idx, item in enumerate(''.join(answer)[::-1]):
#         result += int(item) * 3 ** idx
#     return result
#
# print(int('10', base=3))
# print(num2base(45, 3))

import torch
print(a := torch.randn(4, 4))
print(torch.einsum('ii', a))
print(torch.einsum('ii->i', a))
