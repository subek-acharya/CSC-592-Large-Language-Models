import sys
import torch
from einops import rearrange

def main():
    # A = torch.tensor([[1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #     [9, 10, 11, 12],
    #     [13, 14, 15, 16]])
    # B = torch.tensor([[1, 2, 1, 1],
    #     [3, 4, 2, 5],
    #     [1, 3, 6, 7],
    #     [1, 4, 6, 8]])
    # print(A)
    # print(B)

    # C = torch.einsum('ij, jk -> ik', A, B) # matrix mult.
    # C2 = torch.einsum('ij, kj -> ik', A, B) # Ax(transpose(B) - matrix mult.
    # print(C)
   
    # T = torch.einsum('ij -> ji', A) 
    # print(T)

    # d1 = torch.tensor([3, 5, 7, 9])
    # d2 = torch.tensor([1, 2, 3, 4])
    # # douter = torch.einsum('i, j -> ij', d1, d2) # outer product
    # # print(douter)
    # dinner = torch.einsum('i, i -> ', d1, d2) # inner product
    # # print(dinner)

    # batch_tensor_1 = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    # print(batch_tensor_1)
    # batch_tensor_2 = torch.arange(2 * 4 * 3).reshape(2, 3, 4)
    # print(batch_tensor_2)
    # dmul = torch.einsum('bij, bjk -> bik', batch_tensor_1, batch_tensor_2) # batch matrix multiplication
    # print(dmul)
    # print(dmul.shape)

    q = torch.zeros((2,1024,64)) # 2 is batch size
    q2 = rearrange(q,'b (n s) e->b n s e', s=16)
    print(q2.shape) 
    q3 = rearrange(q2,'b n s e-> (b n) s e')
    print(q3.shape)  




    
	
if __name__ == "__main__":
    sys.exit(int(main() or 0))
