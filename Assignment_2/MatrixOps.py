import sys
import torch

def main():
    a = torch.arange(2*2).reshape(2,2)
    # print(a)
    b = torch.arange(2*2).reshape(2,2)
    # print(b)
    c = a * b
    # print(c)

    # e = torch.arange(2*3).reshape(2,3)
    # # print(e)
    # f = torch.matmul(a,e)  # matrix multiplication
    # # print(f)
    # f1 = a @ e
    # print("f1: ", f1)

    # f1 = torch.unsqueeze(f,dim=0)  # add a dimension in the beginning
    # print("f1 with additional dimension: ", f1)
    # print(f1.shape)
    
    # f1t = torch.transpose(f1,1,2)
    # print("f1t : ", f1t)
    # print(f1t.shape)

    #--------batch matrix mult------------
    tensor1 = torch.randn(10, 3, 4)
    print("tensor1 \n", tensor1)
    tensor2 = torch.randn(10, 4, 5)
    print("tensor2 \n", tensor2)
    res = torch.matmul(tensor1, tensor2)
    print(res)
    print(res.shape)




if __name__ == "__main__":
    sys.exit(int(main() or 0))
