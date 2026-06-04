import random
import tqdm
import numpy as np

import torch
import torch.optim as optim
from AutoRegressiveWrapper import AutoRegressiveWrapper
from SimpleTransformer import SimpleTransformer
import Utils
import sys
import math

# ------constants------------
NUM_BATCHES = int(1e6)
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 3e-4
VALIDATE_EVERY  = 500
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LENGTH = 1024
#---------------------------

def decode_token(token): # convert token to character
    return str(chr(max(32, token)))

def decode_tokens(tokens): # convert sequence of characters to tokens
    return ''.join(list(map(decode_token, tokens)))


def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    simple_transformer = SimpleTransformer(
        dim = 512, # embedding
        num_unique_tokens = 256,  # for character level modeling
        num_layers = 8,
        heads = 8,
        max_seq_len = SEQ_LENGTH,
        causal = True,
    )

    model = AutoRegressiveWrapper(simple_transformer)
    model.cuda()
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")

    train_loader, val_loader, val_dataset = Utils.get_loaders_enwiki8(SEQ_LENGTH, BATCH_SIZE)
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)      # optimizer

    # --------training---------
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
        model.train()
        total_loss = 0
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()
        if (i%100 == 0):
            print(f'training loss: {loss.item()} -- iteration = {i}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            total_len2 = 0
            total_loss2 = 0
            val_count = 1000  # number of validations to compute average BPC
            with torch.no_grad():
                for v in range(0,val_count):
                    loss = model(next(val_loader))
                    total_loss += loss.item()
                    loss_m = loss.mean()
                    total_loss2 += SEQ_LENGTH * loss.float().item() #seq_len
                    total_len2 += SEQ_LENGTH
                print(f'----------validation loss: {total_loss/val_count}')
                print(f'Perplexity : {math.exp(total_loss/val_count)}, BPC: {total_loss/val_count*np.log2(2.7173)}')
                bpc2 = (total_loss2/total_len2)/math.log(2)
                print("BPC 2 = ", bpc2)
                total_loss = 0
        if i % GENERATE_EVERY == 0:  
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            input_start_sequence = decode_tokens(inp)
            print("----------start input------------------")
            print(f'%s \n\n', (input_start_sequence))
            print("----------end of start input-----------")
            sample = model.generate(inp, GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            print("----------generated output-----------")
            print(output_str)
            print("----------end generated output-----------")

if __name__ == "__main__":
    sys.exit(int(main() or 0))
