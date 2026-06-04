import Utils
import sys
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
from MyBertModel import MyBertModel
from torch.nn.utils import clip_grad_norm_
from torch import nn
from tqdm import tqdm
import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_preds, y_true, labels=None):
  cm = confusion_matrix(y_true, y_preds, normalize="true")
  fig, ax = plt.subplots(figsize=(6, 6))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
  disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False) 
  plt.title("Normalized confusion matrix")
  plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, valid_dataloader, test_dataloader, train_df, valid_df, label_names = Utils.get_trainvalidtest_loaders()
    
    model = MyBertModel().to(device) # loads pretrained BERT model and attaches classifier to it

    EPOCHS = 15
    LEARNING_RATE = 3e-6
    BATCH_SIZE = 16
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                num_training_steps=len(train_dataloader)*EPOCHS )

    train_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch_num in range(EPOCHS):
        print('Epoch: ', epoch_num + 1)
        model.train()
        train_loss = 0
        for step_num, batch_data in enumerate(tqdm(train_dataloader,desc='Training')):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_id = input_ids.to(device), mask=att_mask.to(device))
        
            loss = criterion(output, labels.to('cuda'))
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            del loss

            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        train_loss_per_epoch.append(train_loss / (step_num + 1))              

        model.eval() # ------------validation------------
        valid_loss = 0
        valid_pred = []
        with torch.no_grad():
            for step_num_e, batch_data in enumerate(tqdm(valid_dataloader,desc='Validation')):
                input_ids, att_mask, labels = [data.to(device) for data in batch_data]
                output = model(input_id = input_ids.to(device), mask=att_mask.to(device))
                loss = criterion(output, labels.to('cuda'))
                valid_loss += loss.item()
                valid_pred.append(np.argmax(output.cpu().detach().numpy(),axis=-1))
        
        val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
        valid_pred = np.concatenate(valid_pred)

        print("{0}/{1} train loss: {2} ".format(step_num+1, math.ceil(len(train_df) / BATCH_SIZE), train_loss / (step_num + 1)))
        print("{0}/{1} val loss: {2} ".format(step_num_e+1, math.ceil(len(valid_df) / BATCH_SIZE), valid_loss / (step_num_e + 1)))

    from matplotlib import pyplot as plt
    epochs = range(1, EPOCHS +1 )
    fig, ax = plt.subplots()
    ax.plot(epochs,train_loss_per_epoch,label ='training loss')
    ax.plot(epochs, val_loss_per_epoch, label = 'validation loss' )
    ax.set_title('Training and Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()

    print('classifiation report')
    print(classification_report(valid_pred, valid_df['label'].to_numpy(), target_names=label_names))
    plot_confusion_matrix(valid_pred,valid_df['label'].to_numpy(),labels=label_names)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
