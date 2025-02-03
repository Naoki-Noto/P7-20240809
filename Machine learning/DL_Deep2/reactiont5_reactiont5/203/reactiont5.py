import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
#import tensorflow as tf
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
#tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig, PreTrainedModel
from transformers import get_linear_schedule_with_warmup
from rdkit import Chem

import pickle

class ReactionT5Yield(PreTrainedModel):
    config_class  = AutoConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)
        self.model.resize_token_embeddings(self.config.vocab_size)
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size//2)
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.hidden_size//2)
        self.fc3 = nn.Linear(self.config.hidden_size//2*2, self.config.hidden_size)
        self.fc4 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc5 = nn.Linear(self.config.hidden_size, 1)

        self._init_weights(self.fc1)
        self._init_weights(self.fc2)
        self._init_weights(self.fc3)
        self._init_weights(self.fc4)
        self._init_weights(self.fc5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
            }
        encoder_outputs = self.model.encoder(**inputs)
        encoder_hidden_states = encoder_outputs[0]
        outputs = self.model.decoder(input_ids=torch.full((input_ids.size(0), 1),
                                                          self.config.decoder_start_token_id,
                                                          dtype=torch.long).to(input_ids.device),
                                     encoder_hidden_states=encoder_hidden_states)
        last_hidden_states = outputs[0]
        output1 = self.fc1(last_hidden_states[:, 0, :]) #.view(-1, self.config.hidden_size)削除
        output2 = self.fc2(encoder_hidden_states[:, 0, :]) #.view(-1, self.config.hidden_size)削除
        output = self.fc3(torch.hstack((output1, output2)))
        output = self.fc4(output)
        output = self.fc5(output)
        return output * 100

def custom_collate(batch):
    data_list, target_list = zip(*batch)
    batch_data = {key: torch.stack([d[key] for d in data_list]) for key in data_list[0]}
    batch_target = torch.stack(target_list)
    return batch_data, batch_target

class ReactionT5Dataset(Dataset):
    def __init__(self, input_ids, attention_masks, targets):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long).clone().detach(),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long).clone().detach()
        }, torch.tensor(self.targets[idx], dtype=torch.float32).clone().detach()

def canonicalize(smiles):
    try:
        new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except:
        new_smiles = None
    return new_smiles

def make_input_list(react, prod, smiles_list):
    input_list = []
    for ops in smiles_list:
        canonicalize_ops = canonicalize(ops)
        if canonicalize_ops == None:
            print(f'{ops} is not canonicalized')
        input_list.append('REACTANT:' + react + 'REAGENT:' + canonicalize_ops + 'PRODUCT:' + prod)
    return input_list

def tokenize_smiles(smiles_list):
    encodings = tokenizer(smiles_list, padding=True, truncation=True, max_length=300, return_tensors="pt")
    return encodings['input_ids'].tolist(), encodings['attention_mask'].tolist()

def calculate_statistics(group):
    r2_test = group['r2_test']
    r2_test_dict = {f'run{i}': r2_test_val for i, r2_test_val in enumerate(r2_test)}
    return pd.Series({
        **r2_test_dict,
        'r2_test_mean': np.mean(r2_test),
        'r2_test_max': np.max(r2_test),
        'r2_test_min': np.min(r2_test),
        'r2_test_std': np.std(r2_test, ddof=0),
    })

torch.manual_seed(0)

epochs = 500
wd = 0.0 #5
batch_size=32
max_grad_norm=1000
batch_scheduler=True
lr=0.01
eps=1e-6
num_warmup_steps=0
use_apex=True

tokenizer = AutoTokenizer.from_pretrained('sagawa/ReactionT5v2-yield')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = []
results_history=[]
n = 0
for random_state in range(10):
    torch.manual_seed(0)
    print(f'run{n}')
    for t in ["Yield_CO_s", "Yield_CO_l", "Yield_CO_cl"]:
        torch.manual_seed(0)
        #reractantとproductを定義
        prod = canonicalize('OC1=CC=C(C#N)C=C1')
        if t == 'Yield_CO_s':
            react = canonicalize('BrC1=CC=C(C#N)C=C1')
            lr = 1e-4
        elif t == 'Yield_CO_l':
            react = canonicalize('BrC1=CC=C(C#N)C=C1')
            lr = 1e-4
        elif t == 'Yield_CO_cl':
            react = canonicalize('ClC1=CC=C(C#N)C=C1')
            lr = 1e-4#1e-5

        df = pd.read_csv('data_real.csv')
        y = df[t]
        input_list = make_input_list(react, prod, df["SMILES"].tolist()) #T5の入力に適した形に変換
        token_ids_list, attention_masks_list = tokenize_smiles(input_list)

        data_train, data_test,  attention_train, attention_test,  target_train, target_test = train_test_split(
            token_ids_list, attention_masks_list, y, test_size=0.5, random_state=random_state
        )

        data_train, data_valid,  attention_train, attention_valid,  target_train, target_valid = train_test_split(
            data_train, attention_train, target_train, test_size=0.1, random_state=random_state
        )

        ####
        best_state_dict=None
        best_valid_loss=None
        best_epoch=0
        train_loss_history_retry=[]
        valid_loss_history_retry=[]
        for retry_cnt in range(3):
            lr_list=[1e-5,1e-6,1e-7]
            lr=lr_list[retry_cnt]
            ###
            target_train_tensor = torch.tensor(target_train.values, dtype=torch.float32)
            target_test_tensor = torch.tensor(target_test.values, dtype=torch.float32)
            target_valid_tensor = torch.tensor(target_valid.values, dtype=torch.float32)
            
            train_dataset = ReactionT5Dataset(data_train, attention_train, target_train_tensor)
            test_dataset = ReactionT5Dataset(data_test, attention_test, target_test_tensor)
            valid_dataset = ReactionT5Dataset(data_valid, attention_valid, target_valid_tensor)
    
            train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate,shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=custom_collate)
            test_loader  = DataLoader(test_dataset,  batch_size=batch_size, collate_fn=custom_collate)
    
            config = AutoConfig.from_pretrained('sagawa/ReactionT5v2-yield')
            model = ReactionT5Yield(config).from_pretrained('sagawa/ReactionT5v2-yield')
            #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=wd,  eps=eps, betas=(0.9, 0.999))
            """
            num_train_steps = int(len(train_loader) / batch_size * epochs)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=use_apex)
            """
            
            criterion = nn.MSELoss()
    
            model.to(device)
            print(t)
            train_loss_history=[]
            valid_loss_history=[]
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for batch in train_loader:
                    data, target = batch
                    input_ids = data['input_ids'].to(device)
                    attention_mask = data['attention_mask'].to(device)
                    target = target.to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, target.view(-1, 1))
                    loss.backward()
                    """
                    scaler.scale(loss).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    #optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    """
                    optimizer.step()
                    optimizer.zero_grad()
        
                    #if batch_scheduler:
                    #    scheduler.step()
                    ###
                    epoch_loss += loss.item()
                epoch_valid_loss = 0
                with torch.no_grad():
                    for batch in valid_loader:
                        data, target = batch
                        input_ids = data['input_ids'].to(device)
                        attention_mask = data['attention_mask'].to(device)
                        target = target.to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs, target.view(-1, 1))
                        
                        epoch_valid_loss += loss.item()
                train_loss_history.append(epoch_loss / len(train_loader))
                valid_loss_history.append(epoch_valid_loss / len(valid_loader))
                if best_valid_loss is None or best_valid_loss > epoch_valid_loss / len(valid_loader):
                    best_valid_loss=epoch_valid_loss / len(valid_loader)
                    best_state_dict=model.state_dict()
                    best_epoch=epoch + 1
                    
                print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)},  Loss: {epoch_valid_loss / len(valid_loader)}")
            ##retry
            train_loss_history_retry.append(train_loss_history)
            valid_loss_history_retry.append(valid_loss_history)
        model.load_state_dict(best_state_dict)
        model.eval()
        def predict(loader):
            all_preds = []
            with torch.no_grad():
                for batch in loader:
                    data, target = batch
                    input_ids = data['input_ids'].to(device)
                    attention_mask = data['attention_mask'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    all_preds.append(outputs.cpu().numpy())
            return np.concatenate(all_preds)

        pred_train = predict(train_loader)
        pred_test = predict(test_loader)
        r2_train_score = metrics.r2_score(target_train, pred_train)
        r2_test_score = metrics.r2_score(target_test, pred_test)
        rmse_train_score = metrics.root_mean_squared_error(target_train, pred_train)
        rmse_test_score = metrics.root_mean_squared_error(target_test, pred_test)

        results.append({'target': t, 'r2_test': r2_test_score, 'rmse_test':rmse_test_score})
        results_history.append({'target': (t,random_state), 'best_epoch':best_epoch,'train_loss_history': train_loss_history_retry, 'valid_loss_history': valid_loss_history_retry, 'best_state_dict':best_state_dict})
    n += 1
    

with open('results.pkl', 'wb') as fp:
    pickle.dump(results, fp)
with open('results_history.pkl', 'wb') as fp:
    pickle.dump(results_history, fp)

results_df = pd.DataFrame(results)
gen_results = results_df.groupby(['target']).apply(calculate_statistics).reset_index()

