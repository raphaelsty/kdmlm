Knowledge bases with language models.


## 🔧 Installation
```sh
pip install git+https://username:password@github.com/raphaelsty/kdmlm --upgrade
```

## 🤖 Quick start 

```python
from kdmlm import mlm
from kdmlm import datasets

from mkb import datasets as mkb_datasets
from mkb import models as mkb_models
from mkb import evaluation as mkb_evaluation

from transformers import BertTokenizer
from transformers import BertForMaskedLM

from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import TrainingArguments

import torch

import pickle

_ = torch.manual_seed(42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

kb = mkb_datasets.Fb15k237(1, pre_compute = False)

folder = './drive/MyDrive/these/kdmlm/data/wiki_fb15k237/'

dataset = datasets.LoadFromFolder(folder = folder)

train_dataset = datasets.KDDataset(
    dataset=dataset,
    tokenizer=tokenizer,
    entities=kb.entities,
    sep='|'
)

data_collator = datasets.Collator(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir = f'./drive/MyDrive/these/kdmlm/checkpoints',
    overwrite_output_dir = True,
    num_train_epochs = 1,
    per_device_train_batch_size = 125, 
    save_steps = 500, 
    save_total_limit = 1,
    do_train = True,  
    do_predict = True,

)

kb_model = mkb_models.TransE(
    entities=kb.entities, 
    relations=kb.relations,
    hidden_dim=250, 
    gamma=9,
).to('cuda')

negative_sampling_size = 125

alpha = 0.98

evaluation = mkb_evaluation.Evaluation(
    true_triples = kb.true_triples,
    entities   = kb.entities,
    relations  = kb.relations,
    batch_size = 8,
    device     = 'cuda',
)

mlm_trainer = mlm.MlmTrainer(
    model=model, 
    args=training_args,
    data_collator=data_collator, 
    train_dataset=train_dataset,
    tokenizer=tokenizer, 
    kb=kb, 
    kb_model=kb_model,
    negative_sampling_size=negative_sampling_size,
    alpha=alpha, 
    seed=42, 
    kb_evaluation=evaluation,
    eval_kb_every = 3000,
    top_k_size = 20,
    n_random_entities = 20,
    fit_kb=True, 
    fit_bert=False,
    distill=True,
)

mlm_trainer.train()

with open('./drive/MyDrive/these/kdmlm/kb_model.pickle', 'wb') as output_model_file:
    pickle.dump(kb_model.cpu(), output_model_file)

with open('./drive/MyDrive/these/kdmlm/kb_model.pickle', 'rb') as input_model_file:
    kb_model = pickle.load(input_model_file).to('cuda')
```