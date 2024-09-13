import torch
import torch.nn
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import csv
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
from bayes_opt import BayesianOptimization
import warnings
from transformers import BertTokenizer, BertForMaskedLM
import logging


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning,
                        message="Some weights of the model checkpoint at .* were not used when initializing.*")
# 1. Load pre-trained model and tokenizer, define saved model link
MODEL_PATH = './origin_model/bert-base-uncased-tradition'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)

# Data clean
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load the sentiment dictionary and check word embeddings
def load_sentiment_dict_and_check_embeddings(tokenizer, model):
    sentiment_dict = {}
    try:
        with open('./VADER/Senticnet7.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                word = row[0]
                score = float(row[1])
                token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                embeddings = [model.bert.embeddings.word_embeddings.weight[token_id].detach().numpy() for token_id in token_ids]
                sentiment_dict[word] = {'score': score, 'embedding': embeddings}
                # Check
                # if len(sentiment_dict) <= 5:
                #     print(f"Word: {word}, Token IDs: {token_ids}, Sentiment Score: {score}, Embeddings: {embeddings}")
        print(f'Sentiment dictionary loaded successfully with {len(sentiment_dict)} entries.')
    except Exception as e:
        print(f'Failed to load sentiment dictionary: {e}')
    return sentiment_dict
sentiment_dict = load_sentiment_dict_and_check_embeddings(tokenizer, model)

def get_sentiment_score(word):
    return sentiment_dict.get(word, {'score': 0})['score']


class CustomSensitiveTanh(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([a], dtype=torch.float))
        self.b = nn.Parameter(torch.tensor([b], dtype=torch.float))
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

    def forward(self, x):
        cubic_root_x = torch.sign(x) * torch.abs(x).pow(3)
        raw_weights = torch.exp((self.a * torch.abs(x - cubic_root_x) + self.b * torch.abs(x + cubic_root_x)))
        weights = self.scale * ((raw_weights - raw_weights.min()) / (raw_weights.max() - raw_weights.min()))
        return weights * torch.tanh(x)


class SentimentAttentionLayer(nn.Module):
    def __init__(self, hidden_size, a, b):
        super().__init__()
        self.sentiment_transform = nn.Linear(1, hidden_size)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.custom_tanh = CustomSensitiveTanh(a, b)

    def forward(self, hidden_states, sentiment_scores):
        transformed_sentiments = self.sentiment_transform(sentiment_scores.unsqueeze(-1))
        activated_sentiments = self.custom_tanh(transformed_sentiments)
        adjusted_states = hidden_states * self.weight * activated_sentiments
        return adjusted_states


class EchoSentimentBertModel(nn.Module):
    def __init__(self, model_path, hidden_size, a, b):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(model_path)
        self.sentiment_att_layer = SentimentAttentionLayer(hidden_size, a, b)

    def forward(self, input_ids, attention_mask, sentiment_scores, labels=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        adjusted_output = self.sentiment_att_layer(sequence_output, sentiment_scores)

        # Check
        # print(f"Sequence output (first 5 tokens): {sequence_output[:, :5, :]}")
        # print(f"Adjusted output (first 5 tokens): {adjusted_output[:, :5, :]}")

        logits = self.bert.cls(adjusted_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.bert.config.vocab_size)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits[active_loss], active_labels[active_loss])
        return loss, logits



class SentimentDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = [clean_text(text) for text in texts]
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        prompt = "positive or negative? What is the sentiment of this review? the answer is [MASK]." + text
        encoding = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        tokens = self.tokenizer.tokenize(text)
        words = text.split()
        sentiment_scores = []
        word_idx = 0

        for token in tokens:
            if not token.startswith("##"):
                word = words[word_idx]
                word_idx += 1
            else:
                word = words[word_idx - 1]
            sentiment_scores.append(get_sentiment_score(word))

        if len(sentiment_scores) < self.max_len:
            sentiment_scores += [0] * (self.max_len - len(sentiment_scores))
        sentiment_scores = torch.tensor(sentiment_scores[:self.max_len], dtype=torch.float32)

    
        # if item < 5: 
        #     print(f"Tokens: {tokens}")
        #     print(f"Sentiment scores: {sentiment_scores}")

        labels = torch.full(encoding['input_ids'].shape, -100)
        mask_token_index = (encoding['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        labels[0, mask_token_index] = self.tokenizer.convert_tokens_to_ids(self.targets[item].lower().strip())

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_scores': sentiment_scores,
            'labels': labels.flatten()
        }

def bayesian_objective(a, b):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="Some weights of the model checkpoint at .* were not used when initializing.*")

    #Loading the model
    MODEL_PATH = './origin_model/bert-base-uncased-tradition'
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    learning_rate = 2e-5

    train_dataset_path = './Dataset/train/social.csv'
    df = pd.read_csv(train_dataset_path, encoding='ISO-8859-1')
    texts = df['sentence'].tolist()
    targets = df['sentiment'].tolist()
    train_texts, val_texts, train_targets, val_targets = train_test_split(texts, targets, test_size=0.1, random_state=42)

    # Data
    train_dataset = SentimentDataset(train_texts, train_targets, tokenizer, max_len=256)
    val_dataset = SentimentDataset(val_texts, val_targets, tokenizer, max_len=256)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # initialization
    model = EchoSentimentBertModel(MODEL_PATH, hidden_size=768, a=a, b=b)
    model.to(device)

    def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
        model.train()
        total_loss = 0.0
        total_correct = 0.0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sentiment_scores = batch['sentiment_scores'].to(device) 

            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, sentiment_scores, labels)

            if loss is not None:
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                _, preds = torch.max(logits, dim=-1)
                mask = labels != -100
                correct_predictions = (preds == labels) & mask
                total_correct += correct_predictions.sum().item()

        avg_loss = total_loss / len(data_loader)
        avg_correct = total_correct / n_examples

        return avg_correct, avg_loss

    def eval_model(model, data_loader, device, n_examples):
        model.eval()
        total_loss = 0.0
        total_correct = 0.0
        all_preds = []
        all_labels = []

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sentiment_scores = batch['sentiment_scores'].to(device)

            with torch.no_grad():
                loss, logits = model(input_ids, attention_mask, sentiment_scores, labels)

                if loss is not None:
                    total_loss += loss.item()

                _, preds = torch.max(logits, dim=-1)
                mask = labels != -100
                correct_predictions = (preds == labels) & mask
                total_correct += correct_predictions.sum().item()

                active_preds = preds[mask]
                active_labels = labels[mask]

                all_preds.extend(active_preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        avg_correct = total_correct / n_examples
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)

        return avg_correct, avg_loss, precision, recall, f1


    # Setting up the optimizer and training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.2,
                                                num_training_steps=total_steps)

    for epoch in range(EPOCHS):  
        train_epoch(model, train_loader, optimizer, device, scheduler, len(train_dataset))

        val_acc, loss, pre, recall, f1_s = eval_model(model, val_loader, device, len(val_dataset))
    # val_acc, val_loss, val_precision, val_recall, val_f1 = eval_model(model, val_loader, device, len(val_dataset))
    return val_acc  


# Configuring the Bayesian Optimizer
optimizer = BayesianOptimization(
    f=bayesian_objective,
    pbounds={'a': (-1, 1), 'b': (-1, 1)},
    random_state=42,
)

optimizer.maximize(init_points=10, n_iter=40)

#
best_params = optimizer.max['params']
print(f"Best parameters: {best_params}")

MODEL_PATH = './origin_model/bert-base-uncased-tradition'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = EchoSentimentBertModel(MODEL_PATH, hidden_size=768, a=best_params['a'], b=best_params['b'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

MODEL_SAVE_PATH = './Echo_model/best_model.bin'
MODEL_UNCARELOSS_ACC_SAVE_PATH = './Echo_model/best_acc_model.bin'
train_dataset_path = './Dataset/train/social.csv'
test_dataset_path = './Dataset/test/social_test.csv'
test_dataset_path2 = './Dataset/test/other2_BFRU.csv'
EPOCHS = 10
learning_rate = 2e-5


# 
df = pd.read_csv(train_dataset_path, encoding='ISO-8859-1')  
texts = df['sentence'].tolist()
targets = df['sentiment'].tolist()

train_texts, val_texts, train_targets, val_targets = train_test_split(texts, targets, test_size=0.1, random_state=42)

train_dataset = SentimentDataset(train_texts, train_targets, tokenizer, max_len=256)
val_dataset = SentimentDataset(val_texts, val_targets, tokenizer, max_len=256)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print("Let’s look at a few random input samples with prompts:")
for i, batch in enumerate(train_loader):
    if i < 3:  
        print(f"Batch {i + 1}:")
        input_ids_example = batch['input_ids'][0] 
        decoded_example = tokenizer.decode(input_ids_example, skip_special_tokens=False)
        print(decoded_example)
    else:
        break



def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model.train()
    total_loss = 0.0
    total_correct = 0.0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        sentiment_scores = batch['sentiment_scores'].to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, sentiment_scores, labels)

        if loss is not None:
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=-1)
            mask = labels != -100
            correct_predictions = (preds == labels) & mask
            total_correct += correct_predictions.sum().item()

    avg_loss = total_loss / len(data_loader)
    avg_correct = total_correct / n_examples

    return avg_correct, avg_loss


def eval_model(model, data_loader, device, n_examples):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    all_preds = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        sentiment_scores = batch['sentiment_scores'].to(device)

        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask, sentiment_scores, labels)

            if loss is not None:
                total_loss += loss.item()

            _, preds = torch.max(logits, dim=-1)
            mask = labels != -100
            correct_predictions = (preds == labels) & mask
            total_correct += correct_predictions.sum().item()

            active_preds = preds[mask]
            active_labels = labels[mask]

            all_preds.extend(active_preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    avg_correct = total_correct / n_examples
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)

    return avg_correct, avg_loss, precision, recall, f1


class EarlyStopping:
    

    def __init__(self, patience=7, delta=0.0):
    
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.2,
                                            num_training_steps=total_steps)

loss_fn = torch.nn.CrossEntropyLoss().to(device)


early_stopping = EarlyStopping(patience=5, delta=0.001)


best_val_acc = 0.0
best_val_uncareloss_acc = 0.0
best_val_loss = float('inf')
best_val_f1 = 0.0
scale_values = []

for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, len(train_dataset))
    val_acc, val_loss, val_precision, val_recall, val_f1 = eval_model(model, val_loader, device, len(val_dataset))
    print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}, Train Acc: {train_acc}')
    print(f'Val Loss: {val_loss}, Val Acc: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}')
    current_scale = model.sentiment_att_layer.custom_tanh.scale.item()
    scale_values.append(current_scale)
    print('***************************************')
    print('***************************************')
    print(f'Epoch {epoch + 1}/{EPOCHS},Params: a: {model.sentiment_att_layer.custom_tanh.a.item()}, '
          f'b: {model.sentiment_att_layer.custom_tanh.b.item()}, '
          f'scale: {current_scale}')
    print('***************************************')
    print('***************************************')


    if val_acc > best_val_uncareloss_acc: 
        best_val_uncareloss_acc = val_acc
        torch.save(model.state_dict(), MODEL_UNCARELOSS_ACC_SAVE_PATH)
        print(f'Model saved with Val unloss Acc: {val_acc}')


        
        best_params = {
            'a': model.sentiment_att_layer.custom_tanh.a.item(),
            'b': model.sentiment_att_layer.custom_tanh.b.item(),
            'scale': current_scale
        }
        torch.save(best_params, MODEL_UNCARELOSS_ACC_SAVE_PATH.replace('.bin', '_params.pth'))


    if val_acc > best_val_acc and val_loss <= best_val_loss:
        best_val_acc = val_acc
        best_val_loss = val_loss  
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_params = {
            'a': model.sentiment_att_layer.custom_tanh.a.item(),
            'b': model.sentiment_att_layer.custom_tanh.b.item(),
            'scale': current_scale
        }
        torch.save(best_params, MODEL_SAVE_PATH.replace('.bin', '_params.pth'))
        print(f'Model saved with Val Acc: {val_acc}, best_loss: {best_val_loss}')
        
    early_stopping(val_acc)
    if early_stopping.early_stop:
        print("-----Early stopping-----")
        break

# print_predictions_with_all_outputs_and_labels(model_ACP, val_loader, device, tokenizer, k=5,
#                                               output_file='./model_val_with_labels.txt')
model.load_state_dict(torch.load(MODEL_SAVE_PATH))


test_df = pd.read_csv(test_dataset_path, encoding='ISO-8859-1')
test_texts = test_df['sentence'].tolist()
test_targets = test_df['sentiment'].tolist()


test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=256)
test_loader = DataLoader(test_dataset, batch_size=16)


test_acc_list = []
test_loss_list = []
test_precision_list = []
test_recall_list = []
test_f1_list = []


test_acc, test_loss, test_precision, test_recall, test_f1 = eval_model(model, test_loader, device, len(test_dataset))
test_acc_list.append(test_acc)
test_loss_list.append(test_loss)
test_precision_list.append(test_precision)
test_recall_list.append(test_recall)
test_f1_list.append(test_f1)
print(f'Test: Loss = {test_loss}, Acc = {test_acc}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')

# print_predictions_with_all_outputs_and_labels(model_ACP, val_loader, device, tokenizer, k=5, output_file='./test_with_labels.txt')

model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
test_acc, test_loss, test_precision, test_recall, test_f1 = eval_model(model, test_loader, device, len(test_dataset))
test_acc_list.append(test_acc)
test_loss_list.append(test_loss)
test_precision_list.append(test_precision)
test_recall_list.append(test_recall)
test_f1_list.append(test_f1)
print(f'Test UNCARELOSS: Loss = {test_loss}, Acc = {test_acc}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')




test_df = pd.read_csv(test_dataset_path2, encoding='ISO-8859-1')
test_texts = test_df['sentence'].tolist()
test_targets = test_df['sentiment'].tolist()


test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=256)
test_loader = DataLoader(test_dataset, batch_size=16)


test_acc_list = []
test_loss_list = []
test_precision_list = []
test_recall_list = []
test_f1_list = []


test_acc, test_loss, test_precision, test_recall, test_f1 = eval_model(model, test_loader, device, len(test_dataset))
test_acc_list.append(test_acc)
test_loss_list.append(test_loss)
test_precision_list.append(test_precision)
test_recall_list.append(test_recall)
test_f1_list.append(test_f1)
print(f'Test: Loss = {test_loss}, Acc = {test_acc}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')

# print_predictions_with_all_outputs_and_labels(model_ACP, val_loader, device, tokenizer, k=5, output_file='./test_with_labels.txt')

model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
test_acc, test_loss, test_precision, test_recall, test_f1 = eval_model(model, test_loader, device, len(test_dataset))
test_acc_list.append(test_acc)
test_loss_list.append(test_loss)
test_precision_list.append(test_precision)
test_recall_list.append(test_recall)
test_f1_list.append(test_f1)
print(f'Test UNCARELOSS: Loss = {test_loss}, Acc = {test_acc}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}')
