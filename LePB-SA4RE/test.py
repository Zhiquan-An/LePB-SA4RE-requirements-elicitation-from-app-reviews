import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import torch.nn
import re
import csv
import torch.nn
from torch import nn

MODEL_PATH = './origin_model/bert-base-uncased-tradition'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = './best_model.bin'
MODEL_UNCARELOSS_ACC_SAVE_PATH = './best_acc_model.bin'
MODEL_PATH = './origin_model/bert-base-uncased-tradition'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = EchoSentimentBertModel(MODEL_PATH, hidden_size=768, a=-0.869896814029441, b=0.8977710745066665)

test_dataset_path5 = './Dataset/test_robustness/test.csv'
test_dataset_path1 = './Dataset/test/total_test.csv'
test_dataset_path3 = './Dataset/test/product_test.csv'
test_dataset_path4 = './Dataset/test/game_test.csv'
test_dataset_path2 = './Dataset/test/other2_BFRU.csv'
test_dataset_path6 = './Dataset/test/other1_opinion.csv'


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

def eval_model(model, data_loader, device, n_examples):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sentiment_scores = batch['sentiment_scores'].to(device)


            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, sentiment_scores=sentiment_scores, labels=labels)


            total_loss += loss.item()

            _, preds = torch.max(logits, dim=-1)
            mask = labels != -100
            correct_predictions = (preds == labels) & mask
            total_correct += correct_predictions.sum().item()


            valid_labels = labels[mask]
            valid_preds = preds[mask]
            all_labels.extend(valid_labels.cpu().numpy())
            all_preds.extend(valid_preds.cpu().numpy())

    # 计算性能指标
    avg_loss = total_loss / len(data_loader)
    avg_correct = total_correct / n_examples
    all_labels_true = []
    all_preds_true = []
    for i, j in zip(all_labels, all_preds):
        if i == 3893:
            all_labels_true.append(1)
        else:
            all_labels_true.append(0)
        if j == 3893:
            all_preds_true.append(1)
        else:
            all_preds_true.append(0)

    precision = precision_score(all_labels_true, all_preds_true)
    recall = recall_score(all_labels_true, all_preds_true)
    f1 = f1_score(all_labels_true, all_preds_true)

    return avg_correct, avg_loss, f1, precision, recall



test_df = pd.read_csv(test_dataset_path1, encoding='ISO-8859-1')
test_texts = test_df['sentence'].tolist()
test_targets = test_df['sentiment'].tolist()


test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=256)
test_loader = DataLoader(test_dataset, batch_size=16)


model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device)
test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

print(f'Test: Loss = {test_loss}, Acc = {test_acc}, F1 = {test_f1}, Pre = {test_precision}, Recall = {test_recall}')



model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
model.to(device)

test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

print(f'Test: UNCARELOSS_Loss = {test_loss}, UNCARELOSS_Acc = {test_acc}, UNCARELOSS_F1 = {test_f1}, UNCARELOSS_Pre = {test_precision}, UNCARELOSS_Recall = {test_recall}')




print('=========2222222=========================================')


test_df = pd.read_csv(test_dataset_path2, encoding='ISO-8859-1')
test_texts = test_df['sentence'].tolist()
test_targets = test_df['sentiment'].tolist()


test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=256)
test_loader = DataLoader(test_dataset, batch_size=16)


model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device)
test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

print(f'Test: Loss = {test_loss}, Acc = {test_acc}, F1 = {test_f1}, Pre = {test_precision}, Recall = {test_recall}')



model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
model.to(device)

test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

print(f'Test: UNCARELOSS_Loss = {test_loss}, UNCARELOSS_Acc = {test_acc}, UNCARELOSS_F1 = {test_f1}, UNCARELOSS_Pre = {test_precision}, UNCARELOSS_Recall = {test_recall}')


# print('===========3333333=======================================')
#

# test_df = pd.read_csv(test_dataset_path3, encoding='ISO-8859-1')
# test_texts = test_df['sentence'].tolist()
# test_targets = test_df['sentiment'].tolist()
#

# test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=128)
# test_loader = DataLoader(test_dataset, batch_size=32)
#

# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# model.to(device)
# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: Loss = {test_loss}, Acc = {test_acc}, F1 = {test_f1}, Pre = {test_precision}, Recall = {test_recall}')
#
#

# model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
# model.to(device)

# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: UNCARELOSS_Loss = {test_loss}, UNCARELOSS_Acc = {test_acc}, UNCARELOSS_F1 = {test_f1}, UNCARELOSS_Pre = {test_precision}, UNCARELOSS_Recall = {test_recall}')
#
# print('=============44444444=====================================')
#

# test_df = pd.read_csv(test_dataset_path4, encoding='ISO-8859-1')
# test_texts = test_df['sentence'].tolist()
# test_targets = test_df['sentiment'].tolist()
#

# test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=128)
# test_loader = DataLoader(test_dataset, batch_size=32)

# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# model.to(device)
# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: Loss = {test_loss}, Acc = {test_acc}, F1 = {test_f1}, Pre = {test_precision}, Recall = {test_recall}')
#
#

# model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
# model.to(device)

# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: UNCARELOSS_Loss = {test_loss}, UNCARELOSS_Acc = {test_acc}, UNCARELOSS_F1 = {test_f1}, UNCARELOSS_Pre = {test_precision}, UNCARELOSS_Recall = {test_recall}')
#
# print('==================================================')
# print('==================================================')
# print('==================================================')
# print('==================================================')
# print('==================================================')
# print('==================================================')
# print('==================================================')
#

# test_df = pd.read_csv(test_dataset_path5, encoding='ISO-8859-1')
# test_texts = test_df['sentence'].tolist()
# test_targets = test_df['sentiment'].tolist()
#

# test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=128)
# test_loader = DataLoader(test_dataset, batch_size=32)
#

# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# model.to(device)
# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: Loss = {test_loss}, Acc = {test_acc}, F1 = {test_f1}, Pre = {test_precision}, Recall = {test_recall}')
#
#

# model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
# model.to(device)

# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: UNCARELOSS_Loss = {test_loss}, UNCARELOSS_Acc = {test_acc}, UNCARELOSS_F1 = {test_f1}, UNCARELOSS_Pre = {test_precision}, UNCARELOSS_Recall = {test_recall}')
#
# print('==================================================')
# print('==================================================')

# test_df = pd.read_csv(test_dataset_path6, encoding='ISO-8859-1')
# test_texts = test_df['sentence'].tolist()
# test_targets = test_df['sentiment'].tolist()
#

# test_dataset = SentimentDataset(test_texts, test_targets, tokenizer, max_len=128)
# test_loader = DataLoader(test_dataset, batch_size=32)
#

# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# model.to(device)
# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: Loss = {test_loss}, Acc = {test_acc}, F1 = {test_f1}, Pre = {test_precision}, Recall = {test_recall}')
#
#

# model.load_state_dict(torch.load(MODEL_UNCARELOSS_ACC_SAVE_PATH))
# model.to(device)

# test_acc, test_loss, test_f1, test_precision, test_recall = eval_model(model, test_loader, device, len(test_dataset))

# print(f'Test: UNCARELOSS_Loss = {test_loss}, UNCARELOSS_Acc = {test_acc}, UNCARELOSS_F1 = {test_f1}, UNCARELOSS_Pre = {test_precision}, UNCARELOSS_Recall = {test_recall}')