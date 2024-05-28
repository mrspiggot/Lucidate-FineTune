import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, TextClassificationPipeline

device = torch.device("mps")


class DataPreprocessor:
    def __init__(self, dataset_name, tokenizer, train_size=1000, eval_size=250, test_size=500):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name)
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_datasets()
        self.tokenize_datasets()

    def split_datasets(self):
        train_val_dataset = self.dataset['train'].train_test_split(test_size=0.2)
        train_dataset = train_val_dataset['train'].shuffle(seed=42).select(range(self.train_size))
        val_dataset = train_val_dataset['test'].shuffle(seed=42).select(range(self.eval_size))
        test_dataset = self.dataset['test'].shuffle(seed=42).select(range(self.test_size))
        return train_dataset, val_dataset, test_dataset

    def tokenize_function(self, example):
        return self.tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

    def tokenize_datasets(self):
        self.train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)
        self.val_dataset = self.val_dataset.map(self.tokenize_function, batched=True)
        self.test_dataset = self.test_dataset.map(self.tokenize_function, batched=True)


class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.metric = load_metric('accuracy')
        self.precision_metric = load_metric('precision')
        self.recall_metric = load_metric('recall')
        self.f1_metric = load_metric('f1')
        self.label_map = {0: "Negative", 1: "Positive"}
        self.reverse_label_map = {"NEGATIVE": 0, "POSITIVE": 1}

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = self.metric.compute(predictions=predictions, references=labels)['accuracy']
        precision = self.precision_metric.compute(predictions=predictions, references=labels, average='binary')['precision']
        recall = self.recall_metric.compute(predictions=predictions, references=labels, average='binary')['recall']
        f1 = self.f1_metric.compute(predictions=predictions, references=labels, average='binary')['f1']
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def evaluate(self, dataset, sample_size=None):
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, truncation=True, padding=True)
        if sample_size:
            sample_indices = np.random.choice(len(dataset), size=sample_size, replace=False).tolist()
            sample_texts = [dataset[i]['text'] for i in sample_indices]
            sample_labels = [dataset[i]['label'] for i in sample_indices]
        else:
            sample_texts = dataset['text']
            sample_labels = dataset['label']

        predictions = [self.reverse_label_map[pipeline(text)[0]['label']] for text in sample_texts]
        accuracy = self.metric.compute(predictions=predictions, references=sample_labels)['accuracy']
        precision = self.precision_metric.compute(predictions=predictions, references=sample_labels, average='binary')['precision']
        recall = self.recall_metric.compute(predictions=predictions, references=sample_labels, average='binary')['recall']
        f1 = self.f1_metric.compute(predictions=predictions, references=sample_labels, average='binary')['f1']

        if sample_size:
            print("\nSample Reviews and Actual Sentiments:")
            for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
                print(f"Review {i + 1}:")
                print(text)
                print(f"Actual Sentiment: {self.label_map[label]}")
                print()
                print(f"Prediction: {self.label_map[predictions[i]]} (Confidence: {pipeline(text)[0]['score']:.4f})")

        return accuracy, precision, recall, f1


class ModelTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=ModelEvaluator(self.model, self.tokenizer).compute_metrics
        )

        print("Fine-tuning the model...")
        trainer.train()



class SentimentAnalysis:
    def __init__(self, model_name, dataset_name, train_size=1000, eval_size=250, test_size=500, sample_size=10):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.device("mps"))
        self.sample_size = sample_size
        self.data_preprocessor = DataPreprocessor(dataset_name, self.tokenizer, train_size, eval_size, test_size)
        self.model_evaluator = ModelEvaluator(self.model, self.tokenizer)
        self.model_trainer = ModelTrainer(self.model, self.tokenizer, self.data_preprocessor.train_dataset, self.data_preprocessor.val_dataset)

    def evaluate_pretrained_model(self):
        accuracy, precision, recall, f1 = self.model_evaluator.evaluate(self.data_preprocessor.test_dataset, sample_size=self.sample_size)
        print(f"\nPre-trained model results on sample: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

        accuracy, precision, recall, f1 = self.model_evaluator.evaluate(self.data_preprocessor.test_dataset)
        print(f"\nPre-trained model results on the entire test set: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

    def fine_tune_model(self):
        self.model_trainer.train()

    def evaluate_fine_tuned_model(self):
        accuracy, precision, recall, f1 = self.model_evaluator.evaluate(self.data_preprocessor.test_dataset)
        print(f"\nFine-tuned model results on the entire test set: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")


if __name__ == "__main__":
    sentiment_analysis = SentimentAnalysis(
        model_name='distilbert-base-uncased-finetuned-sst-2-english',
        dataset_name='imdb',
        train_size=5000,
        eval_size=1000,
        test_size=1000,
        sample_size=10
    )
    sentiment_analysis.evaluate_pretrained_model()
    sentiment_analysis.fine_tune_model()
    sentiment_analysis.evaluate_fine_tuned_model()
