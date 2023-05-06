from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import pickle

def s_bert(query,docs,p):
    # Load pre-trained S-BERT model
    model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sbert_model = SentenceTransformer(model_name)
    # t=int(0.8*len(docs))
    # train_dataset=p
    # val_dataset=p
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # from torch.utils.data import DataLoader


    # # Create a DataLoader for the training dataset
    # train_dataset= DataLoader(train_dataset, batch_size=16, shuffle=True)
    # # Define the training arguments
    # training_args = TrainingArguments(
    #     output_dir='./results',
    #     evaluation_strategy='epoch',
    #     save_strategy='epoch',
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=64,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     push_to_hub=False,
    # )

    # # Define the trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     tokenizer=tokenizer,
    # )

    # # Fine-tune the model
    # trainer.train()

    # # Evaluate the model
    # trainer.evaluate()

    # # Save the fine-tuned model
    # model_path = './fine-tuned-model'
    # model.save_pretrained(model_path)
    # tokenizer.save_pretrained(model_path)
    query_embeddings = sbert_model.encode(query, convert_to_tensor=True)

    try:
        with open("sbert.txt", "rb") as fp:  # Unpickling to avoid reruning the code
            doc_embeddings = pickle.load(fp)
    except FileNotFoundError:
        doc_embeddings = sbert_model.encode(docs, convert_to_tensor=True)
        with open("sbert.txt", "wb") as fp:  # Pickling
            pickle.dump(doc_embeddings, fp)
    
    #print(query_embeddings.T)
    #print(np.shape(query_embeddings))
    #print(np.shape(doc_embeddings))
    # Calculate similarity
    similarity_scores = np.dot(query_embeddings, doc_embeddings.T)
    #json.dump(similarity_scores, open("s_bert.txt", 'w'))
    #print(similarity_scores[:2])
    # Rank documents and present top results
    top_results = np.argsort(similarity_scores)[0][::-1]
    top_results=[(i+1)%len(top_results) for i in top_results]
    ans=[]
    t=[]
    for i, result in enumerate(top_results):
        if docs[result]:
            ans.append(result)
        else:
            t.append(result)
    ans+=t
    return ans