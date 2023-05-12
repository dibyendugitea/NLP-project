from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import pickle

def s_bert(query,docs):
    # Load pre-trained S-BERT model
    model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sbert_model = SentenceTransformer(model_name)
    query=' '.join(query)
    # print(query)
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
    
    # with open("pickle/sbert_doc.txt", "rb") as fp:  # Unpickling to avoid reruning the code
    #         doc_embeddings = pickle.load(fp)

    try:
        with open("pickle/sbert_doc.txt", "rb") as fp:  # Unpickling to avoid reruning the code
            doc_embeddings = pickle.load(fp)
    except FileNotFoundError:
        doc_embeddings = sbert_model.encode(docs, convert_to_tensor=True)
        with open("pickle/sbert_doc.txt", "wb") as fp:  # Pickling
            pickle.dump(doc_embeddings, fp)
    # # try:
    #     with open("/pickle/sbert_query.txt", "rb") as fp:  # Unpickling to avoid reruning the code
    #         query_embeddings = pickle.load(fp)
    # except FileNotFoundError:
    #     query_embeddings=[]
    #     for q in query:
    query_embeddings=sbert_model.encode(query, convert_to_tensor=True)
    #     with open("sbert_query.txt", "wb") as fp:  # Pickling
    #         pickle.dump(query_embeddings, fp)
    #print(query_embeddings.T)
    #print(np.shape(query_embeddings))
    #print(np.shape(doc_embeddings))
    # Calculate similarity
    ids_ordered=[]
    # for query,doc in zip(query_embeddings, doc_embeddings):
        
    # ids_ordered.append(ans)
    # return ids_ordered
    #json.dump(similarity_scores, open("s_bert.txt", 'w'))
    #print(similarity_scores[:2])
    # Rank documents and present top results
    

    similarity_scores = np.dot(query_embeddings, doc_embeddings.T)
    top_results = np.argsort(similarity_scores)[::-1]
    top_results=[(i+1)%len(top_results) for i in top_results]
    ans=[]
    t=[]
    # print(top_results)
    for i, result in enumerate(top_results):
        if docs[result]:
            ans.append(result)
        else:
            t.append(result)
    return ans