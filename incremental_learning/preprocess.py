import os
import pandas as pd
import tensorflow_hub as hub

"""
    load processed data into database for later offline and online training
"""
def process_text(name_stock_idx, model):
    """ implement processed file in form of CSV instead of loading into database"""
    
    df = pd.DataFrame(columns=['Date', 'ID', 'feature'])
    for stock_idx in name_stock_idx:
        path = 'stocknet-dataset/tweet/raw/' + stock_idx + '/'
        days = os.listdir(path)
        row_list = []
        for day in days:
                data = pd.read_json(path + day,lines=True)
                emb = model(data['text']).numpy()
                new_row = {'Date': data['created_at'][0].date(),
                        'ID': stock_idx,
                        'feature' : [emb]}
                row_list.append(new_row)
        appended_df = pd.DataFrame(row_list,columns=['Date', 'ID', 'feature'])
        df = pd.concat([df,appended_df], ignore_index= True)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
    out_path = 'processed_data/tweet.csv'
    df.to_csv(out_path)
    
def process_price(name_stock_idx):
    path = 'stocknet-dataset/price/raw/'
    df = pd.DataFrame(columns=['Date', 'ID', 'feature'])
    for i in name_stock_idx:
        price_path = path + i + '.csv'
        price = pd.read_csv (price_path, delimiter = ",")
        price['ID'] = i
        feature = price[['Adj Close', 'High', 'Low']].to_numpy().tolist()
        price['feature'] = feature
        df = pd.concat([df,price[['Date', 'ID', 'feature']]], ignore_index=True)
    
    out_path = 'processed_data/price.csv'
    df.to_csv(out_path)

if __name__ == 'main':
    
    module_url = "https://tfhub.dev/google/universal-sentence-encoder"
    model = hub.load(module_url)
    stock_idx = pd.read_csv('stock_idx.csv')
    process_text(stock_idx, model)
    process_price(stock_idx)