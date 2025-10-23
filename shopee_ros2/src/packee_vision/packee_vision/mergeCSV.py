import pandas as pd

buldak_df = pd.read_csv('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/buldak_can/datasets.csv')
candy_df = pd.read_csv('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/eclipse/datasets.csv')
meat_df = pd.read_csv('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/pork/datasets.csv') 
butter_df = pd.read_csv('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/buldak_can/datasets.csv')
wasabi_df = pd.read_csv('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/wasabi/datasets.csv')
fish_df = pd.read_csv('/home/addinedu/dev_ws/shopee/src/DataCollector/DataCollector/datasets/fish/datasets.csv')

merged_df = pd.concat([buldak_df, candy_df, meat_df, butter_df, wasabi_df, fish_df], ignore_index=True)

merged_df.to_csv('/home/addinedu/dev_ws/shopee/datasets/labels.csv', index=False)