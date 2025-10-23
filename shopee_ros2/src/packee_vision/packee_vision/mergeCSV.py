import pandas as pd

eclipse_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/eclipse/datasets.csv')
wasabi_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/wasabi/datasets.csv')
fish_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/fish/datasets.csv')

merged_df = pd.concat([eclipse_df, wasabi_df, fish_df], ignore_index=True)

merged_df.to_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/labels.csv', index=False)