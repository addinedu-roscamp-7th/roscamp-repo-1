import pandas as pd

eclipse_standby_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/eclipse/stanby.csv')
eclipse_grid1_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/eclipse/grid1.csv')
eclipse_grid2_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/eclipse/grid2.csv')
eclipse_grid3_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/eclipse/grid3.csv')
wasabi_standby_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/wasabi/stanby.csv')
wasabi_grid1_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/wasabi/grid1.csv')
wasabi_grid2_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/wasabi/grid2.csv')
wasabi_grid3_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/wasabi/grid3.csv')
fish_standby_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/fish/stanby.csv')
fish_grid1_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/fish/grid1.csv')
fish_grid2_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/fish/grid2.csv')
fish_grid3_df = pd.read_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/fish/grid3.csv')

merged_df = pd.concat([eclipse_standby_df, eclipse_grid1_df, eclipse_grid2_df, eclipse_grid3_df, wasabi_standby_df, wasabi_grid1_df, wasabi_grid2_df, wasabi_grid3_df, fish_standby_df, fish_grid1_df, fish_grid2_df, fish_grid3_df], ignore_index=True)

merged_df.to_csv('/home/addinedu/dev_ws/roscamp-repo-1/shopee_ros2/datasets/labels.csv', index=False)