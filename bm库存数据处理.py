import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def process_json_files_in_subfolders(base_folder, output_path):
    combined_df = pd.DataFrame()
    if os.path.exists(output_path):
        combined_df = pd.read_csv(output_path, nrows=0, encoding='utf-8')

    # 遍历所有07开头的文件夹
    for subfolder in os.listdir(base_folder):
        if subfolder.startswith("07"):
            subfolder_path = os.path.join(base_folder, subfolder)
            if os.path.isdir(subfolder_path):
                # 遍历子文件夹下所有包含"cart"的文件
                for filename in os.listdir(subfolder_path):
                    if "cart" in filename:
                        file_path = os.path.join(subfolder_path, filename)
                        try:
                            # 读取并处理JSON文件
                            with open(file_path, 'r', encoding='utf-8') as f:
                                js = json.load(f)

                            datas = js['cart_items']
                            records = []
                            for data in datas:
                                records.append({
                                    'sku': data['sku'],
                                    'title': data['title'],
                                    'model': data['model'],
                                    'grade_name': data['grade_name'],
                                    'battery(1新,0旧)': data['gradeExtended']['hasNewBattery'],
                                    'storage': data['subtitleElements'][0],
                                    'color': data['color'],
                                    'quantity_max': data['quantity_max'],
                                    'local': data['link']['params']['locale'],
                                    'price': data['price'],
                                    'price_with_currency': data['price_with_currency'],
                                    'href': data['link']['href'],
                                    'available': data['available'],
                                    '商家': data['merchant'],
                                    'merchant_id': data['merchant_id'],
                                    'merchant_public_id': data['merchant_public_id'],
                                    'category': data['category'],
                                    'country_tax': data['country_tax'],
                                    'product_id': data['product_id'],
                                    'akeneoId': data['akeneoId'],
                                    'date_add_to_bag': data['date_add_to_bag']
                                })
                            df = pd.DataFrame(records)
                            combined_df = pd.concat([combined_df, df], ignore_index=True)

                        except Exception as e:
                            print(f"Error processing file {filename}: {str(e)}")

    # 导出合并后的数据
    header = not os.path.exists(output_path)
    mode = 'a' if os.path.exists(output_path) else 'w'
    combined_df.to_csv(
        output_path,
        index=False,
        encoding='utf_8_sig',
        mode=mode,
        header=header
    )
    print(f"Processed {len(combined_df)} records into {output_path}")


if __name__ == "__main__":
    # print(datetime.today().date())
    # 配置路径参数
    base_folder = 'C:/Users/HP/Desktop/backmarket'
    output_path = os.path.join(base_folder, '库存信息.csv')
    process_json_files_in_subfolders(base_folder, output_path)