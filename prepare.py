import pandas as pd
import matplotlib.pyplot as plt
import pytz

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def prepare():
    # 读取 Excel（或 CSV）
    df = pd.read_excel(r"库存信息-更新至0701.xlsx", sheet_name="库存信息")
    # 数据清洗
    # 移除无关列（包括“日期”列）

    columns_to_keep = ['型号', '磨损中文','battery(1新,0旧)', 'storage', '颜色中文', 'sim类型', 'quantity_max', 'local','price', 'date_add_to_bag','商家','merchant_public_id']
    df = df[columns_to_keep].copy()

    # 重命名列为英文，方便处理
    df.columns = ['model', 'grade_name','battery', 'storage', 'color', 'sim_type', 'quantity', 'local', 'price', 'date_add_to_bag','seller','merchant_public_id']

    # 数据类型转换
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['date_add_to_bag'] = pd.to_datetime(df['date_add_to_bag'], errors='coerce')

    # 移除无效行（缺失关键字段）
    df = df.dropna(subset=['model', 'storage', 'sim_type', 'date_add_to_bag'])

    # 将法国时间（假设为 UTC+2）转换为中国时间（UTC+8）
    def convert_to_china_time(dt):
        if pd.isna(dt):
            return dt
        france_tz = pytz.timezone('Europe/Paris')
        china_tz = pytz.timezone('Asia/Shanghai')
        dt_france = france_tz.localize(dt)
        dt_china = dt_france.astimezone(china_tz)
        return dt_china

    df['date_add_to_bag'] = df['date_add_to_bag'].apply(convert_to_china_time)
    df['date'] = df['date_add_to_bag'].dt.date

    def process_unique_records_per_day(df):
        """
        根据指定字段组合创建唯一值，并按日期整理数据
        每个唯一值在每天最多出现两次（第一次和最后一次）
        如果只出现一次，则复制该记录
        保留所有原始列的数据
        """
        # 定义用于创建唯一值的字段
        unique_fields = ['model', 'grade_name', 'battery', 'storage', 'color', 'sim_type', 'local']
        
        # 创建唯一标识符
        df['unique_id'] = df[unique_fields].astype(str).agg('_'.join, axis=1)
        
        # 按唯一标识符和日期分组
        grouped = df.groupby(['unique_id', 'date'])
        
        processed_records = []
        
        for (unique_id, date), group in grouped:
            # 按时间排序
            group_sorted = group.sort_values('date_add_to_bag')
            
            if len(group_sorted) == 1:
                # 如果只有一条记录，复制两次
                record = group_sorted.iloc[0].copy()
                processed_records.append(record)  # 第一次
                processed_records.append(record)  # 第二次（复制）
            else:
                # 如果有多条记录，取第一条和最后一条
                first_record = group_sorted.iloc[0]
                last_record = group_sorted.iloc[-1]
                processed_records.append(first_record)
                processed_records.append(last_record)
        
        # 创建新的DataFrame，保留所有原始列
        result_df = pd.DataFrame(processed_records)
        result_df['storage'] = result_df['storage'].str.replace('GB', '').str.replace('Go', '').astype(str)
        
        # 重置索引
        result_df = result_df.reset_index(drop=True)
        
        return result_df

    # 处理数据
    processed_df = process_unique_records_per_day(df)
    return processed_df