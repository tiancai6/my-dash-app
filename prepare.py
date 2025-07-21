from doctest import debug
import pandas as pd
import matplotlib.pyplot as plt
import pytz

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def prepare():
    # 读取 Excel（或 CSV）
    df = pd.read_excel(r"库存信息-更新至0701.xlsx", sheet_name="库存信息")
    # 数据清洗
    # 移除无关列（包括"日期"列）

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
    
    # ===== 新增：价格有效性验证函数 =====
    def validate_price_by_grade(df):
        """
        根据grade_name等级验证价格有效性
zhebu'f        在相同['model','battery', 'storage', 'color', 'sim_type', 'local','date']条件下，
        分别对每天的第一次和最后一次数据进行验证：
        - 如果等级1不是4个等级中价格最小的，则删掉
        - 如果等级2的价格不比等级3小，则删掉
        - 如果等级3的价格不比等级4小，则删掉
        """
        # 记录删除前的数据条数
        original_count = len(df)
        print(f"价格验证前数据条数: {original_count}")
        
        # 提取等级数字（假设grade_name包含1-4的数字）
        def extract_grade_number(grade_name):
            if pd.isna(grade_name):
                return None
            grade_str = str(grade_name)
            for i in range(1, 5):  # 1-4等级
                if str(i) in grade_str:
                    return i
            return None
        
        df['grade_number'] = df['grade_name'].apply(extract_grade_number)
        
        # 移除无法提取等级的记录
        df_before_grade_filter = df.copy()
        df = df.dropna(subset=['grade_number'])
        df['grade_number'] = df['grade_number'].astype(int)
        
        grade_filter_removed = len(df_before_grade_filter) - len(df)
        if grade_filter_removed > 0:
            print(f"移除无法提取等级的记录: {grade_filter_removed} 条")
        
        # 重新创建unique_id来识别每个唯一组合
        unique_fields = ['model', 'grade_name', 'battery', 'storage', 'color', 'sim_type', 'local']
        df['unique_id'] = df[unique_fields].astype(str).agg('_'.join, axis=1)
        
        # 按unique_id和日期分组，然后按时间排序来识别第一次和最后一次
        df_sorted = df.sort_values(['unique_id', 'date', 'date_add_to_bag'])
        
        # 为每个unique_id和date组合标记第一次和最后一次
        df_sorted['record_order'] = df_sorted.groupby(['unique_id', 'date']).cumcount() + 1
        df_sorted['total_records'] = df_sorted.groupby(['unique_id', 'date'])['unique_id'].transform('count')
        
        # 标记是否为第一次或最后一次
        df_sorted['is_first'] = df_sorted['record_order'] == 1
        df_sorted['is_last'] = df_sorted['record_order'] == df_sorted['total_records']
        
        valid_records = []
        removed_count = 0
        
        # 分别处理第一次和最后一次的数据
        for time_type in ['first', 'last']:
            if time_type == 'first':
                current_df = df_sorted[df_sorted['is_first']].copy()
                print(f"\n处理第一次数据，共 {len(current_df)} 条")
            else:
                current_df = df_sorted[df_sorted['is_last']].copy()
                print(f"\n处理最后一次数据，共 {len(current_df)} 条")
            
            if len(current_df) == 0:
                continue
                
            # 按条件分组（不包括grade_name，因为我们要在组内比较不同等级）
            group_columns = ['model', 'battery', 'storage', 'color', 'sim_type', 'local', 'date']
            grouped = current_df.groupby(group_columns)
            
            time_removed_count = 0
            
            for group_key, group_data in grouped:
                if len(group_data) <= 1:
                    # 只有一条记录，直接保留
                    valid_records.extend(group_data.to_dict('records'))
                    continue
                
                # 按等级排序
                group_sorted = group_data.sort_values('grade_number')
                
                # 获取每个等级的价格
                grade_prices = {}
                for _, row in group_sorted.iterrows():
                    grade = row['grade_number']
                    price = row['price']
                    if grade not in grade_prices:
                        grade_prices[grade] = []
                    grade_prices[grade].append((row.name, price))
                
                # 验证价格逻辑并标记要删除的记录
                indices_to_remove = set()
                
                # 规则1：如果等级1不是所有等级中价格最小的，则删除等级1
                if 1 in grade_prices:
                    grade1_prices = [price for _, price in grade_prices[1]]
                    all_other_prices = []
                    for g in [2, 3, 4]:
                        if g in grade_prices:
                            all_other_prices.extend([price for _, price in grade_prices[g]])
                    
                    if all_other_prices:
                        min_grade1 = min(grade1_prices)
                        min_others = min(all_other_prices)
                        if min_grade1 >= min_others:
                            # 删除所有等级1的记录
                            for idx, _ in grade_prices[1]:
                                indices_to_remove.add(idx)
                                time_removed_count += 1
                
                # 规则2：如果等级2的价格不比等级3小，则删除等级2
                if 2 in grade_prices and 3 in grade_prices:
                    grade2_prices = [price for _, price in grade_prices[2]]
                    grade3_prices = [price for _, price in grade_prices[3]]
                    
                    min_grade2 = min(grade2_prices)
                    min_grade3 = min(grade3_prices)
                    
                    if min_grade2 >= min_grade3:
                        # 删除所有等级2的记录
                        for idx, _ in grade_prices[2]:
                            indices_to_remove.add(idx)
                            time_removed_count += 1
                
                # 规则3：如果等级3的价格不比等级4小，则删除等级3
                if 3 in grade_prices and 4 in grade_prices:
                    grade3_prices = [price for _, price in grade_prices[3]]
                    grade4_prices = [price for _, price in grade_prices[4]]
                    
                    min_grade3 = min(grade3_prices)
                    min_grade4 = min(grade4_prices)
                    
                    if min_grade3 >= min_grade4:
                        # 删除所有等级3的记录
                        for idx, _ in grade_prices[3]:
                            indices_to_remove.add(idx)
                            time_removed_count += 1
                
                # 保留未被标记删除的记录
                for _, row in group_sorted.iterrows():
                    if row.name not in indices_to_remove:
                        valid_records.append(row.to_dict())
            
            print(f"{time_type}数据删除了 {time_removed_count} 条")
            removed_count += time_removed_count
        
        # 创建新的DataFrame
        result_df = pd.DataFrame(valid_records)
        
        # 删除临时列
        temp_columns = ['grade_number', 'unique_id', 'record_order', 'total_records', 'is_first', 'is_last']
        for col in temp_columns:
            if col in result_df.columns:
                result_df = result_df.drop(col, axis=1)
        
        # 记录删除后的数据条数
        final_count = len(result_df)
        total_removed = original_count - final_count
        
        print(f"\n价格验证后数据条数: {final_count}")
        print(f"总共删除数据条数: {total_removed}")
        print(f"其中因价格逻辑不符删除: {removed_count} 条")
        
        return result_df
    
    # 应用价格验证 - 注意：这个应该在process_unique_records_per_day之后调用
    # df = validate_price_by_grade(df)  # 先注释掉这行
    
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
    
    # 在处理完每日记录后再进行价格验证
    validated_df = validate_price_by_grade(processed_df)
    
    return validated_df

# if __name__ == "__main__":
#     df = prepare()
   