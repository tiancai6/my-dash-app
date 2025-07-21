import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timedelta

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 在文件开头添加新的导入
import json
import os

# 新增Excel数据加载函数
def load_excel_data():
    """加载Excel数据"""
    try:
        excel_file = "e:\\项目\\app\\数据汇总后用于分析（定期更新模板）.xlsx"
        
        # 读取讯强全新机数据
        xq_data = pd.read_excel(excel_file, sheet_name="讯强全新机")
        
        # 读取靓机汇二手回收数据
        ljh_data = pd.read_excel(excel_file, sheet_name="靓机汇二手回收")
        
        return xq_data, ljh_data
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return pd.DataFrame(), pd.DataFrame()

def process_memory_column(df, memory_col='内存'):
    """处理内存列，去除G并转换为整数"""
    if memory_col in df.columns:
        df[memory_col] = df[memory_col].astype(str).str.replace('G', '').str.replace('g', '')
        # 处理纳米：将"纳米"替换为"1"
        df[memory_col] = df[memory_col].str.replace('纳米', '1')
        df[memory_col] = pd.to_numeric(df[memory_col], errors='coerce')
    return df

def prepare_data():
    """完整的数据读取和预处理函数"""
    # 步骤1: 读取 Excel（或 CSV）
    df = pd.read_excel("e:\项目\补全后的库存数据_20250717_181814.xlsx", sheet_name="库存信息")
    print(f"步骤1 - 原始数据读取完成，总行数: {len(df)}")
    
    # 步骤2: 数据清洗 - 移除无关列（包括"日期"列）
    columns_to_keep = ['型号', '磨损中文','battery(1新,0旧)', 'storage', '颜色中文', 'sim类型', 'quantity_max', 'local','price', 'date_add_to_bag','商家','merchant_public_id']
    df_before_column_filter = df.copy()
    df = df[columns_to_keep].copy()
    print(f"步骤2 - 选择指定列后，行数: {len(df)} (无变化，只是选择了列)")
    
    # 步骤3: 重命名列为英文，方便处理
    df.columns = ['model', 'grade_name','battery', 'storage', 'color', 'sim_type', 'quantity', 'local', 'price', 'date_add_to_bag','seller','merchant_public_id']
    print(f"步骤3 - 重命名列后，行数: {len(df)} (无变化，只是重命名)")
    
    # 步骤4: 数据类型转换
    df_before_type_conversion = df.copy()
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['date_add_to_bag'] = pd.to_datetime(df['date_add_to_bag'], errors='coerce')
    print(f"步骤4 - 数据类型转换后，行数: {len(df)} (无变化，只是转换类型)")
    
    # 步骤5: 移除无效行（缺失关键字段）
    df_before_dropna = df.copy()
    df_after_dropna = df.dropna(subset=['model', 'storage', 'sim_type', 'date_add_to_bag'])
    
    # 记录被删除的数据
    dropped_rows = df_before_dropna[~df_before_dropna.index.isin(df_after_dropna.index)]
    if len(dropped_rows) > 0:
        dropped_rows_copy = dropped_rows.copy()
        dropped_rows_copy['删除原因'] = '关键字段缺失(model/storage/sim_type/date_add_to_bag)'
        dropped_rows_copy['删除步骤'] = '步骤5-移除关键字段缺失'
        dropped_rows_copy.to_csv('e:\\项目\\app\\步骤5_删除的关键字段缺失数据.csv', index=False, encoding='utf-8-sig')
        print(f"步骤5 - 移除关键字段缺失行: 删除 {len(dropped_rows)} 条，剩余 {len(df_after_dropna)} 条")
        print(f"       删除的数据已保存到: 步骤5_删除的关键字段缺失数据.csv")
    else:
        print(f"步骤5 - 移除关键字段缺失行: 无数据被删除，行数: {len(df_after_dropna)}")
    
    df = df_after_dropna
    
    # 步骤6: 时区转换
    # def convert_to_china_time(dt):
    #     if pd.isna(dt):
    #         return dt
    #     france_tz = pytz.timezone('Europe/Paris')
    #     china_tz = pytz.timezone('Asia/Shanghai')
    #     dt_france = france_tz.localize(dt)
    #     dt_china = dt_france.astimezone(china_tz)
    #     return dt_china
    
    # df['date_add_to_bag'] = df['date_add_to_bag'].apply(convert_to_china_time)
    df['date'] = df['date_add_to_bag'].dt.date
    print(f"步骤6 - 保持法国时区，添加date列后，行数: {len(df)} (无变化，保持原始时区)")
    
    # 步骤7: storage单位处理 - 智能处理，保留原本就是数值的
    df_before_storage = df.copy()
    
    # 先检查storage列的数据类型分布
    print(f"步骤7 - storage处理前分析:")
    
    # 统计不同类型的storage数据
    storage_stats = {
        '包含GB的': 0,
        '包含Go的': 0, 
        '包含其他G字符的': 0,
        '包含纳米的': 0,
        '纯数字的': 0,
        '其他格式的': 0
    }
    
    for idx, storage_val in df['storage'].items():
        if pd.isna(storage_val):
            continue
        storage_str = str(storage_val)
        
        if 'GB' in storage_str:
            storage_stats['包含GB的'] += 1
        elif 'Go' in storage_str:
            storage_stats['包含Go的'] += 1
        elif '纳米' in storage_str:
            storage_stats['包含纳米的'] += 1
        elif 'G' in storage_str and not storage_str.replace('G', '').replace('.', '').isdigit():
            storage_stats['包含其他G字符的'] += 1
        else:
            # 尝试转换为数字
            try:
                float(storage_str)
                storage_stats['纯数字的'] += 1
            except ValueError:
                storage_stats['其他格式的'] += 1
    
    # 打印统计信息
    for key, count in storage_stats.items():
        if count > 0:
            print(f"       {key}: {count} 条")
    
    # 处理storage：移除GB、Go单位，纳米替换为1，最后转为整数
    def clean_storage(storage_val):
        if pd.isna(storage_val):
            return None
        storage_str = str(storage_val)
        # 移除GB和Go单位
        cleaned = storage_str.replace('GB', '').replace('Go', '').strip()
        # 纳米替换为1
        cleaned = cleaned.replace('纳米', '1')
        try:
            return int(float(cleaned))  # 先转float再转int，处理小数情况
        except (ValueError, TypeError):
            return None  # 无法转换的设为None
    
    df['storage'] = df['storage'].apply(clean_storage)
    
    # 移除storage为None的行
    before_count = len(df)
    df = df.dropna(subset=['storage'])
    after_count = len(df)
    if before_count != after_count:
        print(f"步骤7 - storage处理: 删除了 {before_count - after_count} 条无效storage数据")
    
    print(f"步骤7 - storage处理完成后，行数: {len(df)} (已转换为int类型)")
    
    # 步骤8: 按加入购物车时间排序
    df = df.sort_values('date_add_to_bag').reset_index(drop=True)
    print(f"步骤8 - 按时间排序后，行数: {len(df)} (无变化，只是排序)")
    
    print(f"\n=== 数据预处理完成 ===")
    print(f"最终数据行数: {len(df)}")
    
    return df

def validate_price_by_grade(df):
    """根据grade_name等级验证价格有效性"""
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

def predict_seller_change(df, current_seller, unique_id, current_date, days=7):
    """
    预测商家变化
    
    Args:
        df: 数据框
        current_seller: 当前商家
        unique_id: 唯一标识符
        current_date: 当前日期
        days: 历史数据天数
    
    Returns:
        predicted_seller: 预测的商家
        change_prob: 变化概率
    """
    try:
        # 获取历史数据
        end_date = pd.to_datetime(current_date)
        start_date = end_date - timedelta(days=days)
        
        # 筛选相同产品的历史数据
        historical_data = df[
            (df['unique_id'] == unique_id) & 
            (pd.to_datetime(df['date_add_to_bag']).dt.tz_localize(None) >= start_date.tz_localize(None)) & 
            (pd.to_datetime(df['date_add_to_bag']).dt.tz_localize(None) < end_date.tz_localize(None))
        ].copy()
        
        if len(historical_data) == 0:
            # 没有历史数据，假设商家不变
            return current_seller, 0.1
        
        # 分析商家变化模式
        seller_counts = historical_data['seller'].value_counts()
        
        if len(seller_counts) == 1:
            # 只有一个商家，变化概率较低
            return current_seller, 0.2
        
        # 计算最常见的商家
        most_common_seller = seller_counts.index[0]
        most_common_count = seller_counts.iloc[0]
        
        # 计算变化概率
        total_records = len(historical_data)
        stability_ratio = most_common_count / total_records
        
        if most_common_seller == current_seller:
            # 当前商家是最常见的，变化概率较低
            change_prob = 1 - stability_ratio
            predicted_seller = current_seller
        else:
            # 当前商家不是最常见的，可能会变化
            change_prob = stability_ratio
            predicted_seller = most_common_seller
        
        # 限制概率范围
        change_prob = max(0.1, min(0.9, change_prob))
        
        return predicted_seller, change_prob
        
    except Exception as e:
        print(f"预测商家变化时出错: {e}")
        return current_seller, 0.1

def get_seller_historical_data(df, seller, unique_id, current_date, days=7):
    """
    获取指定商家的历史数据
    
    Args:
        df: 数据框
        seller: 商家名称
        unique_id: 唯一标识符
        current_date: 当前日期
        days: 历史数据天数
    
    Returns:
        historical_data: 历史数据DataFrame
    """
    try:
        # 获取历史数据
        end_date = pd.to_datetime(current_date)
        start_date = end_date - timedelta(days=days)
        
        # 筛选指定商家和产品的历史数据
        historical_data = df[
            (df['seller'] == seller) &
            (df['unique_id'] == unique_id) & 
            (pd.to_datetime(df['date_add_to_bag']).dt.tz_localize(None) >= start_date.tz_localize(None)) & 
            (pd.to_datetime(df['date_add_to_bag']).dt.tz_localize(None) < end_date.tz_localize(None))
        ].copy()
        
        return historical_data
        
    except Exception as e:
        print(f"获取商家历史数据时出错: {e}")
        return pd.DataFrame()

def estimate_seller_inventory_trend(seller_historical_data, seller):
    """
    估算商家库存变化趋势
    
    Args:
        seller_historical_data: 商家历史数据
        seller: 商家名称
    
    Returns:
        inventory_trend: 库存趋势系数
    """
    try:
        if len(seller_historical_data) == 0:
            return 1.0  # 默认无变化
        
        # 计算平均库存变化
        if 'quantity' in seller_historical_data.columns:
            quantities = seller_historical_data['quantity'].dropna()
            if len(quantities) > 1:
                # 计算库存变化趋势
                trend = quantities.iloc[-1] / quantities.iloc[0] if quantities.iloc[0] != 0 else 1.0
                # 限制趋势范围
                trend = max(0.5, min(2.0, trend))
                return trend
        
        return 1.0  # 默认无变化
        
    except Exception as e:
        print(f"估算库存趋势时出错: {e}")
        return 1.0

def estimate_seller_price_strategy(seller_historical_data, seller, current_price):
    """
    估算商家价格策略对库存的影响
    
    Args:
        seller_historical_data: 商家历史数据
        seller: 商家名称
        current_price: 当前价格
    
    Returns:
        price_impact: 价格影响系数
    """
    try:
        if len(seller_historical_data) == 0:
            return 1.0  # 默认无影响
        
        # 分析价格与库存的关系
        if 'price' in seller_historical_data.columns and 'quantity' in seller_historical_data.columns:
            price_data = seller_historical_data[['price', 'quantity']].dropna()
            if len(price_data) > 1:
                # 计算价格变化对库存的影响
                avg_price = price_data['price'].mean()
                if avg_price != 0:
                    price_ratio = current_price / avg_price
                    # 价格越高，库存影响越小（假设需求下降）
                    price_impact = 1.0 / price_ratio if price_ratio > 1 else price_ratio
                    # 限制影响范围
                    price_impact = max(0.7, min(1.3, price_impact))
                    return price_impact
        
        return 1.0  # 默认无影响
        
    except Exception as e:
        print(f"估算价格策略影响时出错: {e}")
        return 1.0

def process_unique_records_per_day_with_seller_focus(df):
    """
    优先基于第一次商家的近期销售数据进行估算
    """
    # 定义用于创建唯一值的字段
    unique_fields = ['model', 'grade_name', 'battery', 'storage', 'color', 'sim_type', 'local']
    
    # 创建唯一标识符
    df['unique_id'] = df[unique_fields].astype(str).agg('_'.join, axis=1)
    
    # 按唯一标识符和日期分组
    grouped = df.groupby(['unique_id', 'date'])
    
    processed_records = []
    estimated_records = []
    
    for (unique_id, date), group in grouped:
        group_sorted = group.sort_values('date_add_to_bag')
        
        if len(group_sorted) == 1:
            original_record = group_sorted.iloc[0].copy()
            first_seller = original_record['seller']
            
            # 原始记录
            original_record['is_estimated'] = False
            original_record['estimation_method'] = 'original'
            processed_records.append(original_record)
            
            # ========== 注释掉估算逻辑部分 ==========
            # # 预测商家变化
            # predicted_seller, change_prob = predict_seller_change(
            #     df, first_seller, unique_id, date
            # )
            # 
            # # 创建估算记录
            # estimated_record = original_record.copy()
            # 
            # # 更新商家信息 - 添加(补)标记
            # if predicted_seller != first_seller:
            #     estimated_record['seller'] = f"{predicted_seller}(补)"
            # else:
            #     estimated_record['seller'] = f"{first_seller}(补)"
            # estimated_record['seller_change_probability'] = change_prob
            # 
            # # 基于新商家的估算逻辑
            # if predicted_seller != first_seller:
            #     # 如果预测商家变化，使用新商家的历史数据
            #     seller_historical_data = get_seller_historical_data(
            #         df, predicted_seller, unique_id, date, days=7
            #     )
            #     estimation_method = f'seller_changed({change_prob:.2f})'
            # else:
            #     # 如果商家不变，使用原商家数据
            #     seller_historical_data = get_seller_historical_data(
            #         df, first_seller, unique_id, date, days=7
            #     )
            #     estimation_method = f'seller_same({change_prob:.2f})'
            # 
            # # 库存估算逻辑
            # seller_inventory_trend = estimate_seller_inventory_trend(
            #     seller_historical_data, predicted_seller
            # )
            # seller_price_impact = estimate_seller_price_strategy(
            #     seller_historical_data, predicted_seller, original_record['price']
            # )
            # 
            # # 商家变化对库存的影响
            # if predicted_seller != first_seller:
            #     # 新商家可能有不同的库存策略
            #     change_impact = -2 if change_prob > 0.6 else -1  # 商家变化通常意味着库存减少
            # else:
            #     change_impact = 0
            # 
            # # 综合估算
            # estimated_quantity = (
            #     original_record['quantity'] + 
            #     seller_inventory_trend + 
            #     seller_price_impact + 
            #     change_impact
            # )
            # estimated_quantity = max(0, int(estimated_quantity))
            # 
            # # 更新估算记录
            # estimated_record['quantity'] = estimated_quantity
            # estimated_record['is_estimated'] = True
            # estimated_record['estimation_method'] = estimation_method
            # estimated_record['date_add_to_bag'] = original_record['date_add_to_bag'] + timedelta(hours=2)
            # 
            # processed_records.append(estimated_record)
            # estimated_records.append(estimated_record)
            # ========== 估算逻辑注释结束 ==========
            
        else:
            # 如果有多条记录，取第一条和最后一条
            first_record = group_sorted.iloc[0].copy()
            last_record = group_sorted.iloc[-1].copy()
            
            # 添加标注字段
            first_record['is_estimated'] = False
            first_record['estimation_method'] = 'original'
            last_record['is_estimated'] = False
            last_record['estimation_method'] = 'original'
            
            processed_records.append(first_record)
            processed_records.append(last_record)
    
    # 创建新的DataFrame，保留所有原始列
    result_df = pd.DataFrame(processed_records)
    # result_df['storage'] = result_df['storage'].str.replace('GB', '').str.replace('Go', '').astype(str)
    
    # 重置索引
    result_df = result_df.reset_index(drop=True)
    
    return result_df

# def main():
#     """主函数：完整的数据处理和导出流程"""
#     print("开始数据处理...")
    
#     # 1. 读取和预处理数据
#     print("\n1. 读取和预处理数据")
#     df = prepare_data()
#     print(f"原始数据条数: {len(df)}")
    
#     # 2. 处理每日唯一记录（带估算和标注）
#     print("\n2. 处理每日唯一记录")
#     processed_df, estimated_records = process_unique_records_per_day_with_seller_focus(df)
#     print(f"处理后数据条数: {len(processed_df)}")
#     print(f"新增估算记录条数: {len(estimated_records)}")
    
#     # 3. 价格验证
#     print("\n3. 进行价格验证")
#     validated_df = validate_price_by_grade(processed_df)
#     print(f"验证后数据条数: {len(validated_df)}")
    
#     # 4. 修复时区问题 - 在导出前移除时区信息
#     print("\n4. 修复时区问题")
#     if 'date_add_to_bag' in validated_df.columns:
#         validated_df['date_add_to_bag'] = validated_df['date_add_to_bag'].dt.tz_localize(None)
    
#     # 5. 导出到Excel
#     print("\n5. 导出数据到Excel")
#     output_filename = f"processed_inventory_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
#     with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
#         # 导出完整处理后的数据
#         validated_df.to_excel(writer, sheet_name='完整数据', index=False)
        
#         # 导出仅新增的估算记录
#         if estimated_records:
#             estimated_df = pd.DataFrame(estimated_records)
#             # 同样修复估算记录的时区问题
#             if 'date_add_to_bag' in estimated_df.columns:
#                 estimated_df['date_add_to_bag'] = estimated_df['date_add_to_bag'].dt.tz_localize(None)
#             estimated_df.to_excel(writer, sheet_name='新增估算记录', index=False)
        
#         # 导出统计摘要
#         summary_data = {
#             '统计项目': [
#                 '原始数据条数',
#                 '处理后数据条数', 
#                 '新增估算记录条数',
#                 '价格验证后数据条数',
#                 '最终数据条数'
#             ],
#             '数量': [
#                 len(df),
#                 len(processed_df),
#                 len(estimated_records),
#                 len(validated_df),
#                 len(validated_df)
#             ]
#         }
#         summary_df = pd.DataFrame(summary_data)
#         summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
    
#     print(f"\n数据处理完成！")
#     print(f"输出文件: {output_filename}")
#     print(f"新增估算记录条数: {len(estimated_records)}")
    
    # return validated_df
    # return validated_df, estimated_records



def get_complete_dataframe():
    """
    返回包含原始数据和估算数据的完整DataFrame，供app.py使用
    """
    # 1. 基础数据预处理
    df = prepare_data()
    
    # 2. 处理每日唯一记录（带估算）
    processed_df = process_unique_records_per_day_with_seller_focus(df)  # 只接收一个返回值
    
    # 3. 价格验证
    validated_df = validate_price_by_grade(processed_df)
    
    # 4. 修复时区问题
    if 'date_add_to_bag' in validated_df.columns:
        validated_df['date_add_to_bag'] = validated_df['date_add_to_bag'].dt.tz_localize(None)
    
    # 统计估算记录数量
    estimated_count = len(validated_df[validated_df.get('is_estimated', False) == True])
    
    print(f"完整数据集包含 {len(validated_df)} 条记录")
    print(f"其中估算记录 {estimated_count} 条")
    
    return validated_df

def export_all_processed_data():
    """导出所有处理后的数据到processed_data文件夹"""
    # 创建输出文件夹
    output_dir = "e:\\项目\\app\\processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始处理所有数据...")
    
    # 1. 处理原始数据
    print("\n1. 处理原始数据")
    df = prepare_data()
    processed_df = process_unique_records_per_day_with_seller_focus(df)
    validated_df = validate_price_by_grade(processed_df)
    
    # 修复时区问题
    if 'date_add_to_bag' in validated_df.columns:
        validated_df['date_add_to_bag'] = validated_df['date_add_to_bag'].dt.tz_localize(None)
    
    # 2. 处理Excel数据
    print("\n2. 处理Excel数据")
    xq_data, ljh_data = load_excel_data()
    
    # 处理内存列
    if not xq_data.empty:
        xq_data = process_memory_column(xq_data)
    if not ljh_data.empty:
        ljh_data = process_memory_column(ljh_data)
    
    # 3. 生成库存和价格数据
    print("\n3. 生成库存和价格数据")
    color_inventory_df, total_inventory_df = create_daily_color_inventory_data(validated_df)
    daily_price_df = create_daily_price_data(validated_df)
    
    # 4. 生成统一筛选选项
    print("\n4. 生成统一筛选选项")
    filter_options = generate_unified_filter_options(validated_df, xq_data, ljh_data)
    
    # 5. 导出所有数据
    print("\n5. 导出数据到文件")
    
    # 导出原始数据处理结果
    validated_df.to_excel(os.path.join(output_dir, "original_data.xlsx"), index=False)
    
    # 导出Excel数据
    with pd.ExcelWriter(os.path.join(output_dir, "excel_data.xlsx"), engine='openpyxl') as writer:
        if not xq_data.empty:
            xq_data.to_excel(writer, sheet_name='讯强全新机', index=False)
        if not ljh_data.empty:
            ljh_data.to_excel(writer, sheet_name='靓机汇二手回收', index=False)
    
    # 导出库存数据
    with pd.ExcelWriter(os.path.join(output_dir, "daily_inventory_data.xlsx"), engine='openpyxl') as writer:
        color_inventory_df.to_excel(writer, sheet_name='颜色库存', index=False)
        total_inventory_df.to_excel(writer, sheet_name='总库存', index=False)
    
    # 导出价格数据
    daily_price_df.to_excel(os.path.join(output_dir, "daily_price_data.xlsx"), index=False)
    
    # 导出筛选选项
    with open(os.path.join(output_dir, "unified_filter_options.json"), 'w', encoding='utf-8') as f:
        json.dump(filter_options, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据处理完成！")
    print(f"输出文件夹: {output_dir}")
    print(f"原始数据: {len(validated_df)} 条")
    print(f"讯强数据: {len(xq_data)} 条")
    print(f"靓机汇数据: {len(ljh_data)} 条")
    print(f"颜色库存数据: {len(color_inventory_df)} 条")
    print(f"价格数据: {len(daily_price_df)} 条")
    
    return output_dir

def create_daily_color_inventory_data(df):
    """创建每日颜色库存数据"""
    # 创建unique_key
    df['unique_key'] = (
        df['model'].astype(str) + '_' +
        df['grade_name'].astype(str) + '_' +
        df['battery'].astype(str) + '_' +
        df['storage'].astype(str) + '_' +
        df['sim_type'].astype(str) + '_' +
        df['local'].astype(str) + '_' +
        df['color'].astype(str)
    )
    
    # 按日期分组处理库存数据
    daily_color_inventory = []
    daily_total_inventory = []
    
    for date in sorted(df['date'].unique()):
        date_data = df[df['date'] == date].copy()
        
        if len(date_data) == 0:
            continue
        
        # 按unique_key分组
        grouped = date_data.groupby('unique_key')
        
        daily_first_color = {}
        daily_last_color = {}
        daily_first_sellers = {}  
        daily_last_sellers = {}   
        daily_first_prices = {}   
        daily_last_prices = {}    
        daily_first_total = 0
        daily_last_total = 0
        
        for unique_key, group in grouped:
            group = group.sort_values('date_add_to_bag')
            color = group['color'].iloc[0]
            
            # 获取seller和价格信息
            first_seller = group['seller'].iloc[0] if 'seller' in group.columns else '未知'
            last_seller = group['seller'].iloc[-1] if 'seller' in group.columns else '未知'
            first_price = group['price'].iloc[0] if 'price' in group.columns else 0
            last_price = group['price'].iloc[-1] if 'price' in group.columns else 0
            
            # 使用quantity列或记录数
            if 'quantity' in group.columns:
                first_inventory = pd.to_numeric(group['quantity'].iloc[0], errors='coerce')
                last_inventory = pd.to_numeric(group['quantity'].iloc[-1], errors='coerce')
                
                if pd.isna(first_inventory):
                    first_inventory = 1
                if pd.isna(last_inventory):
                    last_inventory = len(group)
            else:
                first_inventory = 1
                last_inventory = len(group)
            
            # 累加颜色库存
            daily_first_color[color] = daily_first_color.get(color, 0) + first_inventory
            daily_last_color[color] = daily_last_color.get(color, 0) + last_inventory
            
            # 收集seller信息
            if color not in daily_first_sellers:
                daily_first_sellers[color] = []
            if color not in daily_last_sellers:
                daily_last_sellers[color] = []
            
            daily_first_sellers[color].append(first_seller)
            daily_last_sellers[color].append(last_seller)
            
            # 收集价格信息
            if color not in daily_first_prices:
                daily_first_prices[color] = []
            if color not in daily_last_prices:
                daily_last_prices[color] = []
            
            daily_first_prices[color].append(first_price)
            daily_last_prices[color].append(last_price)
            
            # 累加总库存
            daily_first_total += first_inventory
            daily_last_total += last_inventory
        
        # 记录每个颜色的库存和seller信息
        for color in set(list(daily_first_color.keys()) + list(daily_last_color.keys())):
            # 处理seller信息，去重并合并
            first_sellers_list = list(set(daily_first_sellers.get(color, [])))
            last_sellers_list = list(set(daily_last_sellers.get(color, [])))
            
            # 计算平均价格
            first_prices_list = daily_first_prices.get(color, [])
            last_prices_list = daily_last_prices.get(color, [])
            first_avg_price = sum(first_prices_list) / len(first_prices_list) if first_prices_list else 0
            last_avg_price = sum(last_prices_list) / len(last_prices_list) if last_prices_list else 0
            
            daily_color_inventory.append({
                'date': pd.to_datetime(date),
                'color': color,
                'first_inventory': daily_first_color.get(color, 0),
                'last_inventory': daily_last_color.get(color, 0),
                'first_sellers': ', '.join(first_sellers_list),
                'last_sellers': ', '.join(last_sellers_list),
                'first_avg_price': first_avg_price,
                'last_avg_price': last_avg_price
            })
        
        # 记录总库存
        daily_total_inventory.append({
            'date': pd.to_datetime(date),
            'first_total': daily_first_total,
            'last_total': daily_last_total
        })
    
    # 转换为DataFrame
    color_inventory_df = pd.DataFrame(daily_color_inventory)
    total_inventory_df = pd.DataFrame(daily_total_inventory)
    
    return color_inventory_df, total_inventory_df


def create_daily_price_data(df):
    """创建每日价格数据（K线图用）"""
    daily_price_data = [] 
    
    for date in sorted(df['date'].unique()): 
        date_data = df[df['date'] == date].copy() 
        
        if len(date_data) == 0: 
            continue 
        
        # 按颜色分组，取每种颜色的第一次和最后一次价格 
        color_groups = date_data.groupby('color') 
        first_prices = []  # 存储每种颜色第一次出现的价格 
        last_prices = []   # 存储每种颜色最后一次出现的价格 
        all_prices = []    # 存储所有价格用于计算最高最低价 
        
        for color, group in color_groups: 
            group_sorted = group.sort_values('date_add_to_bag') 
            group_prices = group_sorted['price'].dropna() 
            
            if len(group_prices) > 0: 
                first_prices.append(group_prices.iloc[0])  # 第一次出现 
                last_prices.append(group_prices.iloc[-1])  # 最后一次出现 
                all_prices.extend(group_prices.tolist()) 
        
        if len(first_prices) == 0 or len(last_prices) == 0: 
            continue 
        
        # 计算中位数 
        first_median = pd.Series(first_prices).median()  # 所有颜色第一次价格的中位数 
        second_median = pd.Series(last_prices).median()  # 所有颜色最后一次价格的中位数 
        
        daily_price_data.append({ 
            'date': pd.to_datetime(date), 
            'first_median': first_median, 
            'second_median': second_median, 
            'open': first_median, 
            'close': second_median, 
            'high': max(all_prices) if all_prices else first_median, 
            'low': min(all_prices) if all_prices else first_median 
        }) 
    
    return pd.DataFrame(daily_price_data)


def generate_unified_filter_options(df, xq_data, ljh_data):
    """生成统一的筛选选项"""
    # 统一的型号选项（合并所有数据源）
    all_models = set()
    if not df.empty:
        all_models.update(df['model'].dropna().unique())
    if not xq_data.empty:
        all_models.update(xq_data['型号'].dropna().unique())
    if not ljh_data.empty:
        all_models.update(ljh_data['型号'].dropna().unique())
    
    unified_model_options = [{'label': '全部', 'value': '全部'}] + \
                           [{'label': str(model), 'value': str(model)} for model in sorted(all_models)]
    
    # 统一的内存选项（强制转换为int类型）
    all_memory = set()
    
    # 处理原始数据的storage字段
    if not df.empty:
        for storage in df['storage'].dropna():
            try:
                # 提取数字部分并转换为int
                memory_str = str(storage).replace('G', '').replace('g', '').strip()
                if memory_str.isdigit():
                    all_memory.add(int(memory_str))
            except:
                continue
    
    # 处理讯强数据的内存字段
    if not xq_data.empty:
        for memory in xq_data['内存'].dropna():
            try:
                # 处理纳米：将"纳米"替换为"1"
                memory_str = str(memory).replace('纳米', '1').replace('G', '').replace('g', '').strip()
                memory_int = int(float(memory_str))
                all_memory.add(memory_int)
            except:
                continue
    
    # 处理靓机汇数据的内存字段
    if not ljh_data.empty:
        for memory in ljh_data['内存'].dropna():
            try:
                # 处理纳米：将"纳米"替换为"1"
                memory_str = str(memory).replace('纳米', '1').replace('G', '').replace('g', '').strip()
                memory_int = int(float(memory_str))
                all_memory.add(memory_int)
            except:
                continue
    
    # 创建内存选项（纯数字，按大小排序）
    unified_memory_options = [{'label': '全部', 'value': '全部'}] + \
                            [{'label': str(mem), 'value': str(mem)} for mem in sorted(all_memory)]
    
    # 统一的SIM类型选项（合并所有数据源）
    all_sim_types = set()
    if not df.empty:
        all_sim_types.update(df['sim_type'].dropna().unique())
    if not xq_data.empty:
        all_sim_types.update(xq_data['版本'].dropna().unique())
    if not ljh_data.empty:
        all_sim_types.update(ljh_data['版本'].dropna().unique())
    
    unified_sim_options = [{'label': '全部', 'value': '全部'}] + \
                         [{'label': str(sim_type), 'value': str(sim_type)} for sim_type in sorted(all_sim_types)]
    
    # 其他筛选选项（基于原始数据）- 转换为Python原生类型
    grade_options = [{'label': '全部', 'value': '全部'}] + \
                   [{'label': str(grade), 'value': str(grade)} for grade in sorted(df['grade_name'].dropna().unique().tolist())]
    battery_options = [{'label': '全部', 'value': '全部'}] + \
                     [{'label': str(battery), 'value': str(battery)} for battery in sorted(df['battery'].dropna().unique().tolist())]
    local_options = [{'label': '全部', 'value': '全部'}] + \
                   [{'label': str(local), 'value': str(local)} for local in sorted(df['local'].dropna().unique().tolist())]
    
    return {
        'model_options': unified_model_options,
        'memory_options': unified_memory_options,
        'sim_options': unified_sim_options,
        'grade_options': grade_options,
        'battery_options': battery_options,
        'local_options': local_options
    }

if __name__ == "__main__":
    # 导出所有处理后的数据
    export_all_processed_data()


