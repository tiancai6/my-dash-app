import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_storage_field(df):
    """清理storage字段中的GB和Go单位"""
    df = df.copy()
    if 'storage' in df.columns:
        df['storage'] = df['storage'].astype(str).str.replace('GB', '').str.replace('Go', '').str.strip()
    return df

def process_ipad_data(df, original_columns):
    """处理iPad数据补全"""
    print("\n=== 开始处理iPad数据 ===")
    
    # 筛选iPad数据
    ipad_data = df[df['型号'].str.contains('iPad', na=False)].copy()
    print(f"总iPad数据行数: {len(ipad_data)}")
    
    if len(ipad_data) == 0:
        print("未找到iPad数据")
        return pd.DataFrame()
    
    # 筛选德国和法国iPad数据
    de_ipad = ipad_data[ipad_data['local'].str.contains('de', na=False)].copy()
    fr_ipad = ipad_data[ipad_data['local'].str.contains('fr', na=False)].copy()
    
    print(f"德国iPad数据行数: {len(de_ipad)}")
    print(f"法国iPad数据行数: {len(fr_ipad)}")
    
    if len(de_ipad) == 0 or len(fr_ipad) == 0:
        print("德国或法国iPad数据为空，无法进行补全")
        return pd.DataFrame()
    
    # 创建base_id用于匹配
    for data in [de_ipad, fr_ipad]:
        data['base_id'] = (
            data['型号'].astype(str) + '_' +
            data['磨损中文'].astype(str) + '_' +
            data['battery(1新,0旧)'].astype(str) + '_' +
            data['storage'].astype(str) + '_' +
            data['颜色中文'].astype(str) + '_' +
            pd.to_datetime(data['date_add_to_bag']).dt.strftime('%Y-%m-%d')
        )
    
    # 找出德国只有WF没有CL的组合
    de_wf = de_ipad[de_ipad['sim类型'] == 'WF']
    de_cl = de_ipad[de_ipad['sim类型'] == 'CL']
    
    de_wf_base_ids = set(de_wf['base_id'])
    de_cl_base_ids = set(de_cl['base_id'])
    
    missing_cl_base_ids = de_wf_base_ids - de_cl_base_ids
    print(f"德国缺少CL数据的组合数: {len(missing_cl_base_ids)}")
    
    if len(missing_cl_base_ids) == 0:
        print("德国iPad数据完整，无需补全")
        return pd.DataFrame()
    
    # 从法国数据中查找对应的CL数据
    fr_cl = fr_ipad[fr_ipad['sim类型'] == 'CL']
    fr_cl_base_ids = set(fr_cl['base_id'])
    
    can_supplement_base_ids = missing_cl_base_ids & fr_cl_base_ids
    print(f"可以从法国补全的组合数: {len(can_supplement_base_ids)}")
    
    if len(can_supplement_base_ids) == 0:
        print("法国没有对应的CL数据可供补全")
        return pd.DataFrame()
    
    # 执行补全 - 保持完整的列结构
    supplement_data = []
    for base_id in can_supplement_base_ids:
        fr_record = fr_cl[fr_cl['base_id'] == base_id].iloc[0].copy()
        fr_record['local'] = 'de-de'
        # 删除临时的base_id列
        fr_record = fr_record.drop('base_id')
        supplement_data.append(fr_record)
    
    if supplement_data:
        supplement_df = pd.DataFrame(supplement_data)
        # 确保列顺序与原始数据一致
        supplement_df = supplement_df.reindex(columns=original_columns, fill_value=None)
        print(f"实际补全iPad数据行数: {len(supplement_df)}")
        return supplement_df
    else:
        return pd.DataFrame()

def process_iphone_data(df, original_columns):
    """处理iPhone数据补全"""
    print("\n=== 开始处理iPhone数据 ===")
    
    # 筛选iPhone数据（仅"4.高级"状态）
    iphone_data = df[
        (df['型号'].str.contains('iPhone', na=False)) & 
        (df['磨损中文'] == '4.高级')
    ].copy()
    print(f"总iPhone \"4.高级\"数据行数: {len(iphone_data)}")
    
    if len(iphone_data) == 0:
        print("未找到iPhone \"4.高级\"数据")
        return pd.DataFrame()
    
    # 筛选德国和法国iPhone数据
    de_iphone = iphone_data[iphone_data['local'].str.contains('de', na=False)].copy()
    fr_iphone = iphone_data[iphone_data['local'].str.contains('fr', na=False)].copy()
    
    print(f"德国iPhone \"4.高级\"数据行数: {len(de_iphone)}")
    print(f"法国iPhone \"4.高级\"数据行数: {len(fr_iphone)}")
    
    if len(fr_iphone) == 0:
        print("法国iPhone数据为空，无法进行补全")
        return pd.DataFrame()
    
    # 创建base_id用于匹配
    for data in [de_iphone, fr_iphone]:
        data['base_id'] = (
            data['型号'].astype(str) + '_' +
            data['battery(1新,0旧)'].astype(str) + '_' +
            data['storage'].astype(str) + '_' +
            data['sim类型'].astype(str) + '_' +
            data['颜色中文'].astype(str) + '_' +
            pd.to_datetime(data['date_add_to_bag']).dt.strftime('%Y-%m-%d')
        )
    
    # 找出法国有但德国没有的组合
    fr_base_ids = set(fr_iphone['base_id'])
    de_base_ids = set(de_iphone['base_id']) if len(de_iphone) > 0 else set()
    
    missing_base_ids = fr_base_ids - de_base_ids
    print(f"德国缺少的iPhone组合数: {len(missing_base_ids)}")
    
    if len(missing_base_ids) == 0:
        print("德国iPhone数据完整，无需补全")
        return pd.DataFrame()
    
    # 执行补全 - 保持完整的列结构
    supplement_data = []
    for base_id in missing_base_ids:
        fr_record = fr_iphone[fr_iphone['base_id'] == base_id].iloc[0].copy()
        fr_record['local'] = 'de-de'
        # 删除临时的base_id列
        fr_record = fr_record.drop('base_id')
        supplement_data.append(fr_record)
    
    if supplement_data:
        supplement_df = pd.DataFrame(supplement_data)
        # 确保列顺序与原始数据一致
        supplement_df = supplement_df.reindex(columns=original_columns, fill_value=None)
        print(f"实际补全iPhone数据行数: {len(supplement_df)}")
        return supplement_df
    else:
        return pd.DataFrame()

def main():
    # 读取数据
    file_path = r'e:\项目\app\库存信息-更新至0714.xlsx'
    print(f"正在读取文件: {file_path}")
    
    try:
        df_original = pd.read_excel(file_path, sheet_name="库存信息")
        print(f"成功读取数据，总行数: {len(df_original)}")
        print(f"列名: {list(df_original.columns)}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 保存原始列名顺序
    original_columns = list(df_original.columns)
    
    print("\n=== 数据预处理 ===")
    
    # 检查必要的列是否存在
    required_columns = ['型号', '磨损中文', 'battery(1新,0旧)', 'storage', '颜色中文', 'sim类型', 'local', 'date_add_to_bag']
    missing_cols = [col for col in required_columns if col not in df_original.columns]
    if missing_cols:
        print(f"缺少必要列: {missing_cols}")
        print(f"实际列名: {list(df_original.columns)}")
        return
    
    # 创建工作副本进行处理
    df_work = df_original.copy()
    
    # 清理storage字段
    df_work = clean_storage_field(df_work)
    
    # 转换日期
    df_work['date_add_to_bag'] = pd.to_datetime(df_work['date_add_to_bag'], errors='coerce')
    
    # 筛选2025年7月1日至7月12日的数据进行补全分析
    start_date = pd.Timestamp('2025-07-01')
    end_date = pd.Timestamp('2025-07-14')
    after_cutoff = df_work[
        (df_work['date_add_to_bag'] >= start_date) & 
        (df_work['date_add_to_bag'] <= end_date)
    ].copy()
    print(f"2025年7月1日至7月12日的数据行数: {len(after_cutoff)}")
    
    # 处理iPad数据
    ipad_supplement = process_ipad_data(after_cutoff, original_columns)
    
    # 处理iPhone数据
    iphone_supplement = process_iphone_data(after_cutoff, original_columns)
    
    # 合并补全数据
    all_supplement = pd.DataFrame()
    if len(ipad_supplement) > 0:
        all_supplement = pd.concat([all_supplement, ipad_supplement], ignore_index=True)
    if len(iphone_supplement) > 0:
        all_supplement = pd.concat([all_supplement, iphone_supplement], ignore_index=True)
    
    print(f"\n=== 补全结果汇总 ===")
    print(f"iPad补全数据行数: {len(ipad_supplement)}")
    print(f"iPhone补全数据行数: {len(iphone_supplement)}")
    print(f"总补全数据行数: {len(all_supplement)}")
    
    # 保存结果
    if len(all_supplement) > 0:
        # 与原始数据合并
        final_df = pd.concat([df_original, all_supplement], ignore_index=True)
        
        # 保存文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'e:\\项目\\补全后的库存数据_{timestamp}.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name='库存信息', index=False)
        
        print(f"\n补全完成！文件已保存为: {output_file}")
        print(f"原始数据行数: {len(df_original)}")
        print(f"补全后总行数: {len(final_df)}")
        print(f"新增数据行数: {len(all_supplement)}")
        
        # 更新process_and_export.py中的文件路径
        update_process_file_path(output_file)
        
    else:
        print("\n没有需要补全的数据")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'e:\\项目\\检查后的库存数据_{timestamp}.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_original.to_excel(writer, sheet_name='库存信息', index=False)
        
        print(f"原始数据已保存为: {output_file}")
        update_process_file_path(output_file)

def update_process_file_path(new_file_path):
    """自动更新process_and_export.py中的文件路径"""
    process_file = r'e:\项目\app\process_and_export.py'
    
    try:
        with open(process_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式替换文件路径
        import re
        pattern = r'df = pd\.read_excel\([^)]+\)'
        replacement = f'df = pd.read_excel("{new_file_path}", sheet_name="库存信息")'
        
        new_content = re.sub(pattern, replacement, content)
        
        with open(process_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"已自动更新 process_and_export.py 中的文件路径为: {new_file_path}")
        
    except Exception as e:
        print(f"更新 process_and_export.py 失败: {e}")
        print(f"请手动将 process_and_export.py 第13行的文件路径改为: {new_file_path}")

if __name__ == '__main__':
    main()