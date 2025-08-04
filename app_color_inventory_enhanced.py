# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
import os

import matplotlib.pyplot as plt
import seaborn as sns

# color 到 RGB 的映射
color_to_rgb = {
    "玫瑰色": (255, 102, 204),    # 保留，符合玫瑰红标准
    "粉红色": (255, 182, 193),    # 修正：标准粉红是(255,182,193)
    "粉色": (255, 182, 193),    # 修正：标准粉红是(255,182,193)
    "蓝色": (0, 0, 255),          # 保留
    "银色": (192, 192, 192),      # 保留
    "黄色": (255, 255, 0),        # 保留
    "太空灰": (142, 142, 147),    # 修正：苹果太空灰(142,142,147)
    "星光色": (245, 245, 240),     # 修正：苹果星光色(245,245,240)
    "紫色": (128, 0, 128),        # 保留
    "深空黑": (30, 30, 30),        # 修正：深空黑(30,30,30)
    "白色": (255, 255, 255),      # 保留
    "群青蓝": (65, 105, 225),     # 保留
    "群青色": (65, 105, 225),     # 保留
    "蓝绿色": (0, 255, 255),      # 保留
    "深青色": (0, 255, 255),      # 保留
    "黑色": (0, 0, 0),            # 保留
    "钛金属原色": (170, 170, 170), # 修正：钛金属原色(170,170,170)
    "钛金属沙色": (191, 177, 136), # 修正：钛金属沙色(191,177,136)
    "钛金属白": (230, 230, 230),   # 修正：钛金属白(230,230,230)
    "钛金属黑": (68, 68, 68)       # 修正：钛金属黑(68,68,68)
}

def load_processed_data():
    """加载处理好的原始数据文件"""
    try:
        df = pd.read_excel("./processed_data/original_data.xlsx")
        return df
    except Exception as e:
        print(f"读取原始数据失败: {e}")
        return pd.DataFrame()

def load_excel_data():
    """从处理好的文件中加载Excel数据"""
    try:
        # 首先尝试读取项目内的excel_data.xlsx文件
        excel_file_path = "./processed_data/excel_data.xlsx"
        
        # 检查文件是否存在
        if not os.path.exists(excel_file_path):
            print(f"文件不存在: {excel_file_path}")
            return pd.DataFrame(), pd.DataFrame()
        
        # 获取所有工作表名称
        try:
            xl_file = pd.ExcelFile(excel_file_path)
            sheet_names = xl_file.sheet_names
            print(f"Excel文件包含的工作表: {sheet_names}")
        except Exception as e:
            print(f"无法读取Excel文件工作表列表: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # 初始化数据框
        xq_data = pd.DataFrame()
        ljh_data = pd.DataFrame()
        
        # 尝试读取讯强数据工作表
        xq_sheet_names = ['讯强全新机', '全新机讯强', '讯强', 'XQ']
        for sheet_name in xq_sheet_names:
            if sheet_name in sheet_names:
                try:
                    xq_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                    print(f"成功读取讯强数据工作表 '{sheet_name}': {len(xq_data)} 行")
                    print(f"讯强数据列名: {list(xq_data.columns)}")
                    if len(xq_data) > 0:
                        print(f"讯强数据前3行:\n{xq_data.head(3)}")
                    break
                except Exception as e:
                    print(f"读取工作表 '{sheet_name}' 失败: {e}")
        
        if xq_data.empty:
            print(f"未找到讯强数据工作表，尝试的名称: {xq_sheet_names}")
        
        # 尝试读取靓机汇数据工作表
        ljh_sheet_names = ['靓机汇二手回收', '靓机汇二手机', '靓机汇', 'LJH']
        for sheet_name in ljh_sheet_names:
            if sheet_name in sheet_names:
                try:
                    ljh_data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                    print(f"成功读取靓机汇数据工作表 '{sheet_name}': {len(ljh_data)} 行")
                    break
                except Exception as e:
                    print(f"读取工作表 '{sheet_name}' 失败: {e}")
        
        if ljh_data.empty:
            print(f"未找到靓机汇数据工作表，尝试的名称: {ljh_sheet_names}")
        
        print(f"总数据行数: {len(xq_data) + len(ljh_data)}")
        
        # 如果项目内文件读取失败，尝试外部文件
        if xq_data.empty and ljh_data.empty:
            print("项目内文件读取失败，尝试外部文件...")
            external_file_path = "E:\楚讯实业\iPhone 16系列所有数据汇总.xlsx"
            try:
                if os.path.exists(external_file_path):
                    xq_data = pd.read_excel(external_file_path, sheet_name='全新机讯强')
                    ljh_data = pd.read_excel(external_file_path, sheet_name='靓机汇二手机')
                    print(f"外部文件读取成功 - 讯强: {len(xq_data)}, 靓机汇: {len(ljh_data)}")
                else:
                    print(f"外部文件不存在: {external_file_path}")
            except Exception as e3:
                print(f"外部文件也读取失败: {e3}")
        
        return xq_data, ljh_data
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return pd.DataFrame(), pd.DataFrame()

def process_memory_column(df, memory_col='内存'):
    """处理内存列（数据已预处理，保持函数接口一致）"""
    # 数据已经在process_and_export.py中处理过，这里只是保持接口一致
    return df

# 全局变量定义
xq_data = pd.DataFrame()
ljh_data = pd.DataFrame()

def create_color_inventory_app(df):
    """
    创建每日颜色库存分析的Dash应用，包含6个筛选器和4个子图。
    新增：最后一次出现的颜色库存数据子图，以及讯强和靓机汇走势图
    """
    global xq_data, ljh_data
    
    # 加载Excel数据
    xq_data, ljh_data = load_excel_data()
    
    # 处理内存列
    if not xq_data.empty:
        xq_data = process_memory_column(xq_data)
    if not ljh_data.empty:
        ljh_data = process_memory_column(ljh_data)
    
    df = df.copy()
    # 确保筛选字段为字符串
    filter_columns = ['model', 'grade_name', 'battery', 'storage', 'sim_type', 'local']
    for col in filter_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'date' not in df.columns:
        df['date'] = df['date_add_to_bag'].dt.date
    df = df.dropna(subset=['price'])
    if df.empty:
        print("数据为空，无法创建应用")
        return None

    # 生成筛选选项
    model_options = [{'label': '全部', 'value': '全部'}] + [{'label': m, 'value': m} for m in sorted(df['model'].unique())]
    memory_set = set()
    for storage in df['storage'].dropna():
        try:
            mem = str(storage).replace('G', '').replace('g', '').strip()
            if mem.isdigit():
                memory_set.add(int(mem))
        except:
            continue
    memory_options = [{'label': '全部', 'value': '全部'}] + [{'label': str(m), 'value': str(m)} for m in sorted(memory_set)]
    sim_options = [{'label': '全部', 'value': '全部'}] + [{'label': s, 'value': s} for s in sorted(df['sim_type'].unique())]
    grade_options = [{'label': '全部', 'value': '全部'}] + [{'label': g, 'value': g} for g in sorted(df['grade_name'].unique())]
    battery_options = [{'label': '全部', 'value': '全部'}] + [{'label': b, 'value': b} for b in sorted(df['battery'].unique())]
    local_options = [{'label': '全部', 'value': '全部'}] + [{'label': l, 'value': l} for l in sorted(df['local'].unique())]

    # Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([
        html.H1("每日颜色库存分析（增强版）", style={'textAlign': 'center', 'marginBottom': 30}),
        
        # 第一行筛选器：型号、内存、SIM类型
        html.Div([
            html.Div([
                html.Label("型号:"),
                dcc.Dropdown(
                    id='model-dropdown', 
                    options=model_options, 
                    value='全部', 
                    style={'width': '250px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '15px'}),
            
            html.Div([
                html.Label("内存:"),
                dcc.Dropdown(
                    id='memory-dropdown', 
                    options=memory_options, 
                    value='全部', 
                    style={'width': '180px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '15px'}),
            
            html.Div([
                html.Label("SIM类型:"),
                dcc.Dropdown(
                    id='sim-dropdown', 
                    options=sim_options, 
                    value='全部', 
                    style={'width': '180px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '15px'}),
        ], style={'textAlign': 'center', 'marginBottom': 20}),
        
        # 第二行筛选器：磨损、电池、地区
        html.Div([
            html.Div([
                html.Label("磨损:"),
                dcc.Dropdown(
                    id='grade-dropdown', 
                    options=grade_options, 
                    value='全部', 
                    style={'width': '180px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '15px'}),
            
            html.Div([
                html.Label("电池:"),
                dcc.Dropdown(
                    id='battery-dropdown', 
                    options=battery_options, 
                    value='全部', 
                    style={'width': '180px'}
                )
            ], style={'display': 'inline-block', 'marginRight': '15px'}),
            
            html.Div([html.Label("地区:"),
                dcc.Dropdown(
                    id='local-dropdown', 
                    options=local_options, 
                    value='全部', 
                    style={'width': '180px'}
                )
            ], style={'display': 'inline-block'}),
        ], style={'textAlign': 'center', 'marginBottom': 20}),
        
        # 添加标签页
        dcc.Tabs(id='main-tabs', value='color-inventory', children=[
            dcc.Tab(label='颜色库存分析', value='color-inventory'),
            dcc.Tab(label='讯强走势图（2psim)', value='xq-analysis'),
            dcc.Tab(label='靓机汇走势图（2psim)', value='ljh-analysis')
        ], style={'marginBottom': 20}),
        
        html.Div(id='chart-content')
    ])

    @app.callback(
        Output('chart-content', 'children'),
        [
            Input('main-tabs', 'value'),
            Input('model-dropdown', 'value'),
            Input('memory-dropdown', 'value'),
            Input('sim-dropdown', 'value'),
            Input('grade-dropdown', 'value'),
            Input('battery-dropdown', 'value'),
            Input('local-dropdown', 'value')
        ]
    )
    def update_chart_content(selected_tab, model_val, memory_val, sim_val, grade_val, battery_val, local_val):
        global xq_data, ljh_data
        
        # 调试信息：检查全局变量状态
        print(f"回调函数中 - xq_data行数: {len(xq_data)}, ljh_data行数: {len(ljh_data)}")
        
        # 设置默认值
        model_val = model_val or '全部'
        memory_val = memory_val or '全部'
        sim_val = sim_val or '全部'
        grade_val = grade_val or '全部'
        battery_val = battery_val or '全部'
        local_val = local_val or '全部'
        
        try:
            if selected_tab == 'color-inventory':
                # 颜色库存分析
                filtered_df = df.copy()
                if model_val != '全部':
                    filtered_df = filtered_df[filtered_df['model'] == model_val]
                if memory_val != '全部':
                    def match_memory(storage_val):
                        try:
                            storage_int = int(str(storage_val).replace('G', '').replace('g', '').strip())
                            return storage_int == int(memory_val)
                        except:
                            return False
                    filtered_df = filtered_df[filtered_df['storage'].apply(match_memory)]
                if sim_val != '全部':
                    filtered_df = filtered_df[filtered_df['sim_type'] == sim_val]
                if grade_val != '全部':
                    filtered_df = filtered_df[filtered_df['grade_name'] == grade_val]
                if battery_val != '全部':
                    filtered_df = filtered_df[filtered_df['battery'] == battery_val]
                if local_val != '全部':
                    filtered_df = filtered_df[filtered_df['local'] == local_val]
                return dcc.Graph(figure=create_color_inventory_figure(filtered_df))
            
            elif selected_tab == 'xq-analysis':
                # 讯强走势图分析
                if xq_data.empty:
                    return html.Div([
                        html.H3("讯强走势图", style={'textAlign': 'center'}),
                        html.Div([
                            html.P("暂无讯强数据", style={'textAlign': 'center', 'color': 'red', 'fontSize': '18px'}),
                            html.P("可能的原因:", style={'textAlign': 'center', 'marginTop': '20px'}),
                            html.Ul([
                                html.Li("Excel文件中没有讯强相关的工作表"),
                                html.Li("工作表名称不匹配（尝试的名称: 讯强全新机, 全新机讯强, 讯强, XQ）"),
                                html.Li("数据文件路径不正确")
                            ], style={'textAlign': 'left', 'maxWidth': '500px', 'margin': '0 auto'}),
                            html.P("请检查 ./processed_data/excel_data.xlsx 文件", 
                                  style={'textAlign': 'center', 'marginTop': '20px', 'fontStyle': 'italic'})
                        ], style={'padding': '40px'})
                    ])
                
                filtered_data = xq_data.copy()
                
                if model_val != '全部':
                    filtered_data = filtered_data[filtered_data['型号'] == model_val]
                if memory_val != '全部':
                    def match_xq_memory(mem_val):
                        try:
                            mem_str = str(mem_val).replace('纳米', '1').replace('G', '').replace('g', '').strip()
                            mem_int = int(float(mem_str))
                            return mem_int == int(memory_val)
                        except:
                            return False
                    filtered_data = filtered_data[filtered_data['内存'].apply(match_xq_memory)]
                if sim_val != '全部':
                    filtered_data = filtered_data[filtered_data['版本'] == sim_val]
                
                # 返回两个图表：箱型图和颜色价格折线图
                return html.Div([
                    html.H3("价格分布箱型图", style={'textAlign': 'center', 'marginTop': 20}),
                    dcc.Graph(figure=create_xq_analysis_chart(filtered_data)),
                    html.H3("各颜色价格趋势", style={'textAlign': 'center', 'marginTop': 30}),
                    dcc.Graph(figure=create_xq_color_price_chart(filtered_data))
                ])
            
            elif selected_tab == 'ljh-analysis':
                # 靓机汇走势图分析
                if ljh_data.empty:
                    return html.Div([
                        html.H3("靓机汇走势图", style={'textAlign': 'center'}),
                        html.Div([
                            html.P("暂无靓机汇数据", style={'textAlign': 'center', 'color': 'red', 'fontSize': '18px'}),
                            html.P("可能的原因:", style={'textAlign': 'center', 'marginTop': '20px'}),
                            html.Ul([
                                html.Li("Excel文件中没有靓机汇相关的工作表"),
                                html.Li("工作表名称不匹配（尝试的名称: 靓机汇二手回收, 靓机汇二手机, 靓机汇, LJH）"),
                                html.Li("数据文件路径不正确")
                            ], style={'textAlign': 'left', 'maxWidth': '500px', 'margin': '0 auto'}),
                            html.P("请检查 ./processed_data/excel_data.xlsx 文件", 
                                  style={'textAlign': 'center', 'marginTop': '20px', 'fontStyle': 'italic'})
                        ], style={'padding': '40px'})
                    ])
                
                filtered_data = ljh_data.copy()
                
                if model_val != '全部':
                    filtered_data = filtered_data[filtered_data['型号'] == model_val]
                if memory_val != '全部':
                    def match_ljh_memory(mem_val):
                        try:
                            mem_str = str(mem_val).replace('纳米', '1').replace('G', '').replace('g', '').strip()
                            mem_int = int(float(mem_str))
                            return mem_int == int(memory_val)
                        except:
                            return False
                    filtered_data = filtered_data[filtered_data['内存'].apply(match_ljh_memory)]
                if sim_val != '全部':
                    filtered_data = filtered_data[filtered_data['版本'] == sim_val]
                
                return dcc.Graph(figure=create_ljh_analysis_chart(filtered_data))
        
        except Exception as e:
            print(f"图表更新错误: {e}")
            return html.Div(f"数据处理错误: {str(e)}")
        
        # 默认返回空图表
        return dcc.Graph(figure=go.Figure())

    def create_color_inventory_figure(filtered_df):
        try:
            required_columns = ['model', 'grade_name', 'battery', 'storage', 'sim_type', 'local', 'color', 'date', 'date_add_to_bag']
            missing_columns = [col for col in required_columns if col not in filtered_df.columns]
            if missing_columns:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"数据缺少必需的列: {', '.join(missing_columns)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=20)
                )
                return fig
            if filtered_df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="处理后数据为空",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=20)
                )
                return fig
            df_copy = filtered_df.copy()
            filter_columns = ['model', 'grade_name', 'battery', 'storage', 'sim_type', 'local']
            for col in filter_columns:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].astype(str)
            df_copy['price'] = pd.to_numeric(df_copy['price'], errors='coerce')
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
            df_copy = df_copy.dropna(subset=['date', 'price'])
            if df_copy.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="处理后数据为空",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=20)
                )
                return fig
            # ===== K线图数据处理（第一个子图）=====
            daily_price_data = []
            date_list = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in df_copy['date'].dt.date.unique()]
            date_ticktext = [pd.to_datetime(date).strftime('%m-%d') for date in df_copy['date'].dt.date.unique()]
            for date in df_copy['date'].dt.date.unique():
                date_data = df_copy[df_copy['date'].dt.date == date].copy()
                if len(date_data) == 0:
                    continue
                color_groups = date_data.groupby('color')
                first_prices = []
                last_prices = []
                all_prices = []
                for color, group in color_groups:
                    group_sorted = group.sort_values('date_add_to_bag')
                    group_prices = group_sorted['price'].dropna()
                    if len(group_prices) > 0:
                        first_prices.append(group_prices.iloc[0])
                        last_prices.append(group_prices.iloc[-1])
                        all_prices.extend(group_prices.tolist())
                if len(first_prices) == 0 or len(last_prices) == 0:
                    continue
                first_median = pd.Series(first_prices).median()
                second_median = pd.Series(last_prices).median()
                daily_price_data.append({
                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                    'first_median': first_median,
                    'second_median': second_median,
                    'open': first_median,
                    'close': second_median,
                    'high': max(all_prices) if all_prices else first_median,
                    'low': min(all_prices) if all_prices else first_median
                })
            daily_df = pd.DataFrame(daily_price_data)
            # ===== 库存数据处理（第二、第三、第四子图）=====
            df_copy['unique_key'] = (
                df_copy['model'].astype(str) + '_' + df_copy['grade_name'].astype(str) + '_' + df_copy['battery'].astype(str) + '_' + df_copy['storage'].astype(str) + '_' + df_copy['sim_type'].astype(str) + '_' + df_copy['local'].astype(str) + '_' + df_copy['color'].astype(str)
            )
            daily_color_inventory = []
            daily_total_inventory = []
            for date in df_copy['date'].dt.date.unique():
                date_data = df_copy[df_copy['date'].dt.date == date].copy()
                if len(date_data) == 0:
                    continue
                grouped = date_data.groupby('unique_key')
                daily_first_color = {}
                daily_first_sellers = {}
                daily_first_prices = {}
                daily_first_total = 0
                
                # 新增：最后一次出现的数据
                daily_last_color = {}
                daily_last_sellers = {}
                daily_last_prices = {}
                daily_last_total = 0
                
                for unique_key, group in grouped:
                    group = group.sort_values('date_add_to_bag')
                    color = group['color'].iloc[0]
                    
                    # 第一次出现的数据
                    first_seller = group['seller'].iloc[0] if 'seller' in group.columns else '未知'
                    first_price = group['price'].iloc[0] if 'price' in group.columns else 0
                    if 'quantity' in group.columns:
                        first_inventory = pd.to_numeric(group['quantity'].iloc[0], errors='coerce')
                        if pd.isna(first_inventory):
                            first_inventory = 1
                    else:
                        first_inventory = 1
                    daily_first_color[color] = daily_first_color.get(color, 0) + first_inventory
                    if color not in daily_first_sellers:
                        daily_first_sellers[color] = []
                    daily_first_sellers[color].append(first_seller)
                    if color not in daily_first_prices:
                        daily_first_prices[color] = []
                    daily_first_prices[color].append(first_price)
                    daily_first_total += first_inventory
                    
                    # 最后一次出现的数据
                    last_seller = group['seller'].iloc[-1] if 'seller' in group.columns else '未知'
                    last_price = group['price'].iloc[-1] if 'price' in group.columns else 0
                    if 'quantity' in group.columns:
                        last_inventory = pd.to_numeric(group['quantity'].iloc[-1], errors='coerce')
                        if pd.isna(last_inventory):
                            last_inventory = 1
                    else:
                        last_inventory = 1
                    daily_last_color[color] = daily_last_color.get(color, 0) + last_inventory
                    if color not in daily_last_sellers:
                        daily_last_sellers[color] = []
                    daily_last_sellers[color].append(last_seller)
                    if color not in daily_last_prices:
                        daily_last_prices[color] = []
                    daily_last_prices[color].append(last_price)
                    daily_last_total += last_inventory
                
                # 处理第一次出现的数据
                for color in daily_first_color.keys():
                    first_sellers_list = list(set(daily_first_sellers.get(color, [])))
                    first_prices_list = daily_first_prices.get(color, [])
                    first_avg_price = sum(first_prices_list) / len(first_prices_list) if first_prices_list else 0
                    daily_color_inventory.append({
                        'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                        'color': color,
                        'first_inventory': daily_first_color.get(color, 0),
                        'first_sellers': ', '.join(first_sellers_list),
                        'first_avg_price': first_avg_price,
                        'last_inventory': daily_last_color.get(color, 0),
                        'last_sellers': ', '.join(list(set(daily_last_sellers.get(color, [])))),
                        'last_avg_price': sum(daily_last_prices.get(color, [])) / len(daily_last_prices.get(color, [])) if daily_last_prices.get(color, []) else 0
                    })
                
                daily_total_inventory.append({
                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                    'first_total': daily_first_total,
                    'last_total': daily_last_total
                })
            
            color_inventory_df = pd.DataFrame(daily_color_inventory)
            total_inventory_df = pd.DataFrame(daily_total_inventory)
            
            if not color_inventory_df.empty:
                first_color_df = color_inventory_df[['date', 'color', 'first_inventory', 'first_sellers', 'first_avg_price']].rename(columns={'first_inventory': 'inventory', 'first_sellers': 'sellers', 'first_avg_price': 'price'})
                last_color_df = color_inventory_df[['date', 'color', 'last_inventory', 'last_sellers', 'last_avg_price']].rename(columns={'last_inventory': 'inventory', 'last_sellers': 'sellers', 'last_avg_price': 'price'})
            else:
                first_color_df = pd.DataFrame()
                last_color_df = pd.DataFrame()
            
            if not total_inventory_df.empty:
                first_total_df = total_inventory_df[['date', 'first_total']].rename(columns={'first_total': 'total_inventory'})
            else:
                first_total_df = pd.DataFrame()
            
            # ===== 创建四子图布局 =====
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=False,
                vertical_spacing=0.06,
                subplot_titles=('价格K线图',
                               '每天每个颜色的库存数量（第一次出现）',
                               '每天每个颜色的库存数量（最后一次出现）',
                               '每天第一次出现的总库存（不区分颜色）'),
                row_heights=[0.25, 0.25, 0.25, 0.25]
            )
            
            # 第一图：K线图
            if not daily_df.empty:
                fig.add_trace(
                    go.Candlestick(
                        x=daily_df['date'],
                        open=daily_df['open'],
                        high=daily_df['high'],
                        low=daily_df['low'],
                        close=daily_df['close'],
                        name='价格',
                        text=[f"日期: {date}<br>第一次中位数: {first:.2f}€<br>第二次中位数: {second:.2f}€<br>最高价: {high:.2f}€<br>最低价: {low:.2f}€"
                              if not pd.isna(first) else f"日期: {date}<br>无数据"
                              for date, first, second, high, low in zip(
                                  daily_df['date'], daily_df['first_median'],
                                  daily_df['second_median'], daily_df['high'], daily_df['low']
                              )],
                        hoverinfo='text'
                    ),
                    row=1, col=1
                )
            
            # 第二图：每天每个颜色的库存数量（第一次出现）
            if not first_color_df.empty:
                colors_list = sorted(first_color_df['color'].unique())
                for color in colors_list:
                    color_data = first_color_df[first_color_df['color'] == color]
                    if not color_data.empty:
                        custom_data = []
                        for _, row in color_data.iterrows():
                            sellers = row['sellers']
                            price = row['price']
                            custom_data.append([sellers, price])
                        # 转换为 'rgb(r,g,b)' 字符串
                        rgb = color_to_rgb.get(color, (128,128,128))
                        rgb_str = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
                        fig.add_trace(
                            go.Bar(
                                x=color_data['date'],
                                y=color_data['inventory'],
                                name=f'{color}（第一次）',
                                marker_color=rgb_str,
                                hovertemplate=f'颜色: {color}<br>日期: %{{x}}<br>库存数量: %{{y}}<br>价格: %{{customdata[1]:.2f}}€<br>商家: %{{customdata[0]}}<extra></extra>',
                                customdata=custom_data,
                                offsetgroup=color,
                                legendgroup=f'first_{color}'
                            ),
                            row=2, col=1
                        )
            
            # 第三图：每天每个颜色的库存数量（最后一次出现）
            if not last_color_df.empty:
                colors_list = sorted(last_color_df['color'].unique())
                for color in colors_list:
                    color_data = last_color_df[last_color_df['color'] == color]
                    if not color_data.empty:
                        custom_data = []
                        for _, row in color_data.iterrows():
                            sellers = row['sellers']
                            price = row['price']
                            custom_data.append([sellers, price])
                        # 转换为 'rgb(r,g,b)' 字符串，但稍微调暗以区分
                        rgb = color_to_rgb.get(color, (128,128,128))
                        # 将颜色调暗30%以区分第一次和最后一次
                        rgb_dark = tuple(int(c * 0.7) for c in rgb)
                        rgb_str = f'rgb({rgb_dark[0]},{rgb_dark[1]},{rgb_dark[2]})'
                        fig.add_trace(
                            go.Bar(
                                x=color_data['date'],
                                y=color_data['inventory'],
                                name=f'{color}（最后一次）',
                                marker_color=rgb_str,
                                hovertemplate=f'颜色: {color}<br>日期: %{{x}}<br>库存数量: %{{y}}<br>价格: %{{customdata[1]:.2f}}€<br>商家: %{{customdata[0]}}<extra></extra>',
                                customdata=custom_data,
                                offsetgroup=color,
                                legendgroup=f'last_{color}'
                            ),
                            row=3, col=1
                        )
            
            # 第四图：每天第一次出现的总库存
            if not first_total_df.empty:
                fig.add_trace(
                    go.Bar(
                        x=first_total_df['date'],
                        y=first_total_df['total_inventory'],
                        name='总库存',
                        marker_color='skyblue',
                        hovertemplate='日期: %{x}<br>总库存数量: %{y}<extra></extra>'
                    ),
                    row=4, col=1
                )
            
            fig.update_layout(
                title=f'每日颜色库存分析（增强版：包含第一次和最后一次出现数据）',
                height=1600,
                width=1200,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                barmode='group',
                xaxis_tickangle=45,   # 第一子图x轴 45°
                xaxis2_tickangle=45,  # 第二子图x轴 45°
                xaxis3_tickangle=45,  # 第三子图x轴 45°
                xaxis4_tickangle=45   # 第四子图x轴 45°
            )
            
            fig.update_layout(
                xaxis=dict(
                    type='category',
                    tickmode='array',
                    tickvals=date_list,
                    ticktext=date_ticktext,
                    categoryorder='category ascending'
                ),
                xaxis2=dict(
                    type='category',
                    tickmode='array',
                    tickvals=date_list,
                    ticktext=date_ticktext,
                    categoryorder='category ascending'
                ),
                xaxis3=dict(
                    type='category',
                    tickmode='array',
                    tickvals=date_list,
                    ticktext=date_ticktext,
                    categoryorder='category ascending'
                ),
                xaxis4=dict(
                    type='category',
                    tickmode='array',
                    tickvals=date_list,
                    ticktext=date_ticktext,
                    categoryorder='category ascending'
                )
            )
            
            return fig
        except Exception as e:
            print(f"创建图表时出错: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"数据处理错误: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
    
    def create_xq_analysis_chart(filtered_data):
        """创建讯强全新机箱型图分析"""
        if filtered_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="筛选后数据为空",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 确保日期列格式正确
        filtered_data['日期'] = pd.to_datetime(filtered_data['日期'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['日期'])
        
        # 获取价格列
        price_column = None
        for col in ['价格', 'price', '售价', '报价']:
            if col in filtered_data.columns:
                price_column = col
                break
        
        if price_column is None:
            print(f"未找到价格列，可用列名: {list(filtered_data.columns)}")
            fig = go.Figure()
            fig.add_annotation(
                text="未找到价格列",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 确保价格列是数值类型
        filtered_data[price_column] = pd.to_numeric(filtered_data[price_column], errors='coerce')
        valid_data = filtered_data.dropna(subset=[price_column])
        
        print(f"使用价格列: {price_column}, 有效数据行数: {len(valid_data)}")
        
        if valid_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="无有效价格数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 创建图表
        fig = go.Figure()
        
        # 按日期分组创建箱型图
        date_list = sorted(valid_data['日期'].dt.strftime('%m-%d').unique())
        
        for date_str in date_list:
            date_data = valid_data[valid_data['日期'].dt.strftime('%m-%d') == date_str]
            if len(date_data) > 0:
                fig.add_trace(go.Box(
                    y=date_data[price_column],
                    name=date_str,
                    boxpoints='outliers',
                    hovertemplate=(
                        "日期: " + date_str + "<br>"
                        "价格: ¥%{y:.0f}<br>"
                        "<extra></extra>"
                    )
                ))
        
        fig.update_layout(
            title='讯强全新机价格分布箱型图',
            xaxis_title='日期',
            yaxis_title='价格 (¥)',
            height=400,
            showlegend=False,
            xaxis=dict(
                tickangle=45,
                type='category'
            ),
            yaxis=dict(
                tickformat=',.0f'
            )
        )
        
        return fig
    
    def create_xq_color_price_chart(filtered_data):
        """创建讯强全新机各颜色价格趋势图"""
        if filtered_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="筛选后数据为空",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 确保日期列格式正确
        filtered_data['日期'] = pd.to_datetime(filtered_data['日期'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['日期'])
        
        # 获取价格列
        price_column = None
        for col in ['价格', 'price', '售价', '报价']:
            if col in filtered_data.columns:
                price_column = col
                break
        
        # 获取颜色列
        color_column = None
        for col in ['颜色', 'color', '色彩']:
            if col in filtered_data.columns:
                color_column = col
                break
        
        if price_column is None or color_column is None:
            print(f"未找到必要列，价格列: {price_column}, 颜色列: {color_column}")
            print(f"可用列名: {list(filtered_data.columns)}")
            fig = go.Figure()
            fig.add_annotation(
                text="未找到价格或颜色列",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 确保价格列是数值类型
        filtered_data[price_column] = pd.to_numeric(filtered_data[price_column], errors='coerce')
        valid_data = filtered_data.dropna(subset=[price_column])
        
        print(f"使用价格列: {price_column}, 颜色列: {color_column}, 有效数据行数: {len(valid_data)}")
        
        if valid_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="无有效价格数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 创建图表
        fig = go.Figure()
        
        # 使用颜色库存分析的RGB配色方案
        def get_color_for_name(color_name):
            """根据颜色名称获取对应的RGB颜色"""
            # 清理颜色名称，去除空格和特殊字符
            clean_name = str(color_name).strip()
            
            # 尝试直接匹配
            if clean_name in color_to_rgb:
                rgb = color_to_rgb[clean_name]
                print(f"直接匹配颜色: {clean_name} -> RGB{rgb}")
                return f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
            
            # 尝试模糊匹配（包含关系）
            for color_key in color_to_rgb.keys():
                if clean_name in color_key or color_key in clean_name:
                    rgb = color_to_rgb[color_key]
                    print(f"模糊匹配颜色: {clean_name} -> {color_key} -> RGB{rgb}")
                    return f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
            
            # 如果都没匹配到，使用默认灰色
            print(f"未匹配到颜色: {clean_name}，使用默认灰色")
            return 'rgb(128, 128, 128)'
        
        # 获取第四列的所有唯一值作为分组依据
        unique_colors = valid_data[color_column].unique()
        print(f"数据中的颜色: {list(unique_colors)}")
        print(f"可用的RGB颜色映射: {list(color_to_rgb.keys())}")
        
        # 为每个颜色创建一条线
        for i, color_name in enumerate(unique_colors):
            # 获取该颜色的所有数据
            color_data = valid_data[valid_data[color_column] == color_name].copy()
            
            if len(color_data) > 0:
                # 按日期排序
                color_data = color_data.sort_values('日期')
                
                # 转换日期为字符串格式
                date_strings = color_data['日期'].dt.strftime('%m-%d').tolist()
                
                # 获取颜色
                line_color = get_color_for_name(color_name)
                
                # 添加线条
                fig.add_trace(go.Scatter(
                    x=date_strings,
                    y=color_data[price_column],
                    mode='lines+markers',
                    name=str(color_name),
                    line=dict(color=line_color, width=2),
                    marker=dict(size=4),
                    connectgaps=True,
                    hovertemplate=(
                        f"{color_column}: %{{fullData.name}}<br>"
                        "日期: %{x}<br>"
                        "价格: ¥%{y:.0f}<br>"
                        "<extra></extra>"
                    )
                ))
        
        fig.update_layout(
            title='讯强全新机各颜色价格趋势',
            xaxis_title='日期',
            yaxis_title='价格 (¥)',
            height=400,
            hovermode='x unified',
            xaxis=dict(
                type='category',
                categoryorder='category ascending',
                tickangle=45
            ),
            yaxis=dict(
                tickformat=',.0f'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_ljh_analysis_chart(filtered_data):
        """创建靓机汇二手回收分析图表（第5-9列折线图）"""
        if filtered_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="筛选后数据为空",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 确保日期列格式正确
        filtered_data['日期'] = pd.to_datetime(filtered_data['日期'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['日期'])
        
        # 获取第5-9列
        line_columns = filtered_data.columns[4:9]  # 第5-9列
        
        # 创建图表
        fig = go.Figure()
        
        # 收集所有数值用于计算y轴范围
        all_values = []
        
        # 只保留有数据的日期，且格式为月-日
        date_list_unique = sorted(filtered_data['日期'].dt.strftime('%m-%d').unique())
        
        # 为每一列创建一条线
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, col in enumerate(line_columns):
            if col in filtered_data.columns:
                # 按日期分组，计算每日平均值
                daily_avg = []
                date_list = []
                for date_str in date_list_unique:
                    date_data = filtered_data[filtered_data['日期'].dt.strftime('%m-%d') == date_str]
                    if len(date_data) > 0:
                        avg_val = pd.to_numeric(date_data[col], errors='coerce').mean()
                        if pd.notna(avg_val):
                            daily_avg.append(avg_val)
                            date_list.append(date_str)
                            all_values.append(avg_val)
                        else:
                            daily_avg.append(None)
                            date_list.append(date_str)
                    else:
                        daily_avg.append(None)
                        date_list.append(date_str)
                fig.add_trace(go.Scatter(
                    x=date_list,
                    y=daily_avg,
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors[i % len(colors)]),
                    connectgaps=True,  # 连接间隙
                    hovertemplate=(
                        "日期: %{x}<br>"
                        "指标: " + col + "<br>"
                        "数值: %{y}<br>"
                        "<extra></extra>"
                    )
                ))
        
        # 计算价格范围和y轴刻度
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            val_range = max_val - min_val
            
            # 设置合适的间隔，确保整数显示
            if val_range <= 100:
                tick_interval = 10
            elif val_range <= 500:
                tick_interval = 50
            elif val_range <= 1000:
                tick_interval = 100
            elif val_range <= 5000:
                tick_interval = 500
            else:
                tick_interval = 1000
            
            # 计算y轴范围
            y_min = int(min_val // tick_interval) * tick_interval
            y_max = int((max_val // tick_interval) + 1) * tick_interval
            
            # 生成刻度值
            tick_vals = list(range(y_min, y_max + tick_interval, tick_interval))
        else:
            tick_vals = None
            y_min = None
            y_max = None
        
        fig.update_layout(
            title='靓机汇二手回收分析（第5-9列折线图）',
            xaxis_title='日期',
            yaxis_title='价格',
            height=400,
            showlegend=True,
            xaxis=dict(
                    type='category',
                    categoryorder='category ascending'
                ),
            yaxis=dict(
                tickvals=tick_vals,
                ticktext=[str(int(val)) for val in tick_vals] if tick_vals else None,
                range=[y_min, y_max] if y_min is not None and y_max is not None else None
            )
        )
        
        return fig
    
    return app

processed_df = load_processed_data()
app = create_color_inventory_app(processed_df)
server = app.server if app is not None else None

if __name__ == '__main__':
    if app is not None:
        port = int(os.environ.get('PORT', 8051))  # 使用不同端口避免冲突
        app.run(host='0.0.0.0', port=port, debug=False)
        print(f"\n每日颜色库存分析应用（增强版）已启动！请在浏览器中访问: http://0.0.0.0:{port}")
    else:
        print("应用创建失败，请检查数据")