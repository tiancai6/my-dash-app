import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
from prepare import prepare
import os

def create_integrated_dash_app(df):
    """
    创建集成的Plotly Dash交互式网页应用
    包含价格分析和每日颜色库存分析两个功能
    """
    # 数据预处理
    df = df.copy()
    
    # 将所有筛选字段转换为字符串，避免类型混合问题
    filter_columns = ['model', 'grade_name', 'battery', 'storage', 'sim_type', 'local']
    for col in filter_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # 确保price是数值类型
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # 确保date列存在且格式正确
    if 'date' not in df.columns:
        df['date'] = df['date_add_to_bag'].dt.date
    
    # 移除空值
    df = df.dropna(subset=['price'])
    
    if df.empty:
        print("数据为空，无法创建应用")
        return None
    
    # 获取筛选选项
    model_options = [{'label': '全部', 'value': '全部'}] + [{'label': model, 'value': model} for model in sorted(df['model'].unique())]
    grade_options = [{'label': '全部', 'value': '全部'}] + [{'label': grade, 'value': grade} for grade in sorted(df['grade_name'].unique())]
    battery_options = [{'label': '全部', 'value': '全部'}] + [{'label': battery, 'value': battery} for battery in sorted(df['battery'].unique())]
    storage_options = [{'label': '全部', 'value': '全部'}] + [{'label': storage, 'value': storage} for storage in sorted(df['storage'].unique())]
    sim_options = [{'label': '全部', 'value': '全部'}] + [{'label': sim, 'value': sim} for sim in sorted(df['sim_type'].unique())]
    local_options = [{'label': '全部', 'value': '全部'}] + [{'label': local, 'value': local} for local in sorted(df['local'].unique())]
    
    # 创建Dash应用
    app = dash.Dash(__name__)
    
    # 定义应用布局
    app.layout = html.Div([
        html.H1("交互式价格和库存分析系统", style={'textAlign': 'center', 'marginBottom': 30}),
        
        # 标签页选择器
        dcc.Tabs(id="tabs", value='price-analysis', children=[
            dcc.Tab(label='价格K线分析', value='price-analysis'),
            dcc.Tab(label='每日颜色库存分析', value='daily-color-analysis'),
        ], style={'marginBottom': 20}),
        
        # 筛选控件区域
        html.Div([
            html.Div([
                html.Label("型号:"),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=model_options,
                    value='全部',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("磨损:"),
                dcc.Dropdown(
                    id='grade-dropdown',
                    options=grade_options,
                    value='全部',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("电池:"),
                dcc.Dropdown(
                    id='battery-dropdown',
                    options=battery_options,
                    value='全部',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
        ], style={'marginBottom': 20, 'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label("容量:"),
                dcc.Dropdown(
                    id='storage-dropdown',
                    options=storage_options,
                    value='全部',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("SIM类型:"),
                dcc.Dropdown(
                    id='sim-dropdown',
                    options=sim_options,
                    value='全部',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("地区:"),
                dcc.Dropdown(
                    id='local-dropdown',
                    options=local_options,
                    value='全部',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
        ], style={'marginBottom': 30, 'textAlign': 'center'}),
        
        # 图表显示区域
        html.Div(id='chart-content')
    ])
    
    # 主回调函数 - 根据标签页切换图表内容
    @app.callback(
        Output('chart-content', 'children'),
        [
            Input('tabs', 'value'),
            Input('model-dropdown', 'value'),
            Input('grade-dropdown', 'value'),
            Input('battery-dropdown', 'value'),
            Input('storage-dropdown', 'value'),
            Input('sim-dropdown', 'value'),
            Input('local-dropdown', 'value')
        ]
    )
    def update_chart_content(active_tab, model_val, grade_val, battery_val, storage_val, sim_val, local_val):
        # 根据筛选条件过滤数据
        filtered_df = df.copy()
        
        if model_val != '全部':
            filtered_df = filtered_df[filtered_df['model'] == model_val]
        if grade_val != '全部':
            filtered_df = filtered_df[filtered_df['grade_name'] == grade_val]
        if battery_val != '全部':
            filtered_df = filtered_df[filtered_df['battery'] == battery_val]
        if storage_val != '全部':
            filtered_df = filtered_df[filtered_df['storage'] == storage_val]
        if sim_val != '全部':
            filtered_df = filtered_df[filtered_df['sim_type'] == sim_val]
        if local_val != '全部':
            filtered_df = filtered_df[filtered_df['local'] == local_val]
        
        if filtered_df.empty:
            # 返回空图表
            fig = go.Figure()
            fig.add_annotation(
                text="筛选后数据为空",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return dcc.Graph(figure=fig, style={'height': '800px'})
        
        if active_tab == 'price-analysis':
            # 价格K线分析图表
            fig = create_price_analysis_chart(filtered_df)
            return dcc.Graph(figure=fig, style={'height': '1200px'})
        
        elif active_tab == 'daily-color-analysis':
            # 每日颜色库存分析图表
            fig = create_daily_color_analysis_chart(filtered_df)
            return dcc.Graph(figure=fig, style={'height': '1500px'})
    
    def create_price_analysis_chart(filtered_df):
        """创建价格K线分析图表"""
        # 数据预处理
        df_copy = filtered_df.copy()
        df_copy['price'] = pd.to_numeric(df_copy['price'], errors='coerce')
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        df_copy = df_copy.dropna(subset=['date', 'price'])
        
        # 按日期分组，计算每天的K线数据（修改为按颜色分组）
        daily_stats = []
        
        for date in sorted(df_copy['date'].dt.date.unique()):
            date_data = df_copy[df_copy['date'].dt.date == date].copy()
            
            if len(date_data) == 0:
                continue
            
            # 检查是否有颜色列
            if 'color' not in date_data.columns:
                # 如果没有颜色列，使用原来的逻辑
                date_data = date_data.sort_values('date_add_to_bag')
                prices = date_data['price'].dropna()
                
                if len(prices) == 0:
                    continue
                elif len(prices) == 1:
                    price = prices.iloc[0]
                    daily_stats.append({
                        'date': pd.to_datetime(date),
                        'open': price,
                        'close': price,
                        'high': price,
                        'low': price
                    })
                else:
                    first_half = prices[:len(prices)//2]
                    second_half = prices[len(prices)//2:]
                    
                    first_median = first_half.median()
                    second_median = second_half.median()
                    
                    daily_stats.append({
                        'date': pd.to_datetime(date),
                        'open': first_median,
                        'close': second_median,
                        'high': prices.max(),
                        'low': prices.min()
                    })
            else:
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
                last_median = pd.Series(last_prices).median()    # 所有颜色最后一次价格的中位数
                
                daily_stats.append({
                    'date': pd.to_datetime(date),
                    'open': first_median,
                    'close': last_median,
                    'high': max(all_prices) if all_prices else first_median,
                    'low': min(all_prices) if all_prices else first_median
                })
        
        if not daily_stats:
            fig = go.Figure()
            fig.add_annotation(
                text="没有足够的数据生成图表",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        daily_df = pd.DataFrame(daily_stats)
        daily_df = daily_df.sort_values('date')
        
        # 计算库存数据
        first_inventory_data, last_inventory_data = calculate_inventory_data(filtered_df)
        
        # 创建三子图布局
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.15,
            subplot_titles=('价格K线图', '每天第一次库存数量', '每天最后一次库存数量'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=daily_df['date'],
                open=daily_df['open'],
                high=daily_df['high'],
                low=daily_df['low'],
                close=daily_df['close'],
                name='价格K线',
                text=[f"日期: {date}<br>开盘: {open_p:.2f}€<br>收盘: {close_p:.2f}€<br>最高: {high_p:.2f}€<br>最低: {low_p:.2f}€" 
                      for date, open_p, close_p, high_p, low_p in zip(
                          daily_df['date'], daily_df['open'], 
                          daily_df['close'], daily_df['high'], daily_df['low']
                      )],
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # 添加库存图表
        add_inventory_charts(fig, first_inventory_data, last_inventory_data)
        
        # 更新布局
        fig.update_layout(
            title='价格K线分析图表',
            height=1200,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        # 更新坐标轴
        fig.update_xaxes(row=1, col=1, tickangle=0)
        fig.update_xaxes(row=2, col=1, tickangle=45)
        fig.update_xaxes(row=3, col=1, tickangle=45)
        
        fig.update_yaxes(title_text="价格 (€)", row=1, col=1)
        fig.update_yaxes(title_text="库存数量", row=2, col=1)
        fig.update_yaxes(title_text="库存数量", row=3, col=1)
        
        return fig
    
    def create_daily_color_analysis_chart(filtered_df):
        """创建每日颜色库存分析图表"""
        try:
            # 检查必需的列是否存在
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
            
            # 检查数据是否为空
            if filtered_df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="筛选后数据为空",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=20)
                )
                return fig
            
            # 数据预处理
            df_copy = filtered_df.copy()
            
            # 确保筛选列的数据类型正确
            filter_columns = ['model', 'grade_name', 'battery', 'storage', 'sim_type', 'local']
            for col in filter_columns:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].astype(str)
            
            # 确保价格列是数值类型
            df_copy['price'] = pd.to_numeric(df_copy['price'], errors='coerce')
            
            # 处理日期列
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
            
            # 移除NaN值
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
            
            # ===== K线图数据处理（修改为按颜色分组）===== 
            daily_price_data = [] 
            
            for date in sorted(df_copy['date'].dt.date.unique()): 
                date_data = df_copy[df_copy['date'].dt.date == date].copy() 
                
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
            
            daily_df = pd.DataFrame(daily_price_data)
            
            # ===== 库存数据处理（修改为包含seller信息）=====
            # 创建unique_key
            df_copy['unique_key'] = (
                df_copy['model'].astype(str) + '_' +
                df_copy['grade_name'].astype(str) + '_' +
                df_copy['battery'].astype(str) + '_' +
                df_copy['storage'].astype(str) + '_' +
                df_copy['sim_type'].astype(str) + '_' +
                df_copy['local'].astype(str) + '_' +
                df_copy['color'].astype(str)
            )
            
            # 按日期分组处理库存数据
            daily_color_inventory = []
            daily_total_inventory = []
            
            for date in sorted(df_copy['date'].dt.date.unique()):
                date_data = df_copy[df_copy['date'].dt.date == date].copy()
                
                if len(date_data) == 0:
                    continue
                
                # 按unique_key分组
                grouped = date_data.groupby('unique_key')
                
                daily_first_color = {}
                daily_last_color = {}
                daily_first_sellers = {}  
                daily_last_sellers = {}   
                daily_first_prices = {}   # 新增：记录第一次出现的价格信息
                daily_last_prices = {}    # 新增：记录最后一次出现的价格信息
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
                        'first_avg_price': first_avg_price,  # 新增：第一次平均价格
                        'last_avg_price': last_avg_price     # 新增：最后一次平均价格
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
            
            # 分离第一次和最后一次数据
            if not color_inventory_df.empty:
                first_color_df = color_inventory_df[['date', 'color', 'first_inventory', 'first_sellers', 'first_avg_price']].rename(columns={'first_inventory': 'inventory', 'first_sellers': 'sellers', 'first_avg_price': 'price'})
                last_color_df = color_inventory_df[['date', 'color', 'last_inventory', 'last_sellers', 'last_avg_price']].rename(columns={'last_inventory': 'inventory', 'last_sellers': 'sellers', 'last_avg_price': 'price'})
            else:
                first_color_df = pd.DataFrame()
                last_color_df = pd.DataFrame()
            
            if not total_inventory_df.empty:
                first_total_df = total_inventory_df[['date', 'first_total']].rename(columns={'first_total': 'total_inventory'})
                last_total_df = total_inventory_df[['date', 'last_total']].rename(columns={'last_total': 'total_inventory'})
            else:
                first_total_df = pd.DataFrame()
                last_total_df = pd.DataFrame()
            
            # ===== 创建五子图布局 =====
            fig = make_subplots(
                rows=5, cols=1,
                shared_xaxes=False,
                vertical_spacing=0.08,
                subplot_titles=('价格K线图',
                               '每天每个颜色的库存数量（第一次出现）',
                               '每天每个颜色的库存数量（最后一次出现）',
                               '每天第一次出现的总库存（不区分颜色）',
                               '每天最后一次出现的总库存（不区分颜色）'),
                row_heights=[0.25, 0.2, 0.2, 0.175, 0.175]
            )
            
            # ===== 第一图：K线图（与create_interactive_price_chart完全一样） =====
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
            
            # ===== 第二图：每天每个颜色的库存数量（第一次出现）- 同一天的颜色靠在一起 =====
            if not first_color_df.empty:
                # 为不同颜色分配颜色
                colors_list = sorted(first_color_df['color'].unique())
                colors_map = px.colors.qualitative.Set3[:len(colors_list)]
                if len(colors_list) > len(colors_map):
                    colors_map = colors_map * (len(colors_list) // len(colors_map) + 1)
                color_mapping = dict(zip(colors_list, colors_map))
                
                # 重新组织数据，让同一天的颜色靠在一起
                for color in colors_list:
                    color_data = first_color_df[first_color_df['color'] == color]
                    if not color_data.empty:
                        # 准备customdata，包含商家和价格信息
                        custom_data = []
                        for _, row in color_data.iterrows():
                            sellers = row['sellers']
                            price = row['price']  # 直接使用price字段
                            custom_data.append([sellers, price])
                        
                        fig.add_trace(
                            go.Bar(
                                x=color_data['date'],  # 直接使用日期作为x轴
                                y=color_data['inventory'],
                                name=f'第一次-{color}',
                                marker_color=color_mapping[color],
                                hovertemplate=f'颜色: {color}<br>日期: %{{x}}<br>库存数量: %{{y}}<br>价格: %{{customdata[1]:.2f}}€<br>商家: %{{customdata[0]}}<extra></extra>',
                                customdata=custom_data,  # 传递商家和价格信息
                                offsetgroup=color,  # 使用offsetgroup让同一天的不同颜色靠在一起
                                legendgroup=f'first_{color}'
                            ),
                            row=2, col=1
                        )
            
            # ===== 第三图：每天每个颜色的库存数量（最后一次出现）- 同一天的颜色靠在一起 =====
            if not last_color_df.empty:
                # 为不同颜色分配颜色
                colors_list = sorted(last_color_df['color'].unique())
                colors_map = px.colors.qualitative.Set3[:len(colors_list)]
                if len(colors_list) > len(colors_map):
                    colors_map = colors_map * (len(colors_list) // len(colors_map) + 1)
                color_mapping = dict(zip(colors_list, colors_map))
                
                # 重新组织数据，让同一天的颜色靠在一起
                for color in colors_list:
                    color_data = last_color_df[last_color_df['color'] == color]
                    if not color_data.empty:
                        # 准备customdata，包含商家和价格信息
                        custom_data = []
                        for _, row in color_data.iterrows():
                            sellers = row['sellers']
                            price = row['price']  # 直接使用price字段
                            custom_data.append([sellers, price])
                        
                        fig.add_trace(
                            go.Bar(
                                x=color_data['date'],  # 直接使用日期作为x轴
                                y=color_data['inventory'],
                                name=f'最后一次-{color}',
                                marker_color=color_mapping[color],
                                hovertemplate=f'颜色: {color}<br>日期: %{{x}}<br>库存数量: %{{y}}<br>价格: %{{customdata[1]:.2f}}€<br>商家: %{{customdata[0]}}<extra></extra>',
                                customdata=custom_data,  # 传递商家和价格信息
                                offsetgroup=color,  # 使用offsetgroup让同一天的不同颜色靠在一起
                                legendgroup=f'last_{color}'
                            ),
                            row=3, col=1
                        )
            
            # ===== 第四图：每天第一次出现的总库存 =====
            if not first_total_df.empty:
                fig.add_trace(
                    go.Bar(
                        x=first_total_df['date'],
                        y=first_total_df['total_inventory'],
                        name='第一次总库存',
                        marker_color='skyblue',
                        hovertemplate='日期: %{x}<br>总库存数量: %{y}<extra></extra>'
                    ),
                    row=4, col=1
                )
            
            # ===== 第五图：每天最后一次出现的总库存 =====
            if not last_total_df.empty:
                fig.add_trace(
                    go.Bar(
                        x=last_total_df['date'],
                        y=last_total_df['total_inventory'],
                        name='最后一次总库存',
                        marker_color='lightcoral',
                        hovertemplate='日期: %{x}<br>总库存数量: %{y}<extra></extra>'
                    ),
                    row=5, col=1
                )
            
            # ===== 更新布局 =====
            fig.update_layout(
                title=f'交互式价格和库存分析图表（五子图版本）',
                height=1600,
                width=1400,
                showlegend=True,  # 显示图例以区分不同颜色
                xaxis_rangeslider_visible=False,
                barmode='group'  # 设置柱状图为分组模式，让同一天的不同颜色靠在一起
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
    
    def calculate_inventory_data(filtered_df):
        """计算库存数据"""
        first_inventory_data = []
        last_inventory_data = []
        
        for date in sorted(filtered_df['date'].unique()):
            date_df = filtered_df[filtered_df['date'] == date]
            
            # 按组合键分组（型号、磨损、电池、容量、SIM类型、地区、颜色）
            group_keys = ['model', 'grade_name', 'battery', 'storage', 'sim_type', 'local', 'color']
            grouped = date_df.groupby(group_keys)
            
            first_total = 0
            last_total = 0
            
            for group_key, group_data in grouped:
                group_data = group_data.sort_values('date_add_to_bag')
                
                # 第一次出现的库存
                first_inventory = group_data.iloc[0].get('storage_count', group_data.iloc[0].get('quantity', 1))
                # 最后一次出现的库存
                last_inventory = group_data.iloc[-1].get('storage_count', group_data.iloc[-1].get('quantity', 1))
                
                first_total += first_inventory
                last_total += last_inventory
            
            first_inventory_data.append({
                'date': date,
                'inventory': first_total
            })
            
            last_inventory_data.append({
                'date': date,
                'inventory': last_total
            })
        
        return first_inventory_data, last_inventory_data
    
    def add_inventory_charts(fig, first_inventory_data, last_inventory_data):
        """添加库存图表"""
        if not first_inventory_data and not last_inventory_data:
            return
        
        first_inventory_df = pd.DataFrame(first_inventory_data) if first_inventory_data else pd.DataFrame()
        last_inventory_df = pd.DataFrame(last_inventory_data) if last_inventory_data else pd.DataFrame()
        
        # 添加第一次库存柱状图
        if not first_inventory_df.empty:
            fig.add_trace(
                go.Bar(
                    x=first_inventory_df['date'],
                    y=first_inventory_df['inventory'],
                    name='第一次库存',
                    marker_color='lightblue',
                    hovertemplate='日期: %{x}<br>第一次库存: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 添加最后一次库存柱状图
        if not last_inventory_df.empty:
            fig.add_trace(
                go.Bar(
                    x=last_inventory_df['date'],
                    y=last_inventory_df['inventory'],
                    name='最后一次库存',
                    marker_color='lightcoral',
                    hovertemplate='日期: %{x}<br>最后一次库存: %{y}<extra></extra>'
                ),
                row=3, col=1
            )
    
    return app

# 模块级别定义 app 和 server
processed_df = prepare()  # 确保 prepare() 返回有效 DataFrame
app = create_integrated_dash_app(processed_df)
server = app.server if app is not None else None  # 暴露 server 给 Gunicorn

if __name__ == '__main__':
    if app is not None:
        # 修改为适合Render部署的配置
        port = int(os.environ.get('PORT', 8050))
        app.run(host='0.0.0.0', port=port, debug=False)
        print(f"\n集成应用已启动！请在浏览器中访问: http://0.0.0.0:{port}")
    else:
        print("应用创建失败，请检查数据")
