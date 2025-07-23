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
    "玫瑰色": (255, 102, 204),
    "粉红色": (255, 192, 203),
    "蓝色": (0, 0, 255),
    "银色": (192, 192, 192),
    "黄色": (255, 255, 0),
    "太空灰": (105, 105, 105),
    "星光色": (255, 250, 205),
    "紫色": (128, 0, 128),
    "深空黑": (25, 25, 25),
    "白色": (255, 255, 255),
    "群青蓝": (65, 105, 225),
    "蓝绿色": (0, 255, 255),
    "黑色": (0, 0, 0),
    "钛金属原色": (166, 166, 166),
    "钛金属沙色": (194, 178, 128),
    "钛金属白": (245, 245, 245),
    "钛金属黑": (50, 50, 50)
}

def load_processed_data():
    """加载处理好的原始数据文件"""
    try:
        df = pd.read_excel("./processed_data/original_data.xlsx")
        return df
    except Exception as e:
        print(f"读取原始数据失败: {e}")
        return pd.DataFrame()

def create_color_inventory_app(df):
    """
    创建每日颜色库存分析的Dash应用，包含6个筛选器。
    """
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
        html.H1("每日颜色库存分析", style={'textAlign': 'center', 'marginBottom': 30}),
        
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
            
            html.Div([
                html.Label("地区:"),
                dcc.Dropdown(
                    id='local-dropdown', 
                    options=local_options, 
                    value='全部', 
                    style={'width': '180px'}
                )
            ], style={'display': 'inline-block'}),
        ], style={'textAlign': 'center', 'marginBottom': 20}),
        
        html.Div(id='chart-content')
    ])

    @app.callback(
        Output('chart-content', 'children'),
        [
            Input('model-dropdown', 'value'),
            Input('memory-dropdown', 'value'),
            Input('sim-dropdown', 'value'),
            Input('grade-dropdown', 'value'),
            Input('battery-dropdown', 'value'),
            Input('local-dropdown', 'value')
        ]
    )
    def update_color_inventory_chart(model_val, memory_val, sim_val, grade_val, battery_val, local_val):
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
                    text="筛选后数据为空",
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
            # ===== 库存数据处理（第二、第四子图）=====
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
                for unique_key, group in grouped:
                    group = group.sort_values('date_add_to_bag')
                    color = group['color'].iloc[0]
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
                for color in daily_first_color.keys():
                    first_sellers_list = list(set(daily_first_sellers.get(color, [])))
                    first_prices_list = daily_first_prices.get(color, [])
                    first_avg_price = sum(first_prices_list) / len(first_prices_list) if first_prices_list else 0
                    daily_color_inventory.append({
                        'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                        'color': color,
                        'first_inventory': daily_first_color.get(color, 0),
                        'first_sellers': ', '.join(first_sellers_list),
                        'first_avg_price': first_avg_price
                    })
                daily_total_inventory.append({
                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                    'first_total': daily_first_total
                })
            color_inventory_df = pd.DataFrame(daily_color_inventory)
            total_inventory_df = pd.DataFrame(daily_total_inventory)
            if not color_inventory_df.empty:
                first_color_df = color_inventory_df[['date', 'color', 'first_inventory', 'first_sellers', 'first_avg_price']].rename(columns={'first_inventory': 'inventory', 'first_sellers': 'sellers', 'first_avg_price': 'price'})
            else:
                first_color_df = pd.DataFrame()
            if not total_inventory_df.empty:
                first_total_df = total_inventory_df[['date', 'first_total']].rename(columns={'first_total': 'total_inventory'})
            else:
                first_total_df = pd.DataFrame()
            # ===== 创建三子图布局 =====
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=False,
                vertical_spacing=0.08,
                subplot_titles=('价格K线图',
                               '每天每个颜色的库存数量（第一次出现）',
                               '每天第一次出现的总库存（不区分颜色）'),
                row_heights=[0.35, 0.35, 0.3]
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
                                name=f'{color}',
                                marker_color=rgb_str,  # ✅ 用自定义RGB
                                hovertemplate=f'颜色: {color}<br>日期: %{{x}}<br>库存数量: %{{y}}<br>价格: %{{customdata[1]:.2f}}€<br>商家: %{{customdata[0]}}<extra></extra>',
                                customdata=custom_data,
                                offsetgroup=color,
                                legendgroup=f'first_{color}'
                            ),
                            row=2, col=1
                        )
            # 第三图：每天第一次出现的总库存
            if not first_total_df.empty:
                fig.add_trace(
                    go.Bar(
                        x=first_total_df['date'],
                        y=first_total_df['total_inventory'],
                        name='总库存',
                        marker_color='skyblue',
                        hovertemplate='日期: %{x}<br>总库存数量: %{y}<extra></extra>'
                    ),
                    row=3, col=1
                )
            fig.update_layout(
                title=f'每日颜色库存分析（只显示第一次数据）',
                height=1200,
                width=1200,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                barmode='group',
                xaxis_tickangle=45,   # 第一子图x轴 45°
                xaxis2_tickangle=45,  # 第二子图x轴 45°
                xaxis3_tickangle=45   # 第三子图x轴 45°
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
    return app

processed_df = load_processed_data()
app = create_color_inventory_app(processed_df)
server = app.server if app is not None else None

if __name__ == '__main__':
    if app is not None:
        port = int(os.environ.get('PORT', 8050))
        app.run(host='0.0.0.0', port=port, debug=False)
        print(f"\n每日颜色库存分析应用已启动！请在浏览器中访问: http://0.0.0.0:{port}")
    else:
        print("应用创建失败，请检查数据")