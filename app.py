import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
from prepare import prepare

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
        """创建价格K线分析图表（原create_interactive_price_chart逻辑）"""
        # 按日期和颜色分组，计算每天每种颜色的第一次和第二次价格
        daily_color_stats = []
        
        for date in sorted(filtered_df['date'].unique()):
            date_df = filtered_df[filtered_df['date'] == date]
            
            for color in date_df['color'].unique():
                color_df = date_df[date_df['color'] == color].sort_values('date_add_to_bag')
                
                if len(color_df) >= 2:
                    first_price = color_df.iloc[0]['price']
                    second_price = color_df.iloc[1]['price']
                elif len(color_df) == 1:
                    first_price = second_price = color_df.iloc[0]['price']
                else:
                    continue
                
                daily_color_stats.append({
                    'date': date,
                    'color': color,
                    'first_price': first_price,
                    'second_price': second_price
                })
        
        if not daily_color_stats:
            fig = go.Figure()
            fig.add_annotation(
                text="没有足够的数据生成图表",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        color_stats_df = pd.DataFrame(daily_color_stats)
        
        # 按日期分组，计算每天的开盘价、收盘价、最高价、最低价
        daily_stats = []
        
        for date in sorted(color_stats_df['date'].unique()):
            date_data = color_stats_df[color_stats_df['date'] == date]
            
            # 计算第一次和第二次的中位数
            first_median = date_data['first_price'].median()
            second_median = date_data['second_price'].median()
            
            # 所有价格的最高和最低
            all_prices = list(date_data['first_price']) + list(date_data['second_price'])
            high_price = max(all_prices)
            low_price = min(all_prices)
            
            daily_stats.append({
                'date': date,
                'open': first_median,
                'close': second_median,
                'high': high_price,
                'low': low_price,
                'first_median': first_median,
                'second_median': second_median
            })
        
        daily_df = pd.DataFrame(daily_stats)
        daily_df = daily_df.sort_values('date')
        
        # 处理Storage库存数据
        first_storage_data, second_storage_data = process_storage_data(filtered_df)
        
        # 创建三子图布局
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.15,
            subplot_titles=('价格K线图', '第一次中位数Storage库存', '第二次中位数Storage库存'),
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
        
        # 添加Storage库存图表
        add_storage_charts(fig, first_storage_data, second_storage_data)
        
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
        fig.update_yaxes(title_text="中位数库存", row=2, col=1)
        fig.update_yaxes(title_text="中位数库存", row=3, col=1)
        
        return fig
    
    def create_daily_color_analysis_chart(filtered_df):
        """创建每日颜色库存分析图表（原create_interactive_daily_color_chart逻辑）"""
        # 计算每天每个颜色的第一次和最后一次库存
        daily_color_inventory = []
        daily_total_inventory = []
        
        for date in sorted(filtered_df['date'].unique()):
            date_df = filtered_df[filtered_df['date'] == date]
            
            # 按组合键分组（型号、磨损、电池、容量、SIM类型、地区、颜色）
            group_keys = ['model', 'grade_name', 'battery', 'storage', 'sim_type', 'local', 'color']
            grouped = date_df.groupby(group_keys)
            
            first_total = 0
            last_total = 0
            color_first_counts = {}
            color_last_counts = {}
            
            for group_key, group_data in grouped:
                group_data = group_data.sort_values('date_add_to_bag')
                color = group_key[-1]  # 颜色是最后一个键
                
                # 第一次出现的库存
                first_inventory = group_data.iloc[0].get('storage_count', group_data.iloc[0].get('quantity', 1))
                # 最后一次出现的库存
                last_inventory = group_data.iloc[-1].get('storage_count', group_data.iloc[-1].get('quantity', 1))
                
                # 累加到颜色统计
                color_first_counts[color] = color_first_counts.get(color, 0) + first_inventory
                color_last_counts[color] = color_last_counts.get(color, 0) + last_inventory
                
                # 累加到总计
                first_total += first_inventory
                last_total += last_inventory
            
            # 记录每个颜色的库存
            for color in set(list(color_first_counts.keys()) + list(color_last_counts.keys())):
                daily_color_inventory.append({
                    'date': date,
                    'color': color,
                    'first_inventory': color_first_counts.get(color, 0),
                    'last_inventory': color_last_counts.get(color, 0)
                })
            
            # 记录总库存
            daily_total_inventory.append({
                'date': date,
                'first_total': first_total,
                'last_total': last_total
            })
        
        if not daily_color_inventory:
            fig = go.Figure()
            fig.add_annotation(
                text="没有库存数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        color_inventory_df = pd.DataFrame(daily_color_inventory)
        total_inventory_df = pd.DataFrame(daily_total_inventory)
        
        # 创建五子图布局
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=(
                '价格K线图',
                '每天每个颜色的第一次库存数量',
                '每天每个颜色的最后一次库存数量',
                '每天第一次出现的总库存',
                '每天最后一次出现的总库存'
            ),
            row_heights=[0.25, 0.2, 0.2, 0.175, 0.175]
        )
        
        # 添加K线图（简化版）
        daily_prices = []
        for date in sorted(filtered_df['date'].unique()):
            date_df = filtered_df[filtered_df['date'] == date]
            if not date_df.empty:
                daily_prices.append({
                    'date': date,
                    'price': date_df['price'].median()
                })
        
        if daily_prices:
            price_df = pd.DataFrame(daily_prices)
            fig.add_trace(
                go.Scatter(
                    x=price_df['date'],
                    y=price_df['price'],
                    mode='lines+markers',
                    name='价格趋势',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # 获取所有颜色并分配颜色
        all_colors = sorted(color_inventory_df['color'].unique())
        color_palette = px.colors.qualitative.Set3[:len(all_colors)]
        color_map = dict(zip(all_colors, color_palette))
        
        # 添加每天每个颜色的第一次库存（柱状图）
        for color in all_colors:
            color_data = color_inventory_df[color_inventory_df['color'] == color]
            fig.add_trace(
                go.Bar(
                    x=color_data['date'],
                    y=color_data['first_inventory'],
                    name=f'第一次-{color}',
                    marker_color=color_map[color],
                    offsetgroup='first',
                    legendgroup='first',
                    hovertemplate=f'颜色: {color}<br>日期: %{{x}}<br>第一次库存: %{{y}}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 添加每天每个颜色的最后一次库存（柱状图）
        for color in all_colors:
            color_data = color_inventory_df[color_inventory_df['color'] == color]
            fig.add_trace(
                go.Bar(
                    x=color_data['date'],
                    y=color_data['last_inventory'],
                    name=f'最后一次-{color}',
                    marker_color=color_map[color],
                    offsetgroup='last',
                    legendgroup='last',
                    hovertemplate=f'颜色: {color}<br>日期: %{{x}}<br>最后一次库存: %{{y}}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 添加每天第一次总库存
        fig.add_trace(
            go.Bar(
                x=total_inventory_df['date'],
                y=total_inventory_df['first_total'],
                name='第一次总库存',
                marker_color='lightblue',
                hovertemplate='日期: %{x}<br>第一次总库存: %{y}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # 添加每天最后一次总库存
        fig.add_trace(
            go.Bar(
                x=total_inventory_df['date'],
                y=total_inventory_df['last_total'],
                name='最后一次总库存',
                marker_color='lightcoral',
                hovertemplate='日期: %{x}<br>最后一次总库存: %{y}<extra></extra>'
            ),
            row=5, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title='每日颜色库存分析图表',
            height=1500,
            showlegend=True,
            barmode='group'
        )
        
        # 更新坐标轴
        fig.update_yaxes(title_text="价格 (€)", row=1, col=1)
        fig.update_yaxes(title_text="库存数量", row=2, col=1)
        fig.update_yaxes(title_text="库存数量", row=3, col=1)
        fig.update_yaxes(title_text="总库存", row=4, col=1)
        fig.update_yaxes(title_text="总库存", row=5, col=1)
        
        return fig
    
    def process_storage_data(filtered_df):
        """处理Storage库存数据"""
        first_storage_data = []
        second_storage_data = []
        
        for date in sorted(filtered_df['date'].unique()):
            date_df = filtered_df[filtered_df['date'] == date]
            
            # 第一次Storage数据
            first_prices = []
            for seller in date_df['seller'].unique():
                seller_df = date_df[date_df['seller'] == seller].sort_values('date_add_to_bag')
                if len(seller_df) > 0:
                    first_price_row = seller_df.iloc[0]
                    first_prices.append({
                        'price': first_price_row['price'],
                        'storage': first_price_row['storage'],
                        'seller': seller[:10] + '...' if len(str(seller)) > 10 else str(seller)
                    })
            
            if first_prices:
                prices = [item['price'] for item in first_prices]
                median_price = np.median(prices)
                closest_item = min(first_prices, key=lambda x: abs(x['price'] - median_price))
                
                first_storage_data.append({
                    'date': date,
                    'date_seller': str(date) + '+' + closest_item['seller'],
                    'storage': closest_item['storage'],
                    'storage_count': 1
                })
            
            # 第二次Storage数据
            second_prices = []
            for seller in date_df['seller'].unique():
                seller_df = date_df[date_df['seller'] == seller].sort_values('date_add_to_bag')
                if len(seller_df) >= 2:
                    second_price_row = seller_df.iloc[1]
                    second_prices.append({
                        'price': second_price_row['price'],
                        'storage': second_price_row['storage'],
                        'seller': seller[:10] + '...' if len(str(seller)) > 10 else str(seller)
                    })
                elif len(seller_df) == 1:
                    first_price_row = seller_df.iloc[0]
                    second_prices.append({
                        'price': first_price_row['price'],
                        'storage': first_price_row['storage'],
                        'seller': seller[:10] + '...' if len(str(seller)) > 10 else str(seller)
                    })
            
            if second_prices:
                prices = [item['price'] for item in second_prices]
                median_price = np.median(prices)
                closest_item = min(second_prices, key=lambda x: abs(x['price'] - median_price))
                
                second_storage_data.append({
                    'date': date,
                    'date_seller': str(date) + '+' + closest_item['seller'],
                    'storage': closest_item['storage'],
                    'storage_count': 1
                })
        
        return first_storage_data, second_storage_data
    
    def add_storage_charts(fig, first_storage_data, second_storage_data):
        """添加Storage库存图表"""
        if not first_storage_data and not second_storage_data:
            return
        
        first_storage_df = pd.DataFrame(first_storage_data) if first_storage_data else pd.DataFrame()
        second_storage_df = pd.DataFrame(second_storage_data) if second_storage_data else pd.DataFrame()
        
        # 为不同Storage类型分配颜色
        all_storage_types = set()
        if not first_storage_df.empty:
            all_storage_types.update(first_storage_df['storage'].unique())
        if not second_storage_df.empty:
            all_storage_types.update(second_storage_df['storage'].unique())
        
        storage_types = sorted(list(all_storage_types))
        colors = px.colors.qualitative.Set3[:len(storage_types)]
        storage_color_map = dict(zip(storage_types, colors))
        
        # 添加第一次Storage库存柱状图
        if not first_storage_df.empty:
            for storage_type in storage_types:
                storage_subset = first_storage_df[first_storage_df['storage'] == storage_type]
                if not storage_subset.empty:
                    fig.add_trace(
                        go.Bar(
                            x=storage_subset['date_seller'],
                            y=storage_subset['storage_count'],
                            name=f'第一次-{storage_type}',
                            marker_color=storage_color_map[storage_type],
                            hovertemplate=f'Storage: {storage_type}<br>日期+Seller: %{{x}}<br>中位数库存<extra></extra>'
                        ),
                        row=2, col=1
                    )
        
        # 添加第二次Storage库存柱状图
        if not second_storage_df.empty:
            for storage_type in storage_types:
                storage_subset = second_storage_df[second_storage_df['storage'] == storage_type]
                if not storage_subset.empty:
                    fig.add_trace(
                        go.Bar(
                            x=storage_subset['date_seller'],
                            y=storage_subset['storage_count'],
                            name=f'第二次-{storage_type}',
                            marker_color=storage_color_map[storage_type],
                            hovertemplate=f'Storage: {storage_type}<br>日期+Seller: %{{x}}<br>中位数库存<extra></extra>'
                        ),
                        row=3, col=1
                    )
    
    return app

# 在文件末尾添加以下代码
if __name__ == '__main__':
    import os
    
    # 准备数据
    try:
        df = prepare()
        if df is not None and not df.empty:
            app = create_integrated_dash_app(df)
            if app is not None:
                # 获取端口号，Render会提供PORT环境变量
                port = int(os.environ.get('PORT', 8050))
                
                # 启动应用，适配Render部署
                app.run(
                    host='0.0.0.0',  # 允许外部访问
                    port=port,        # 使用环境变量端口
                    debug=False       # 生产环境关闭调试模式
                )
            else:
                print("应用创建失败，请检查数据")
        else:
            print("数据准备失败或数据为空")
    except Exception as e:
        print(f"应用启动失败: {str(e)}")
