import dash
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os

# 全局筛选器状态
GLOBAL_FILTER_STATE = {
    'model': '全部',
    'memory': '全部', 
    'sim_type': '全部',
    'grade': '全部',
    'battery': '全部',
    'local': '全部'
}

def load_excel_data():
    """从处理好的文件中加载Excel数据"""
    try:
        # 读取处理好的Excel数据的两个工作表
        excel_file_path = ".\\processed_data\\excel_data.xlsx"
        
        # 读取讯强数据工作表
        xq_data = pd.read_excel(excel_file_path, sheet_name='讯强全新机')
        print(f"讯强数据行数: {len(xq_data)}")
        
        # 读取靓机汇数据工作表
        ljh_data = pd.read_excel(excel_file_path, sheet_name='靓机汇二手回收')
        print(f"靓机汇数据行数: {len(ljh_data)}")
        
        print(f"总数据行数: {len(xq_data) + len(ljh_data)}")
        
        return xq_data, ljh_data
    except Exception as e:
        print(f"读取处理后的Excel文件失败: {e}")
        return pd.DataFrame(), pd.DataFrame()

def process_memory_column(df, memory_col='内存'):
    """处理内存列（数据已预处理，保持函数接口一致）"""
    # 数据已经在process_and_export.py中处理过，这里只是保持接口一致
    return df

def load_ipad_data():
    """
    读取并处理bm_ipad数据截止至0707.xlsx，保留指定12列，按9列去重。
    nami为'纳米'时，将容量末尾加1，处理后删除nami列。
    返回处理后的DataFrame。
    """
    ipad_file = r".\processed_data\bm数据截止至0718.xlsx"
    use_cols = [
        '标题', '磨损中文', '容量', 'wifi类型', 'nami类型中文', '颜色中文', '价格处理后（欧元）', '国家', '日期',
        '商家名称', '商家增值税号', '商家存在年月'
    ]
    try:
        df = pd.read_excel(ipad_file, usecols=use_cols)
        # 处理nami列：为'纳米'时，容量末尾加1
        mask_nami = df['nami类型中文'] == '纳米'
        df.loc[mask_nami, '容量'] = df.loc[mask_nami, '容量'].astype(str) + '1'
        df = df.drop(columns=['nami类型中文'])
        # 转为int类型
        df['容量'] = df['容量'].astype(int)
        dedup_cols = ['标题', '磨损中文', '容量', 'wifi类型', '颜色中文', '价格处理后（欧元）', '国家', '日期']
        df = df.drop_duplicates(subset=dedup_cols, keep='first')
        df = df.dropna(subset=['标题', '价格处理后（欧元）', '日期'])
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])
        return df
    except Exception as e:
        print(f"读取iPad数据失败: {e}")
        return pd.DataFrame()

def create_integrated_dash_app(df):
    """
    创建集成的Plotly Dash交互式网页应用
    包含原有功能和新增Excel数据分析
    """
    # 加载Excel数据
    xq_data, ljh_data = load_excel_data()
    
    # 处理内存列
    if not xq_data.empty:
        xq_data = process_memory_column(xq_data)
    if not ljh_data.empty:
        ljh_data = process_memory_column(ljh_data)
    
    # 原有数据预处理
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
        print("原有数据为空，无法创建应用")
        return None
    
    # 获取原有数据筛选选项
    grade_options = [{'label': '全部', 'value': '全部'}] + [{'label': grade, 'value': grade} for grade in sorted(df['grade_name'].unique())]
    battery_options = [{'label': '全部', 'value': '全部'}] + [{'label': battery, 'value': battery} for battery in sorted(df['battery'].unique())]
    local_options = [{'label': '全部', 'value': '全部'}] + [{'label': local, 'value': local} for local in sorted(df['local'].unique())]
    
    # 统一的筛选选项生成函数
    def get_unified_options():
        # 统一的型号选项（合并所有数据源）
        all_models = set()
        if not df.empty:
            all_models.update(df['model'].dropna().unique())
        if not xq_data.empty:
            all_models.update(xq_data['型号'].dropna().unique())
        if not ljh_data.empty:
            all_models.update(ljh_data['型号'].dropna().unique())
        
        unified_model_options = [{'label': '全部', 'value': '全部'}] + \
                               [{'label': model, 'value': model} for model in sorted(all_models)]
        
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
                             [{'label': sim_type, 'value': sim_type} for sim_type in sorted(all_sim_types)]
        
        return unified_model_options, unified_memory_options, unified_sim_options
    
    # 获取统一的筛选选项
    unified_model_options, unified_memory_options, unified_sim_options = get_unified_options()
    
    # 创建Dash应用
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # 定义应用布局
    app.layout = html.Div([
        html.H1("数据分析系统", style={'textAlign': 'center', 'marginBottom': 30}),
        
        # 标签页选择器
        dcc.Tabs(id='main-tabs', value='original-analysis', children=[
            dcc.Tab(label='每日颜色库存分析', value='original-analysis'),
            dcc.Tab(label='讯强全新机分析', value='xq-analysis'),
            dcc.Tab(label='靓机汇二手回收分析', value='ljh-analysis'),
            dcc.Tab(label='iPad箱型图分析', value='ipad-analysis'),  # 新增Tab
        ], style={'marginBottom': 20}),
        
        # 动态筛选控件区域
        html.Div(
            id='filter-controls',
            children=[]  # 确保有初始的children属性
        ),
        
        # 图表显示区域
        html.Div(id='chart-content'),
        
        # 虚拟输出元素（隐藏）
        html.Div(id='dummy-output', style={'display': 'none'})
    ])
    
    # 筛选控件回调，保持状态
    @app.callback(
        Output('filter-controls', 'children'),
        [Input('main-tabs', 'value')]
    )
    def update_filter_controls(selected_tab):
        # 使用全局状态保持所有筛选器值
        model_value = GLOBAL_FILTER_STATE['model']
        memory_value = GLOBAL_FILTER_STATE['memory']
        sim_value = GLOBAL_FILTER_STATE['sim_type']
        grade_value = GLOBAL_FILTER_STATE['grade']  # 从全局状态获取
        battery_value = GLOBAL_FILTER_STATE['battery']  # 从全局状态获取
        local_value = GLOBAL_FILTER_STATE['local']  # 从全局状态获取
        
        # 统一的筛选器布局（所有模块都使用相同的字段）
        common_filters = html.Div([
            html.Div([
                html.Div([
                    html.Label("型号:"),
                    dcc.Dropdown(
                        id='unified-model-dropdown',
                        options=unified_model_options,
                        value=model_value,
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': 20}),
                
                html.Div([
                    html.Label("内存:"),
                    dcc.Dropdown(
                        id='unified-memory-dropdown',
                        options=unified_memory_options,
                        value=memory_value,
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': 20}),
                
                html.Div([
                    html.Label("SIM类型:"),
                    dcc.Dropdown(
                        id='unified-sim-dropdown',
                        options=unified_sim_options,
                        value=sim_value,
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': 20}),
            ], style={'marginBottom': 20, 'textAlign': 'center'}),
        ])
        
        # 额外筛选器（始终创建，但根据标签页显示/隐藏）
        additional_filters_style = {'marginBottom': 30, 'textAlign': 'center'} if selected_tab == 'original-analysis' else {'display': 'none'}
        
        additional_filters = html.Div([
            html.Div([
                html.Label("磨损:"),
                dcc.Dropdown(
                    id='grade-dropdown',
                    options=grade_options,
                    value=grade_value,
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("电池:"),
                dcc.Dropdown(
                    id='battery-dropdown',
                    options=battery_options,
                    value=battery_value,
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("地区:"),
                dcc.Dropdown(
                    id='local-dropdown',
                    options=local_options,
                    value=local_value,
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
        ], style=additional_filters_style)
        
        # iPad箱型图分析Tab也用同样的筛选器（不包含颜色）
        if selected_tab == 'ipad-analysis':
            # 获取iPad数据，以便提取国家和磨损选项
            ipad_df = load_ipad_data()
            # 国家选项
            country_options = [{'label': '全部', 'value': '全部'}]
            if not ipad_df.empty and '国家' in ipad_df.columns:
                country_options += [
                    {'label': str(x), 'value': str(x)} for x in sorted(ipad_df['国家'].dropna().unique())
                ]
            # 磨损选项
            grade_options_ipad = [{'label': '全部', 'value': '全部'}]
            if not ipad_df.empty and '磨损中文' in ipad_df.columns:
                grade_options_ipad += [
                    {'label': str(x), 'value': str(x)} for x in sorted(ipad_df['磨损中文'].dropna().unique())
                ]
            # 第一行：型号、内存、SIM类型
            first_row = html.Div([
                html.Label("型号:"),
                dcc.Dropdown(
                    id='unified-model-dropdown',
                    options=unified_model_options,
                    value=model_value,
                    style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}
                ),
                html.Label("内存:"),
                dcc.Dropdown(
                    id='unified-memory-dropdown',
                    options=unified_memory_options,
                    value=memory_value,
                    style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}
                ),
                html.Label("SIM类型:"),
                dcc.Dropdown(
                    id='unified-sim-dropdown',
                    options=unified_sim_options,
                    value=sim_value,
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center', 'marginBottom': 10})
            # 第二行：磨损、国家
            second_row = html.Div([
                html.Label("磨损:"),
                dcc.Dropdown(
                    id='ipad-grade-dropdown',
                    options=grade_options_ipad,
                    value='全部',
                    style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}
                ),
                html.Label("国家:"),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=country_options,
                    value='全部',
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'textAlign': 'center', 'marginBottom': 10, 'display': 'flex', 'justifyContent': 'center', 'gap': '20px'})
            return html.Div([first_row, second_row])
        return html.Div([common_filters, additional_filters])
    

    
    # 统一的筛选器状态更新回调（包含所有筛选器）
    @app.callback(
        Output('dummy-output', 'children', allow_duplicate=True),
        [Input('unified-model-dropdown', 'value'),
         Input('unified-memory-dropdown', 'value'),
         Input('unified-sim-dropdown', 'value')],
        [State('grade-dropdown', 'value'),
         State('battery-dropdown', 'value'),
         State('local-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_all_filter_state(model_val, memory_val, sim_val, grade_val, battery_val, local_val):
        # 更新全局筛选器状态
        if model_val is not None:
            GLOBAL_FILTER_STATE['model'] = model_val
        if memory_val is not None:
            GLOBAL_FILTER_STATE['memory'] = memory_val
        if sim_val is not None:
            GLOBAL_FILTER_STATE['sim_type'] = sim_val
        if grade_val is not None:
            GLOBAL_FILTER_STATE['grade'] = grade_val
        if battery_val is not None:
            GLOBAL_FILTER_STATE['battery'] = battery_val
        if local_val is not None:
            GLOBAL_FILTER_STATE['local'] = local_val
        return ''
    
    # 第一个回调：处理original-analysis标签页（包含所有筛选器）
    @app.callback(
        Output('chart-content', 'children'),
        [Input('main-tabs', 'value'),
         Input('unified-model-dropdown', 'value'),
         Input('unified-memory-dropdown', 'value'),
         Input('unified-sim-dropdown', 'value'),
         Input('grade-dropdown', 'value'),
         Input('battery-dropdown', 'value'),
         Input('local-dropdown', 'value')],
        prevent_initial_call=False
    )
    def update_original_analysis_chart(selected_tab, model_val, memory_val, sim_val, grade_val, battery_val, local_val):
        # 只处理 original-analysis 标签页
        if selected_tab != 'original-analysis':
            raise PreventUpdate
        
        # 设置默认值
        model_val = model_val or '全部'
        memory_val = memory_val or '全部'
        sim_val = sim_val or '全部'
        grade_val = grade_val or '全部'
        battery_val = battery_val or '全部'
        local_val = local_val or '全部'
        
        try:
            # 筛选原始数据
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
            
            return dcc.Graph(figure=create_daily_color_analysis_chart(filtered_df))
        
        except Exception as e:
            print(f"原始分析图表更新错误: {e}")
            return dcc.Graph(figure=create_daily_color_analysis_chart(df))
    
    # 第二个回调：处理其他标签页（只使用基础筛选器）
    @app.callback(
        Output('chart-content', 'children', allow_duplicate=True),
        [Input('main-tabs', 'value'),
         Input('unified-model-dropdown', 'value'),
         Input('unified-memory-dropdown', 'value'),
         Input('unified-sim-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_other_tabs_chart(selected_tab, model_val, memory_val, sim_val):
        # 只处理其他标签页
        if selected_tab == 'original-analysis':
            raise PreventUpdate
        
        # 设置默认值
        model_val = model_val or '全部'
        memory_val = memory_val or '全部'
        sim_val = sim_val or '全部'
        
        try:
            if selected_tab == 'xq-analysis':
                # 筛选讯强数据
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
                # 筛选靓机汇数据
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
            print(f"其他分析图表更新错误: {e}")
            if selected_tab == 'xq-analysis':
                return html.Div([
                    dcc.Graph(figure=create_xq_analysis_chart(xq_data)),
                    dcc.Graph(figure=create_xq_color_price_chart(xq_data))
                ])
            elif selected_tab == 'ljh-analysis':
                return dcc.Graph(figure=create_ljh_analysis_chart(ljh_data))
        
        # 默认返回空图表
        return dcc.Graph(figure=go.Figure())
    
    @app.callback(
        Output('chart-content', 'children', allow_duplicate=True),
        [
            Input('main-tabs', 'value'),
            Input('unified-model-dropdown', 'value'),
            Input('unified-memory-dropdown', 'value'),
            Input('unified-sim-dropdown', 'value'),
            Input('ipad-grade-dropdown', 'value'),
            Input('battery-dropdown', 'value'),
            Input('country-dropdown', 'value')
        ],
        prevent_initial_call=True
    )
    def update_ipad_box_chart(selected_tab, model_val, memory_val, sim_val, grade_val, battery_val, country_val):
        if selected_tab != 'ipad-analysis':
            raise PreventUpdate
        df = load_ipad_data()
        if model_val and model_val != '全部':
            df = df[df['标题'] == model_val]
        if grade_val and grade_val != '全部':
            df = df[df['磨损中文'] == grade_val]
        if memory_val and memory_val != '全部':
            try:
                df = df[df['容量'] == int(memory_val)]
            except Exception:
                df = df[df['容量'] == memory_val]
        if sim_val and sim_val != '全部':
            df = df[df['wifi类型'] == sim_val]
        if battery_val and battery_val != '全部':
            # iPad数据没有电池字段，跳过
            pass
        if country_val and country_val != '全部':
            df = df[df['国家'] == country_val]
        fig = plot_ipad_box(df)
        return dcc.Graph(figure=fig)
    
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
        
        # 获取数值列（假设第5列开始是价格数据）
        price_columns = filtered_data.columns[4:]  # 从第5列开始
        
        # 只保留有数据的日期，且格式为月-日
        date_list = sorted(filtered_data['日期'].dt.strftime('%m-%d').unique())
        
        # 按日期分组，收集每日所有价格数据（忽略颜色）
        daily_price_data = []
        all_prices_for_range = []  # 用于计算价格范围
        
        for date_str in date_list:
            date_data = filtered_data[filtered_data['日期'].dt.strftime('%m-%d') == date_str]
            # 收集当天所有价格数据
            all_prices = []
            if len(date_data) > 0:
                for col in price_columns:
                    prices = pd.to_numeric(date_data[col], errors='coerce').dropna()
                    # 添加价格验证，过滤异常值
                    valid_prices = prices[(prices > 0) & (prices < 10000)]  # 假设合理价格范围是0-10000
                    all_prices.extend(valid_prices.tolist())
            daily_price_data.append({
                'date': date_str,
                'prices': all_prices if all_prices else []
            })
            if all_prices:
                all_prices_for_range.extend(all_prices)
        
        if not all_prices_for_range:
            fig = go.Figure()
            fig.add_annotation(
                text="没有足够的数据生成图表",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 创建箱型图
        fig = go.Figure()
        
        for day_data in daily_price_data:
            if day_data['prices']:  # 有数据的日期
                
                # 计算统计信息
                prices = day_data['prices']
                min_price = min(prices)
                max_price = max(prices)
                median_price = sorted(prices)[len(prices)//2] if len(prices) % 2 == 1 else (sorted(prices)[len(prices)//2-1] + sorted(prices)[len(prices)//2]) / 2
                
                # 添加箱型图
                fig.add_trace(go.Box(
                    x=[day_data['date']] * len(day_data['prices']),
                    y=day_data['prices'],
                    name=day_data['date'],
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8,
                    hovertemplate=(
                        "日期: %{x}<br>"
                        "最大值: %{q3}<br>"
                        "上四分位数: %{q3}<br>"
                        "中位数: %{median}<br>"
                        "下四分位数: %{q1}<br>"
                        "最小值: %{lowerfence}<br>"
                        "数据点数: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    customdata=[len(day_data['prices'])]
                ))
                
                # 如果所有价格相同，添加一个透明的散点图来提供hover
                if min_price == max_price:
                    fig.add_trace(go.Scatter(
                        x=[day_data['date']],
                        y=[min_price],
                        mode='markers',
                        marker=dict(size=15, opacity=0, color='rgba(0,0,0,0)'),  # 使用rgba透明色
                        name=day_data['date'] + '_hover',
                        showlegend=False,
                        hovertemplate=(
                            "日期: %{x}<br>"
                            "价格: " + str(min_price) + "<br>"
                            "数据点数: " + str(len(prices)) + "<br>"
                            "<extra></extra>"
                        )
                    ))
            else:  # 没有数据的日期，添加空的占位符
                fig.add_trace(go.Scatter(
                    x=[day_data['date']],
                    y=[None],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    name=day_data['date'],
                    showlegend=False,
                    hovertemplate="日期: %{x}<br>无数据<extra></extra>"
                ))
        
        # 计算价格范围和y轴刻度
        if all_prices_for_range:
            min_price = min(all_prices_for_range)
            max_price = max(all_prices_for_range)
            price_range = max_price - min_price
            
            # 添加范围检查，防止内存溢出
            if price_range > 50000:  # 如果价格范围过大，使用简化的刻度
                tick_vals = None
                y_min = None
                y_max = None
            else:
                # 设置合适的间隔，确保整数显示
                if price_range <= 100:
                    tick_interval = 10
                elif price_range <= 500:
                    tick_interval = 50
                elif price_range <= 1000:
                    tick_interval = 100
                elif price_range <= 5000:
                    tick_interval = 500
                else:
                    tick_interval = 1000
                
                # 计算y轴范围
                y_min = int(min_price // tick_interval) * tick_interval
                y_max = int((max_price // tick_interval) + 1) * tick_interval
                
                # 添加额外检查，确保刻度数量合理
                tick_count = (y_max - y_min) // tick_interval
                if tick_count > 1000:  # 如果刻度数量过多，使用自动刻度
                    tick_vals = None
                    y_min = None
                    y_max = None
                else:
                    # 生成刻度值
                    tick_vals = list(range(y_min, y_max + tick_interval, tick_interval))
        else:
            tick_vals = None
            y_min = None
            y_max = None
        
        fig.update_layout(
            title='讯强全新机每日价格箱型图分析',
            xaxis_title='日期',
            yaxis_title='价格',
            height=400,
            showlegend=False,
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

    def create_xq_color_price_chart(filtered_data):
        """创建讯强全新机各颜色价格折线图"""
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
        
        # 获取第四列作为颜色分组依据
        color_column = filtered_data.columns[3]  # 第四列（索引为3）
        
        # 确保价格列为数值类型
        price_column = filtered_data.columns[4]  # 第五列是价格
        filtered_data[price_column] = pd.to_numeric(filtered_data[price_column], errors='coerce')
        
        # 过滤有效价格数据
        valid_data = filtered_data[
            (filtered_data[price_column] > 0) & 
            (filtered_data[price_column] < 20000) & 
            (filtered_data[price_column].notna())
        ]
        
        if valid_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="没有有效的价格数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # 创建图表
        fig = go.Figure()
        
        # 定义颜色映射
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 获取第四列的所有唯一值作为分组依据
        unique_colors = valid_data[color_column].unique()
        
        # 为每个颜色创建一条线
        for i, color_name in enumerate(unique_colors):
            # 获取该颜色的所有数据
            color_data = valid_data[valid_data[color_column] == color_name].copy()
            
            if len(color_data) > 0:
                # 按日期排序
                color_data = color_data.sort_values('日期')
                
                # 转换日期为字符串格式
                date_strings = color_data['日期'].dt.strftime('%m-%d').tolist()
                
                # 添加线条
                fig.add_trace(go.Scatter(
                    x=date_strings,
                    y=color_data[price_column],
                    mode='lines+markers',
                    name=str(color_name),
                    line=dict(color=color_palette[i % len(color_palette)], width=2),
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
                categoryorder='category ascending'
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
            
            # 确保price列是数值类型
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
            date_list = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in df_copy['date'].dt.date.unique()]
            date_ticktext = [pd.to_datetime(date).strftime('%m-%d') for date in df_copy['date'].dt.date.unique()]
            for date in df_copy['date'].dt.date.unique(): 
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
                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'), 
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
                df_copy['model'].astype(str) + '_'+ df_copy['grade_name'].astype(str) + '_' + df_copy['battery'].astype(str) + '_' + df_copy['storage'].astype(str) + '_' + df_copy['sim_type'].astype(str) + '_' + df_copy['local'].astype(str) + '_' + df_copy['color'].astype(str)
            )
            # 按日期分组处理库存数据
            daily_color_inventory = []
            daily_total_inventory = []
            for date in df_copy['date'].dt.date.unique():
                date_data = df_copy[df_copy['date'].dt.date == date].copy()
                if len(date_data) == 0:
                    continue
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
                    first_seller = group['seller'].iloc[0] if 'seller' in group.columns else '未知'
                    last_seller = group['seller'].iloc[-1] if 'seller' in group.columns else '未知'
                    first_price = group['price'].iloc[0] if 'price' in group.columns else 0
                    last_price = group['price'].iloc[-1] if 'price' in group.columns else 0
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
                    daily_first_color[color] = daily_first_color.get(color, 0) + first_inventory
                    daily_last_color[color] = daily_last_color.get(color, 0) + last_inventory
                    if color not in daily_first_sellers:
                        daily_first_sellers[color] = []
                    if color not in daily_last_sellers:
                        daily_last_sellers[color] = []
                    daily_first_sellers[color].append(first_seller)
                    daily_last_sellers[color].append(last_seller)
                    if color not in daily_first_prices:
                        daily_first_prices[color] = []
                    if color not in daily_last_prices:
                        daily_last_prices[color] = []
                    daily_first_prices[color].append(first_price)
                    daily_last_prices[color].append(last_price)
                    daily_first_total += first_inventory
                    daily_last_total += last_inventory
                for color in set(list(daily_first_color.keys()) + list(daily_last_color.keys())):
                    first_sellers_list = list(set(daily_first_sellers.get(color, [])))
                    last_sellers_list = list(set(daily_last_sellers.get(color, [])))
                    first_prices_list = daily_first_prices.get(color, [])
                    last_prices_list = daily_last_prices.get(color, [])
                    first_avg_price = sum(first_prices_list) / len(first_prices_list) if first_prices_list else 0
                    last_avg_price = sum(last_prices_list) / len(last_prices_list) if last_prices_list else 0
                    daily_color_inventory.append({
                        'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                        'color': color,
                        'first_inventory': daily_first_color.get(color, 0),
                        'last_inventory': daily_last_color.get(color, 0),
                        'first_sellers': ', '.join(first_sellers_list),
                        'last_sellers': ', '.join(last_sellers_list),
                        'first_avg_price': first_avg_price,
                        'last_avg_price': last_avg_price
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
            
            # ===== 更新布局，xaxis加ticktext =====
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
                ),
                xaxis5=dict(
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
    
    def plot_ipad_box(df):
        """
        绘制iPad每日价格箱型图，鼠标悬停只显示最高、最低、均值、中位数。
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="无数据", xref="paper", yref="paper", x=0.5, y=0.5, xanchor='center', yanchor='middle', showarrow=False, font=dict(size=20))
            return fig

        df['date_str'] = df['日期'].dt.strftime('%m-%d')
        date_list = sorted(df['date_str'].unique())

        fig = go.Figure()
        for date in date_list:
            prices = df[df['date_str'] == date]['价格处理后（欧元）']
            if len(prices) > 0:
                max_val = prices.max()
                min_val = prices.min()
                median_val = prices.median()
                mean_val = prices.mean()
                # 1. 画箱型图，不显示hover
                fig.add_trace(go.Box(
                    x=[date]*len(prices),
                    y=prices,
                    name=date,
                    boxpoints=False,
                    hoverinfo='skip',  # 不显示默认hover
                    marker=dict(color='lightblue')
                ))
                # 2. 画透明散点，hover显示自定义内容
                fig.add_trace(go.Scatter(
                    x=[date],
                    y=[mean_val],
                    mode='markers',
                    marker=dict(size=18, color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hovertemplate=(
                        f'日期: {date}<br>'
                        f'最高价: {max_val:.2f}€<br>'
                        f'最低价: {min_val:.2f}€<br>'
                        f'均值: {mean_val:.2f}€<br>'
                        f'中位数: {median_val:.2f}€'
                        '<extra></extra>'
                    )
                ))
        fig.update_layout(
            title='iPad每日价格箱型图',
            xaxis_title='日期',
            yaxis_title='价格（欧元）',
            height=500,
            showlegend=False,
            xaxis=dict(type='category', categoryorder='category ascending'),
            yaxis=dict(tickformat=',.0f')
        )
        return fig
    
    return app

# 模块级别定义 app 和 server
# 改为直接读取处理好的原始数据文件
processed_df = pd.read_excel(".\\processed_data\\original_data.xlsx")
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
