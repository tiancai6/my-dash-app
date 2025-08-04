import pandas as pd
import seaborn as sns

# 1. 读取数据
df = pd.read_excel('./processed_data/original_data.xlsx')

# 2. 获取唯一 color 值
unique_colors = sorted(df['color'].dropna().unique())

# 3. 自动分配 RGB 颜色
palette = sns.color_palette('hls', len(unique_colors))
color_to_rgb = {
    color: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    for color, (r, g, b) in zip(unique_colors, palette)
}

# 4. 保存为 py 文件
with open('color_to_rgb_map.py', 'w', encoding='utf-8') as f:
    f.write('color_to_rgb = {\n')
    for color, rgb in color_to_rgb.items():
        f.write(f"    {repr(color)}: '{rgb}',\n")
    f.write('}\n')

print(f"已生成 {len(color_to_rgb)} 个颜色映射，并保存到 color_to_rgb_map.py")