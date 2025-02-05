import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('E:/Acoustic_lev/Astar_3d/3D_PathPlanner-master/all_paths.csv')  # Path_ID, X, Y, Z

fig = go.Figure()

# 按 Path_ID 分组
for path_id, group in df.groupby('Path_ID'):
    fig.add_trace(
        go.Scatter3d(
            x=group['X'],
            y=group['Y'],
            z=group['Z'],
            mode='lines+markers',        # 同时画线和点
            name=f'Agent {path_id}',     # 图例名称
            marker=dict(size=4)         # 调整点的大小
        )
    )

# 也可以设置坐标轴名称、背景等
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),
    width=800,  # 图大小
    height=600,
    legend_title_text='Agents'
)

fig.show()
