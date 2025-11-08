pip install -r requirements.txt
python scripts/build_features.py

region only for evaluation, not for training models



global_distribution.png — 全球散点
hotspots_density.png — 密度热点
lat_distribution.png — 纬度直方图
lon_distribution.png — 经度直方图
hemisphere_distribution.png — 南北半球分布



taxon_id (int, 仅 train)：物种标签

lat (float)、lon (float)：原始坐标（用于评估或可视化）

grid_lat (int)、grid\_lon (int)：1° 栅格索引

region (str)："{center_lat}_{center_lon}"（仅用于分组）

hemisphere (category)：N/S

sin_lon (float)、cos_lon (float)：经度周期编码

taxon_name (str)（映射用，不训练）

